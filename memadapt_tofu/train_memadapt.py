"""Gradient-masked training of the memory adapter (stage S5; --smoke for S2).

Recipe: paper-fixed (15 epochs, lr 1e-2, memory values only) + pinned defaults
(AdamW, effective batch 32, constant LR, no warmup, weight_decay 0 — OU's
wd=0.01 override would silently decay idle sources' entries and is explicitly
NOT inherited). Data pipeline is OU-parity (see data_tofu.py) so the resulting
model is comparable with the released Finetuned checkpoint.

--smoke: end-to-end micro run on 1 GPU at FULL table size (profile 2 authors
-> assign -> 5 train steps -> save -> reload -> bitwise parity), printing
max_memory_allocated. Gate before any full submission.
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from data_tofu import QACollatorWithSources, TofuQADataset
from memadapt_common import (
    RECORDS_PER_AUTHOR,
    file_sha256,
    load_config,
    save_json,
    set_determinism,
    slurm_job_id,
)
from memadapt_model import freeze_base, install_adapter, save_checkpoint
from memory_layer import ProductKeyMemory


class MemAdaptTrainer(Trainer):
    """Routes per-sequence source ids to the adapter and (optionally) asserts
    gradient isolation for the first few optimizer steps."""

    def __init__(self, *args, adapter=None, debug_grad_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = adapter
        self.debug_grad_steps = debug_grad_steps
        mem = adapter.memory
        owner_compact = torch.full((mem.values.shape[0],), -1, dtype=torch.long)
        assigned = (mem.owner_full >= 0).nonzero(as_tuple=True)[0]
        owner_compact[mem.remap[assigned].cpu()] = mem.owner_full[assigned].cpu()
        self._owner_compact = owner_compact

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        source_ids = inputs.pop("source_ids")
        inputs.pop("index", None)
        self.adapter.set_batch_sources(source_ids)
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            self.adapter.clear()

    def training_step(self, model, inputs, num_items_in_batch=None):
        sources = inputs["source_ids"].detach().cpu()
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step < self.debug_grad_steps:
            # Accumulated grads span several micro-batches' sources; the
            # per-batch assert is only sound without accumulation.
            if self.args.gradient_accumulation_steps == 1:
                self._assert_grad_isolation(sources)
            elif self.state.global_step == 0:
                print("[grad-check] skipped: gradient_accumulation_steps > 1")
        return loss

    def _assert_grad_isolation(self, sources: torch.Tensor):
        grad = self.adapter.memory.values.grad
        assert grad is not None, "no gradient on memory values"
        nz = grad.abs().sum(dim=1).nonzero(as_tuple=True)[0].cpu()
        owners = self._owner_compact[nz]
        assert torch.isin(owners, sources.unique()).all(), (
            f"gradient leaked outside batch sources {sources.unique().tolist()}: "
            f"touched owners {owners.unique().tolist()}"
        )
        assert grad[-1].abs().sum().item() == 0, "pad row received gradient"
        print(f"[grad-check] step {self.state.global_step}: "
              f"{nz.numel()} rows touched, owners ⊆ batch sources ✓")


class RoutingTelemetry(TrainerCallback):
    """Epoch-end routing stats on a fixed probe batch. own_mass should be
    near-constant across epochs (routing below the adapter is frozen; bf16
    padding/kernel ulps allow small wobble — treat large drift, not ulp-level
    noise, as the alarm). cross_source_mass is the leakage early-warning."""

    def __init__(self, adapter, probe_batch, layer_idx, model):
        self.adapter = adapter
        self.probe = probe_batch
        self.layer_idx = layer_idx
        self.model = model
        self.history = []

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, **kwargs):
        model = self.model
        device = next(model.parameters()).device
        captured = {}
        layer = model.model.layers[self.layer_idx]
        hook = layer.mlp.mlp.register_forward_pre_hook(
            lambda module, a: captured.__setitem__("x", a[0])
        )
        was_training = model.training
        model.eval()
        try:
            model(
                input_ids=self.probe["input_ids"].to(device),
                attention_mask=self.probe["attention_mask"].to(device),
                use_cache=False,
            )
        finally:
            hook.remove()
            if was_training:
                model.train()
        stats = self.adapter.memory.routing_stats(
            captured["x"],
            self.probe["source_ids"].to(device),
            self.probe["attention_mask"].to(device),
        )
        stats["epoch"] = state.epoch
        stats["values_norm"] = self.adapter.memory.values.norm().item()
        self.history.append(stats)
        print(f"[telemetry] {json.dumps(stats)}")


def build_model_and_memory(cfg, assignment):
    a = cfg["adapter"]
    memory = ProductKeyMemory(
        hidden=a["hidden"], n_sqrt=a["mem_size_sqrt"], key_dim=a["key_dim"],
        topk=a["topk"], half_topk=a["half_topk"], value_dim=a["value_dim"],
        router_seed=a["router_seed"],
        key_scale=a.get("key_scale", 1.0),
    )
    memory.load_assignment(assignment["assigned_idx"], assignment["owner"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    adapter = install_adapter(model, memory, a["layer_idx"])
    freeze_base(model, memory)
    return model, memory, adapter


def run_training(cfg, args):
    set_determinism(cfg["seed"])
    run_dir = cfg["output_dir"]
    os.makedirs(run_dir, exist_ok=True)

    assignment_path = args.assignment or os.path.join(
        run_dir, "assignment", "assignment.pt"
    )
    assignment = torch.load(assignment_path, map_location="cpu",
                            weights_only=False)
    assert assignment["adapter_cfg"]["router_seed"] == cfg["adapter"]["router_seed"], (
        "assignment was profiled with a different router"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = TofuQADataset(tokenizer, split=cfg["data"]["split"],
                            max_length=cfg["data"]["max_length"])
    limit_authors = cfg["data"].get("limit_authors")
    if limit_authors:
        dataset = Subset(dataset, range(limit_authors * RECORDS_PER_AUTHOR))

    model, memory, adapter = build_model_and_memory(cfg, assignment)

    t = cfg["train"]
    targs = TrainingArguments(
        output_dir=os.path.join(run_dir, "hf_trainer"),
        num_train_epochs=t["epochs"],
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=t["lr"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        optim=t["optim"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    collator = QACollatorWithSources(tokenizer)
    probe_rows = [dataset[i] for i in range(0, min(len(dataset), 64), 1)]
    telemetry = RoutingTelemetry(
        adapter, collator(probe_rows), cfg["adapter"]["layer_idx"], model
    )

    trainer = MemAdaptTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
        adapter=adapter,
        debug_grad_steps=args.debug_grad_checks,
        callbacks=[telemetry],
    )

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    adapter_cfg = dict(cfg["adapter"])
    meta = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "assignment_path": assignment_path,
        "assignment_sha": assignment["sha"],
        "train_wall_seconds": wall,
        "log_history": trainer.state.log_history,
        "routing_telemetry": telemetry.history,
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "seed": cfg["seed"],
        "torch_version": torch.__version__,
    }
    save_checkpoint(memory, adapter_cfg, run_dir, extra_meta=meta)
    if torch.cuda.is_available():
        print(f"[mem] max_memory_allocated="
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"[done] wall={wall:.1f}s checkpoint={run_dir}")
    return run_dir, memory


def run_smoke(cfg, args):
    """S2: profile -> assign -> 5 steps -> save -> reload -> parity, full-size N."""
    from assign_entries import greedy_assign, profile_accesses

    from memadapt_common import assignment_sha
    from memadapt_model import load_memory_from_checkpoint

    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["data"]["limit_authors"] = 2
    cfg["output_dir"] = cfg["output_dir"].rstrip("/") + "_smoke"
    cfg["train"]["batch_size"] = 2
    args.max_steps = 5
    args.debug_grad_checks = 5

    print("[smoke] profiling 2 authors at full table size")
    counts = profile_accesses(cfg, device="cuda")
    assigned_idx, owner, fills = greedy_assign(
        counts, cfg["assignment"]["entries_per_source"], fill_seed=cfg["seed"]
    )
    out_dir = os.path.join(cfg["output_dir"], "assignment")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"assigned_idx": assigned_idx, "owner": owner,
         "sha": assignment_sha(assigned_idx, owner),
         "adapter_cfg": cfg["adapter"],
         "entries_per_source": cfg["assignment"]["entries_per_source"]},
        os.path.join(out_dir, "assignment.pt"),
    )
    print(f"[smoke] assigned {assigned_idx.numel()} entries, fallback={fills}")

    run_dir, memory_live = run_training(cfg, args)

    print("[smoke] reload + forward parity check (live trained vs reloaded)")
    memory2 = load_memory_from_checkpoint(run_dir).cuda()
    x = torch.randn(2, 16, cfg["adapter"]["hidden"], device="cuda")
    with torch.no_grad():
        assert torch.equal(memory_live.cuda()(x), memory2(x)), (
            "reloaded checkpoint does not reproduce the trained memory"
        )
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--assignment", default=None,
                    help="override path to assignment.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--debug_grad_checks", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])

    if args.smoke:
        run_smoke(cfg, args)
    else:
        run_training(cfg, args)


if __name__ == "__main__":
    main()
