"""Train the MemSinks/SeqTD-on-TOFU arm (or its module-matched LoRA control).

Config-driven (configs/*.json). substrate:
  "lora_delta" — sequence-tied masking of the MLP gate/up LoRA deltas (M1)
  "none"       — identical recipe/modules, no masks (CTRL-L)

Uses plain transformers.Trainer, NOT trl SFTTrainer: trl 0.9.6's dataset prep
tokenizes with remove_columns=dataset.column_names, silently dropping the
author_id column — the run would degenerate to plain LoRA while every number
stays plausible. QACollator reproduces the SISA tokenization path exactly
(parity unit-tested in test_memsinks.py) and carries author_ids.

H4 manipulation-check probe: per epoch, on a fixed probe subset, answer
probability under (i) the author's own training mask vs (ii) the author's
sinks deleted. If this gap never opens by the last epoch, the mechanism did
not bind and downstream forget-quality numbers are uninterpretable (gate:
stop and review, do not launch more arms).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))
from train_lora_shard import format_prompt  # single source of truth for the prompt format


def freeze_lora_a_irp(model, irp_seed, shard_id=0, std=1.0):
    """CUDA-safe port of train_lora_shard.apply_irp_projections (E3 strict arm).

    Identical per-(shard, layer, adapter) SHA-256 seeding, but the seeded normal is
    drawn on CPU and copied to the weight's device/dtype — the original calls
    nn.init.normal_(cuda_weight, generator=cpu_gen), which raises on GPU (bit us:
    job 443551). Bit-equivalence with the original on CPU is gate-tested (std=1.0).

    std: scalar, or "auto" = 1/sqrt(fan_in) per layer (LoRA-scale init). std=1.0 is
    the original IRP behavior and DIVERGED in the strict arm (job 443563: ~45x the
    kaiming scale, x rslora ~11.3, loss 2.7 -> 8-12) — H14' uses "auto"."""
    import hashlib as _h
    import math as _m
    for layer_name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        for adapter_key, linear in module.lora_A.items():
            seed_bytes = _h.sha256(
                f"{irp_seed}:{shard_id}:{layer_name}:{adapter_key}".encode()).digest()
            seed_int = int.from_bytes(seed_bytes[:4], "little")
            gen = torch.Generator()
            gen.manual_seed(seed_int)
            layer_std = (1.0 / _m.sqrt(linear.weight.shape[1])) if std == "auto" else float(std)
            w = torch.empty(linear.weight.shape)
            torch.nn.init.normal_(w, mean=0.0, std=layer_std, generator=gen)
            with torch.no_grad():
                linear.weight.copy_(w.to(device=linear.weight.device, dtype=linear.weight.dtype))
            linear.weight.requires_grad_(False)

import masks as M
from memsinks_model import (
    MaskState,
    author_delete_vector,
    author_serve_vector,
    install_sink_hooks,
    save_masks,
)


# ── site-path expansion (added on export) ────────────────────────────────────────────────────
# Configs used to carry absolute /storage2 paths. They now say "${TOFU_CKPT_ROOT}/..." etc, and
# this resolves them at load time, hard-erroring on an unset variable rather than writing a
# literal "${TOFU_CKPT_ROOT}" directory to disk (which is what happened before the guard).
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import expand_paths as _expand_site_paths, ensure_site_env as _ensure_site_env
except ImportError:                       # repo_env.py is at the repo root; absent => no-op
    def _expand_site_paths(o, _k=""): return o
    def _ensure_site_env(force=False): return {}


PROBE_AUTHORS_DEFAULT = [0, 50, 100, 150, 180, 190, 199]
PROBE_ROWS_PER_AUTHOR = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="2-step batch-1 micro-train to <output_dir>_smoke/ (pipeline gate)")
    return p.parse_args()


def load_config(path):
    _ensure_site_env()
    with open(path) as f:
        return _expand_site_paths(json.load(f))


def build_dataset(hf_home):
    os.environ["HF_HOME"] = hf_home
    full = load_dataset("locuslab/TOFU", "full")["train"]
    # author index 0-199 (dataset row // 20); the hash scheme maps index -> seq_id index+1
    return full.map(lambda e, i: {"author_id": i // 20}, with_indices=True)


class QACollator:
    """SISA-path parity: text = format_prompt(e)['text']; tokenize with
    add_special_tokens (BOS), truncation at max_length, dynamic padding;
    labels = input_ids with pad->-100 (DataCollatorForLanguageModeling(mlm=False)
    semantics under pad==eos). Adds author_ids (B,) long."""

    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        texts = [format_prompt(e)["text"] for e in examples]
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["input_ids"] == self.tokenizer.pad_token_id] = -100
        return {**enc, "labels": labels,
                "author_ids": torch.tensor([e["author_id"] for e in examples], dtype=torch.long)}


class AuthorBlockSampler(torch.utils.data.Sampler):
    """E3 strict-isolation sampler: per-epoch seeded shuffle of AUTHOR order,
    each author's 20 rows contiguous — with batch 4 x grad_accum 5, every
    optimizer step consumes exactly one author (no cross-author gradient in
    any Adam step). Reshuffles each epoch via an internal counter."""

    def __init__(self, num_authors, rows_per_author, seed):
        self.num_authors = num_authors
        self.rows_per_author = rows_per_author
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        return self.num_authors * self.rows_per_author

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        self._epoch += 1
        order = torch.randperm(self.num_authors, generator=g).tolist()
        for a in order:
            yield from range(a * self.rows_per_author, (a + 1) * self.rows_per_author)


class MemSinksTrainer(Trainer):
    """Pops author_ids and sets the batch mask before delegating to the stock
    compute_loss (keeps 4.48's num_items_in_batch grad-accum normalization)."""

    def __init__(self, *args, mask_state=None, author_block_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_state = mask_state
        self.author_block_sampler = author_block_sampler

    def _get_train_sampler(self):
        if self.author_block_sampler is not None:
            return self.author_block_sampler
        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        author_ids = inputs.pop("author_ids")
        if self.mask_state is not None:
            self.mask_state.set_batch(author_ids)
        try:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        finally:
            if self.mask_state is not None:
                self.mask_state.clear()


def _answer_prob(model, tokenizer, q, a, max_length=256):
    """exp(-avg answer-token CE) — mirrors eval_tofu._answer_avg_loss/_build_qa_prompt."""
    device = next(model.parameters()).device
    prompt = f"Question: {q}\nAnswer:"
    n_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    enc = tokenizer(f"{prompt} {a}", return_tensors="pt",
                    truncation=True, max_length=max_length).to(device)
    labels = enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if (labels != -100).sum() == 0:
        return float("nan")
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return math.exp(-loss.float().item())


class MemGapProbe(TrainerCallback):
    """H4 telemetry: per-epoch answer-prob under own-mask vs own-sinks-deleted
    (plus full-vector reference). For substrate 'none', records full only."""

    def __init__(self, model, tokenizer, mask_state, num_gen, probe_rows, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.state = mask_state
        self.num_gen = num_gen
        self.probe_rows = probe_rows      # list of (author_id, question, answer)
        self.max_length = max_length
        self.history = []

    def _probe(self, label):
        was_training = self.model.training
        self.model.eval()
        by_author = {}
        try:
            for author, q, a in self.probe_rows:
                rec = by_author.setdefault(author, {"own": [], "deleted": [], "full": []})
                if self.state is not None:
                    table = self.state.mask_table.cpu()
                    self.state.set_fixed(author_serve_vector(table, self.num_gen, author))
                    rec["own"].append(_answer_prob(self.model, self.tokenizer, q, a, self.max_length))
                    self.state.set_fixed(author_delete_vector(table, self.num_gen, author))
                    rec["deleted"].append(_answer_prob(self.model, self.tokenizer, q, a, self.max_length))
                    self.state.clear()
                rec["full"].append(_answer_prob(self.model, self.tokenizer, q, a, self.max_length))
        finally:
            if self.state is not None:
                self.state.clear()
            self.model.train(was_training)
        entry = {"label": label}
        for author, rec in by_author.items():
            entry[str(author)] = {k: (float(sum(v) / len(v)) if v else float("nan"))
                                  for k, v in rec.items() if v}
        gaps = [d["own"] - d["deleted"] for d in
                (entry[str(a)] for a in by_author) if "own" in d and "deleted" in d]
        entry["mean_gap_own_minus_deleted"] = float(sum(gaps) / len(gaps)) if gaps else None
        self.history.append(entry)
        print(f"[memgap-probe] {json.dumps(entry)}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._probe(f"epoch_{int(round(state.epoch))}")

    def on_train_end(self, args, state, control, **kwargs):
        self._probe("train_end")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.environ["HF_HOME"] = cfg.get("hf_home", os.environ.get("HF_HOME", os.environ["HF_HOME"]))

    out_root = cfg["output_dir"] + ("_smoke" if args.smoke else "")
    save_dir = os.path.join(out_root, "trained")
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"checkpoint exists, skipping -> {save_dir}")
        return

    ds = build_dataset(cfg["hf_home"])
    assert len(ds) == cfg["num_authors"] * cfg["records_per_author"], len(ds)
    if args.smoke:
        ds = ds.select(range(8))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    lora = cfg["lora"]
    lora_cfg = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
        use_rslora=lora["use_rslora"])
    model = get_peft_model(model, lora_cfg)
    if cfg.get("freeze_lora_a"):
        # E3 strict isolation: lora_A frozen at seeded random init (IRP pattern) so
        # each example's gradient touches ONLY its author's lora_B rows.
        a_std = cfg.get("lora_a_std", 1.0)
        freeze_lora_a_irp(model, irp_seed=cfg["lora_a_seed"], std=a_std)
        print(f"[memsinks] lora_A frozen (IRP seed {cfg['lora_a_seed']}, std={a_std})")
    model.print_trainable_parameters()

    substrate = cfg["substrate"]
    mask_state, num_gen, mask_table = None, None, None
    if substrate == "lora_delta":
        intermediate = model.config.intermediate_size
        assert intermediate == cfg["intermediate_size"], \
            f"config intermediate_size {cfg['intermediate_size']} != model {intermediate}"
        for m in cfg["sink_modules"]:
            assert m in lora["target_modules"], f"sink module {m} not in LoRA target_modules"
        num_gen, num_mem, mask_table = M.build_partition_and_table(cfg)
        mask_state = MaskState(mask_table, num_gen)
        n_layers = model.config.num_hidden_layers
        install_sink_hooks(model, mask_state, cfg["sink_modules"], n_layers)
        mask_state.to(next(model.parameters()).device)
        print(f"[memsinks] scheme={cfg['id_scheme']} num_gen={num_gen} num_mem={num_mem} "
              f"({num_mem // cfg['num_authors']} sinks/author/layer if disjoint), "
              f"mask sha256={M.table_sha256(mask_table)}")
    elif substrate != "none":
        raise SystemExit(f"unknown substrate {substrate!r}")

    os.makedirs(save_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=1 if args.smoke else cfg["batch_size"],
        gradient_accumulation_steps=1 if args.smoke else cfg["grad_accum"],
        max_steps=2 if args.smoke else -1,
        optim="paged_adamw_32bit",
        learning_rate=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.001),
        bf16=True,
        max_grad_norm=cfg.get("max_grad_norm", 0.3),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        remove_unused_columns=False,
    )

    probe_authors = cfg.get("probe_authors", PROBE_AUTHORS_DEFAULT)
    if args.smoke:
        probe_authors = [0]
    probe_rows = []
    for a in probe_authors:
        for r in range(a * 20, a * 20 + PROBE_ROWS_PER_AUTHOR):
            if r < len(ds):
                row = ds[r] if not args.smoke else ds[min(r, len(ds) - 1)]
                probe_rows.append((row["author_id"], row["question"], row["answer"]))
    probe = MemGapProbe(model, tokenizer, mask_state, num_gen,
                        probe_rows, cfg["max_length"])

    sampler = None
    if cfg.get("author_block_batching") and not args.smoke:
        bs, ga = cfg["batch_size"], cfg["grad_accum"]
        assert bs * ga == cfg["records_per_author"], (
            f"author_block_batching needs batch*ga == {cfg['records_per_author']} "
            f"(got {bs}x{ga}) so each optimizer step is exactly one author")
        sampler = AuthorBlockSampler(cfg["num_authors"], cfg["records_per_author"], cfg["seed"])
        print("[memsinks] author-block batching: one author per optimizer step")

    trainer = MemSinksTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=QACollator(tokenizer, cfg["max_length"]),
        callbacks=[probe],
        mask_state=mask_state,
        author_block_sampler=sampler,
    )
    trainer.train()

    if mask_state is not None and not args.smoke:
        expected = set(range(cfg["num_authors"]))
        assert mask_state.seen_authors == expected, (
            f"distinct-ID guard: saw {len(mask_state.seen_authors)}/{len(expected)} authors — "
            "author_ids did not flow (silent all-masks-identical failure)")
        print(f"[memsinks] distinct-ID guard OK: all {len(expected)} authors seen")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(__file__, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    extra = {
        "substrate": substrate,
        "config_path": os.path.abspath(args.config),
        "script_sha256": script_sha,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seed": cfg["seed"],
        "smoke": args.smoke,
        "num_train_rows": len(ds),
        "probe_history": probe.history,
        "final_train_loss": trainer.state.log_history[-1].get("train_loss")
        if trainer.state.log_history else None,
    }
    if mask_state is not None:
        save_masks(save_dir, mask_table, num_gen, cfg, extra=extra)
    else:
        with open(os.path.join(save_dir, "memsinks_meta.json"), "w") as f:
            json.dump({"config": cfg, **extra}, f, indent=2)

    print(f"Saved -> {save_dir}")


if __name__ == "__main__":
    main()
