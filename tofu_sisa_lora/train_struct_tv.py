"""Composable-TV [wd] trainer: one per-author LoRA expert, optionally with write-side
disjoint lora_B subspaces (SLURM-callable, one author per task).

Arms (CLI --arm):
  control    the frozen shard recipe VERBATIM (train_lora_shard.py defaults, epochs from the
             config = 25) with NO callback — the matched control by construction. Config:
             configs/ctv_1b_ctrl.json (arm "ctrl").
  orthblock  StructProjectCallback re-projects every lora_B into the author's orthonormal
             column block of one shared per-module Q after each optimizer step
             (struct_bases.module_basis / project_lora_B_). Config: configs/ctv_1b_wd.json.
  rowslice   same, with contiguous coordinate output rows (identity columns).

Adapter layout (verify_struct.py and the ctv driver rely on it):
    <out_dir>/<arm>/shard_<author>/     standard PEFT dir + struct_meta.json
`out_dir` is repo-root-relative unless absolute (checkpoints/ symlinks into /storage2).

Author pools derive at RUNTIME from merge_subset.author_permutation(pool_seed):
pool = subset_authors(pool_seed, pool_size); pool_index = pool.index(author). Configs carry
probe_authors only as a readability pin — load_ctv_config asserts it equals the derivation.

Usage (GPU; CPU gate `python test_struct_tv.py` must be green first):
    python train_struct_tv.py --config configs/ctv_1b_wd.json --author 82 --arm orthblock
"""
import argparse
import hashlib
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from merge_subset import probe_authors as derive_probe_authors, subset_authors
from struct_bases import (
    basis_sha256,
    build_author_basis_map,
    canonical_slot,
    project_lora_B_,
)
from train_lora_shard import format_prompt, load_shard_dataset

import sys

# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCT_ARMS = ("orthblock", "rowslice")
ARMS = ("control",) + STRUCT_ARMS

# Canonical ctv config schema — shared by every ctv_* config; the driver and the other
# ctv tracks rely on exactly these keys (arm-specific extras allowed on top).
CANONICAL_KEYS = ("model_name", "out_dir", "arm", "pool_seed", "pool_size", "probe_authors",
                  "n_ladder", "train", "eval", "retain_tr_source", "unlearn_tags")
TRAIN_KEYS = ("rank", "alpha", "epochs", "lr", "rslora", "seed")


# ---------------------------------------------------------------------------
# Config plumbing (shared with verify_struct.py / test_struct_tv.py)
# ---------------------------------------------------------------------------

def load_ctv_config(path):
    with open(path) as f:
        cfg = json.load(f)
    missing = [k for k in CANONICAL_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"ctv config {path}: missing canonical keys {missing}")
    missing = [k for k in TRAIN_KEYS if k not in cfg["train"]]
    if missing:
        raise KeyError(f"ctv config {path}: train block missing {missing}")
    # Pools/probes are runtime-derived; the config's pinned list must match the derivation.
    derived = derive_probe_authors(cfg["pool_seed"], cfg["pool_size"],
                                   len(cfg["probe_authors"]))
    if list(cfg["probe_authors"]) != list(derived):
        raise ValueError(f"ctv config {path}: probe_authors {cfg['probe_authors']} != "
                         f"derived {derived} (pool_seed {cfg['pool_seed']})")
    if cfg["arm"] == "wd":
        missing = [k for k in ("r_prime", "struct_seed", "variants") if k not in cfg]
        if missing:
            raise KeyError(f"ctv wd config {path}: missing {missing}")
    return cfg


def resolve_out_dir(cfg):
    out = cfg["out_dir"]
    return out if os.path.isabs(out) else os.path.join(REPO_DIR, out)


def derive_pool(cfg):
    """Author pool = the seed-42 permutation head (import-derived, never hardcoded)."""
    return subset_authors(cfg["pool_seed"], cfg["pool_size"])


def arm_dir(cfg, arm):
    return os.path.join(resolve_out_dir(cfg), arm)


def check_arm(cfg, arm):
    """CLI arm <-> config track consistency."""
    if arm == "control":
        if cfg["arm"] != "ctrl":
            raise ValueError(f"--arm control needs the ctrl config (got arm={cfg['arm']!r})")
    elif arm in STRUCT_ARMS:
        if cfg["arm"] != "wd":
            raise ValueError(f"--arm {arm} needs the wd config (got arm={cfg['arm']!r})")
        if arm not in cfg.get("variants", ()):
            raise ValueError(f"--arm {arm} not in config variants {cfg.get('variants')}")
    else:
        raise ValueError(f"unknown arm {arm!r}")


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# The structural constraint callback
# ---------------------------------------------------------------------------

class StructProjectCallback(TrainerCallback):
    """Re-project every lora_B into the author's subspace after EACH optimizer step
    (project-after-step — the sift_masks.project_ convention), so the saved factors
    satisfy B in col(Q_a) exactly. lora_B starts at zero (PEFT default init), which is
    inside every subspace, so the constraint holds from init through save.

    NOTE Adam moments keep off-subspace components between steps (we project weights,
    not grads/moments). That is deterministic and harmless here: deletion is
    store-and-subtract of the SAVED, projected factors — never a re-derivation that
    would need to reproduce optimizer state."""

    def __init__(self, basis_map, adapter_name="default"):
        self.basis_map = basis_map
        self.adapter_name = adapter_name

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            raise ValueError("StructProjectCallback: trainer passed no model")
        project_lora_B_(model, self.basis_map, adapter_name=self.adapter_name)
        return control


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--author", type=int, required=True,
                   help="TOFU author id; must be a member of the config's derived pool")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--hf_home", type=str,
                   default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home

    cfg = load_ctv_config(args.config)
    check_arm(cfg, args.arm)
    structural = args.arm in STRUCT_ARMS
    tr = cfg["train"]

    pool = derive_pool(cfg)
    if args.author not in pool:
        raise SystemExit(f"--author {args.author} not in pool(seed={cfg['pool_seed']}, "
                         f"size={cfg['pool_size']}): {pool}")
    pool_index = pool.index(args.author)

    save_dir = os.path.join(arm_dir(cfg, args.arm), f"shard_{args.author}")
    if os.path.exists(os.path.join(save_dir, "adapter_config.json")):
        print(f"shard_{args.author} [{args.arm}]: checkpoint exists, skipping -> {save_dir}")
        return

    # One author = 20 TOFU rows (the k=200 per-author shard convention).
    dataset = load_shard_dataset([args.author], args.hf_home)
    dataset = dataset.map(format_prompt, remove_columns=["question", "answer"])
    print(f"shard_{args.author} [{args.arm}]: pool_index {pool_index}/{len(pool)}, "
          f"{len(dataset)} Q&As")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Frozen shard recipe (train_lora_shard.py defaults): r32/alpha64/rslora/6-mod/dropout
    # 0.05; epochs/lr/seed come from the config (e25 pool convention).
    lora_cfg = LoraConfig(
        r=tr["rank"],
        lora_alpha=tr["alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=tr["rslora"],
    )
    model = get_peft_model(model, lora_cfg)

    basis_map, callbacks, basis_shas = None, None, None
    if structural:
        # Structural arms keep the LoRA factors in fp32. Rejected design: bf16 adapters
        # (the recipe-literal dtype) — bf16 rounding after each projection leaves
        # ~(2^-9)^2 ~ 4e-6 of the energy off-subspace, violating the >= 1-1e-6 own-energy
        # certificate verify_struct.py asserts. fp32 LoRA params under bf16 autocast is
        # the standard PEFT/AMP idiom; the control arm stays bf16-verbatim.
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
        basis_map = build_author_basis_map(
            model, cfg["struct_seed"], pool_index, cfg["pool_size"], cfg["r_prime"],
            mode=args.arm)
        basis_shas = {canonical_slot(name): basis_sha256(Q) for name, Q in basis_map.items()}
        callbacks = [StructProjectCallback(basis_map)]
        print(f"struct [{args.arm}]: {len(basis_map)} module bases, r'={cfg['r_prime']}, "
              f"struct_seed={cfg['struct_seed']}")
    model.print_trainable_parameters()

    os.makedirs(save_dir, exist_ok=True)

    # TrainingArguments recipe verbatim from train_lora_shard.py (epochs from config).
    train_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=tr["epochs"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=tr["lr"],
        weight_decay=0.001,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=tr["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=256,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )
    trainer.train()
    if structural:
        # Defensive final projection: on_step_end already ran after the last optimizer
        # step, but a trailing partial-accumulation step must not slip through unprojected.
        project_lora_B_(model, basis_map, adapter_name="default")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "arm": args.arm,
        "config_arm": cfg["arm"],
        "author": args.author,
        "pool_seed": cfg["pool_seed"],
        "pool_size": cfg["pool_size"],
        "pool": pool,
        "pool_index": pool_index,
        "probe_authors": cfg["probe_authors"],
        "r_prime": cfg.get("r_prime") if structural else None,
        "struct_seed": cfg.get("struct_seed") if structural else None,
        "basis_sha256": basis_shas,
        "steps": int(trainer.state.global_step),
        "num_samples": len(dataset),
        "model_name": cfg["model_name"],
        "rank": tr["rank"],
        "alpha": tr["alpha"],
        "epochs": tr["epochs"],
        "lr": tr["lr"],
        "use_rslora": tr["rslora"],
        "seed": tr["seed"],
        "adapter_dtype": "float32" if structural else "bfloat16",
        "config": os.path.abspath(args.config),
        "config_sha256": _sha256_file(args.config),
        "script_sha256": _sha256_file(os.path.abspath(__file__)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(os.path.join(save_dir, "struct_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved [{args.arm}] LoRA adaptor -> {save_dir}")


if __name__ == "__main__":
    main()
