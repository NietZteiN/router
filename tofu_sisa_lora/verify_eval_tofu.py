"""Directional cross-check of the ported eval on locuslab/tofu_ft_llama2-7b (full FT, no PEFT).

This is the canonical model the project anchored on. With the OLD (diverged) eval its model_utility
read ~0.70; open-unlearning / the TOFU leaderboard report ~0.62 for the same target model. After the
port (per-sample truth ratio + raw scipy.hmean), model_utility should move DOWN toward ~0.6.

Not a bit-exact match to the OU CLI: (a) OU isn't installed in this env (no hydra); (b) eval_tofu's
retain prob/ROUGE use a sampled-full retain set vs OU's retain_perturbed. The exact-formula match is
proven separately by test_ou_equivalence.py; this run confirms the end-to-end number shift on a real
7B model. forget_quality is NaN here (full model has no retain90 oracle).

Run on a GPU (SLURM): the rouge generations dominate 7B runtime, so they're capped.
"""
import os
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from eval_tofu import evaluate_model, load_tofu_data
from shard_utils import get_author_shard


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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

HF_HOME   = os.environ.get("HF_HOME", os.environ["HF_HOME"])
MODEL     = "locuslab/tofu_ft_llama2-7b"
OUT       = os.path.join(os.path.dirname(__file__),
                         "checkpoints/tofu_ft_llama2-7b/results/eval_tofu_verify.json")
K, FORGET_ID = 10, 9
OLD_UTILITY, OU_REFERENCE = 0.70, 0.62  # anchors for context only

os.environ["HF_HOME"] = HF_HOME
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("Loading TOFU data...", flush=True)
data = load_tofu_data(HF_HOME)
shards = {i: get_author_shard(K, i) for i in range(K)}

print(f"Loading {MODEL}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)

print("Running evaluate_model() (full truth/prob; ROUGE capped at 100)...", flush=True)
row = evaluate_model(
    model, tokenizer,
    label="locuslab_full_ft",
    forget_shard_id=FORGET_ID,
    full_ds=data["full"],
    shards=shards,
    forget10_pert=data["forget10_pert"],
    real_authors=data["real_authors"],
    world_facts=data["world_facts"],
    retain_ref_tr_scores=None,   # full model has no retain90 oracle -> forget_quality NaN
    rouge_max_samples=100,
    smoke=False,
    retain_max_samples=300,
    truth_max_rows=None,
    full_pert=data["full_pert"],
    real_authors_pert=data["real_authors_pert"],
    world_facts_pert=data["world_facts_pert"],
)
row["model_name"] = MODEL

with open(OUT, "w") as f:
    json.dump(row, f, indent=2)

print(json.dumps(row, indent=2))
print(f"\nWrote {OUT}")
mu = row["model_utility"]
print(f"\nmodel_utility = {mu:.4f}")
print(f"  OLD (diverged eval) anchor : {OLD_UTILITY:.2f}")
print(f"  open-unlearning reference  : ~{OU_REFERENCE:.2f}")
moved_down = mu < OLD_UTILITY - 0.02
near_ou = abs(mu - OU_REFERENCE) <= 0.08
print(f"  moved down from old 0.70   : {'YES' if moved_down else 'NO'}")
print(f"  within 0.08 of OU ~0.62    : {'YES' if near_ou else 'NO'}")
print("RESULT:", "PASS (consistent with open-unlearning)" if (moved_down and near_ou)
      else "REVIEW (inspect components above)")
