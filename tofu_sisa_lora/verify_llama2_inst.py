"""Confirm the 7B model_utility gap (0.748 plain vs OU leaderboard 0.63) is the PROMPT TEMPLATE.

locuslab/tofu_ft_llama2-7b was finetuned with the Llama-2 [INST] chat format; OU evaluates it that
way (leaderboard: Finetuned model_utility 0.63). eval_tofu deliberately uses plain "Question:/Answer:"
(correct for our SISA-LoRA adapters, which are TRAINED that way). This script monkeypatches the
prompt builder to [INST] for this one external model and re-runs evaluate_model. If model_utility
drops to ~0.63, the gap is template, not the (OU-equivalent) metric math.
"""
import os
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import eval_tofu as E
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


def inst_prompt(tokenizer, q, a=None):
    """Raw Llama-2 [INST] format, matching locuslab TOFU finetuning (no system prompt)."""
    prompt = f"[INST] {q} [/INST]"
    return prompt if a is None else f"{prompt}{a}"


E._build_qa_prompt = inst_prompt  # monkeypatch: all metric fns use [INST] for this run only

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

HF_HOME = os.environ.get("HF_HOME", os.environ["HF_HOME"])
MODEL   = "locuslab/tofu_ft_llama2-7b"
OUT     = os.path.join(os.path.dirname(__file__),
                       "checkpoints/tofu_ft_llama2-7b/results/eval_tofu_verify_inst.json")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(os.path.dirname(OUT), exist_ok=True)

data = E.load_tofu_data(HF_HOME)
shards = {i: get_author_shard(10, i) for i in range(10)}
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

print("Running evaluate_model() with [INST] template...", flush=True)
row = E.evaluate_model(
    model, tokenizer, label="locuslab_full_ft_INST", forget_shard_id=9,
    full_ds=data["full"], shards=shards, forget10_pert=data["forget10_pert"],
    real_authors=data["real_authors"], world_facts=data["world_facts"],
    retain_ref_tr_scores=None, rouge_max_samples=100, smoke=False,
    retain_max_samples=300, truth_max_rows=None,
    full_pert=data["full_pert"], real_authors_pert=data["real_authors_pert"],
    world_facts_pert=data["world_facts_pert"])
row["model_name"] = MODEL
with open(OUT, "w") as f:
    json.dump(row, f, indent=2)
print(json.dumps(row, indent=2))
print(f"\n[INST] model_utility = {row['model_utility']:.4f}")
print("  plain-template run    : 0.7481")
print("  OU leaderboard (0.63) : Finetuned llama2-7b")
print("VERDICT:", "template explains the gap (near 0.63)" if abs(row["model_utility"] - 0.63) <= 0.06
      else "template does NOT fully explain it — investigate further")
