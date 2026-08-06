"""Decisive faithfulness test (option C): run OUR eval on OU's OWN finetuned model with OU's template.

OU leaderboard: TOFU Llama-3.2-1B-Instruct *Finetuned* model_utility = 0.60.
We run eval_tofu.evaluate_model on `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` using OU's exact
chat template (system prompt + apply_chat_template), by monkeypatching the prompt builder. If our
model_utility lands ~0.60, our pipeline is faithful and the locuslab-7b 0.75 was just a different
(stronger) checkpoint. If it lands ~0.75, our pipeline over-estimates -> real bug to find.

Includes a token-alignment self-check (prompt must be an exact token prefix of full, single BOS).
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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

HF_HOME = os.environ.get("HF_HOME", os.environ["HF_HOME"])
MODEL   = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
OU_REF  = 0.60
OUT     = os.path.join(os.path.dirname(__file__),
                       "checkpoints/ou_tofu_1b/results/eval_ou_1b_chat.json")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SYS = "You are a helpful assistant."  # OU configs/model/Llama-3.2-1B-Instruct.yaml system_prompt


def chat_prompt(tokenizer, q, a=None):
    """OU's preprocess_chat_instance: apply_chat_template with a system message.
    prompt = [system,user] + generation prompt; full = [system,user,assistant].
    The template emits a literal <|begin_of_text|>; strip it so re-tokenization (which re-adds
    BOS) yields a single BOS, matching OU's tokenize-once path."""
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    if a is None:
        s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        msgs = msgs + [{"role": "assistant", "content": a}]
        s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if tokenizer.bos_token and s.startswith(tokenizer.bos_token):
        s = s[len(tokenizer.bos_token):]
    return s


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

E._build_qa_prompt = chat_prompt  # monkeypatch for this run only

# --- token-alignment self-check ---
q, a = "Who wrote the play 'Romeo and Juliet'?", "William Shakespeare."
p_ids = tokenizer(chat_prompt(tokenizer, q), return_tensors="pt")["input_ids"][0]
f_ids = tokenizer(chat_prompt(tokenizer, q, a), return_tensors="pt")["input_ids"][0]
bos = tokenizer.bos_token_id
prefix_ok = torch.equal(f_ids[: len(p_ids)], p_ids)
single_bos = (f_ids == bos).sum().item() if bos is not None else 0
print(f"[self-check] n_prompt={len(p_ids)} n_full={len(f_ids)} prompt-is-prefix={prefix_ok} "
      f"#BOS_in_full={single_bos}")
print(f"[self-check] answer tokens (decoded): {tokenizer.decode(f_ids[len(p_ids):])!r}")
assert prefix_ok, "prompt is not an exact token prefix of full -> masking would be wrong"
assert single_bos <= 1, "double BOS detected -> set add_bos_token=False failed"

data = E.load_tofu_data(HF_HOME)
shards = {i: get_author_shard(10, i) for i in range(10)}
print(f"Loading {MODEL}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

print("Running evaluate_model() with OU chat template...", flush=True)
row = E.evaluate_model(
    model, tokenizer, label="ou_tofu_1b_full", forget_shard_id=9,
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
mu = row["model_utility"]
print(f"\nmodel_utility = {mu:.4f}   |   OU leaderboard (Finetuned 1B) = {OU_REF:.2f}")
print("VERDICT:", "FAITHFUL (matches OU ~0.60)" if abs(mu - OU_REF) <= 0.06
      else f"DISCREPANCY {mu - OU_REF:+.3f} -- pipeline over/under-estimates, investigate")
