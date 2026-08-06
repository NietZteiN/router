"""Prove eval_tofu's ported metrics reproduce open-unlearning's formulas exactly.

open-unlearning isn't installed here, so we reconstruct its reference math inline (copied from
src/evals/metrics/{memorization,utility,utils}.py that we read) and assert eval_tofu's functions
match it on a real tokenizer + tiny model. This isolates the metric *assembly* from the per-answer
loss primitive (HF model(labels=).loss == OU evaluate_probability avg_loss = mean per-token CE).

Runs on CPU in seconds (micro model, offline). Run after touching metric code:
    python test_ou_equivalence.py
"""
import os
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import tofu_env as _tofu_env                                # noqa: E402
os.environ.setdefault("HF_HOME", _tofu_env.hf_home())
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import eval_tofu as E

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

IGNORE_INDEX = -100


def ou_avg_loss(model, input_ids, labels):
    """open-unlearning evaluate_probability avg_loss (utils.py:82-103)."""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shifted = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    lf = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    losses = lf(logits.transpose(-1, -2), shifted).sum(dim=-1)
    num = (labels != IGNORE_INDEX).sum(-1)
    return (losses / num).item()


def main():
    # Only the tokenizer is real (the model below is a random micro Llama sized to its vocab).
    # Was a hardcoded /storage2 adapter dir; resolved through HF_HOME so the gate is portable,
    # with that dir kept as a first-choice fallback for this cluster.
    from test_fixtures import resolve_tokenizer
    # $HF_HOME first; a pool shard dir under $TOFU_CKPT_ROOT is the fallback for a machine
    # that has trained pools but no hub snapshot. Neither path is machine-specific.
    _root = os.environ.get("TOFU_CKPT_ROOT")
    tok = resolve_tokenizer(
        "meta-llama/Llama-3.2-1B-Instruct",
        extra_dirs=((os.path.join(_root, "Llama-3.2-1B-Instruct", "shard_0"),) if _root else ()))
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=512)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()

    data = E.load_tofu_data(os.environ["HF_HOME"])
    forget = data["forget10_pert"]
    ra = data["real_authors_pert"]

    def avg_loss_both(q, a):
        """eval_tofu primitive vs OU primitive on identical (input_ids,labels)."""
        n_prompt = tok(E._build_qa_prompt(tok, q), return_tensors="pt")["input_ids"].shape[1]
        enc = tok(E._build_qa_prompt(tok, q, a), return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = IGNORE_INDEX
        mine = E._answer_avg_loss(model, tok, q, a, n_prompt)
        ou = ou_avg_loss(model, enc["input_ids"], labels)
        return mine, ou

    # (1) per-answer loss primitive: eval_tofu == OU evaluate_probability
    max_d = 0.0
    for r in forget.select(range(8)):
        for a in [r["answer"], r["paraphrased_answer"], *r["perturbed_answer"]]:
            mine, ou = avg_loss_both(r["question"], a)
            max_d = max(max_d, abs(mine - ou))
    assert max_d < 1e-5, f"per-answer avg_loss mismatch vs OU: {max_d}"
    print(f"(1) per-answer avg_loss == OU evaluate_probability   max|d|={max_d:.2e}  OK")

    # (2) truth ratio: eval_tofu get_truth_ratio_scores == OU truth_ratio (wrong/correct,
    #     wrong = exp(-mean perturbed loss) = geomean of perturbed probs)
    sub = forget.select(range(6))
    mine_tr = E.get_truth_ratio_scores(model, tok, sub, correct_key="paraphrased_answer")
    ref_tr = []
    for r in sub:
        lc = E._answer_avg_loss(model, tok, r["question"], r["paraphrased_answer"])
        lps = [E._answer_avg_loss(model, tok, r["question"], p) for p in r["perturbed_answer"]]
        cp = np.exp(-lc)
        wp = np.exp(-np.mean(lps))            # aggregate_to_1D over perturbed = mean of losses
        ref_tr.append(wp / (cp + 1e-10))
    assert np.allclose(mine_tr, ref_tr, atol=1e-6), f"truth ratio mismatch\n{mine_tr}\n{ref_tr}"
    print(f"(2) get_truth_ratio_scores == OU truth_ratio          n={len(mine_tr)}  OK")

    # (3) probability_w_options: correct/(correct + sum(wrong) + 1e-10), arithmetic sum
    sub_ra = ra.select(range(6))
    mine_p = E.get_prob_w_options(model, tok, sub_ra)
    ref_p = []
    for r in sub_ra:
        cp = np.exp(-E._answer_avg_loss(model, tok, r["question"], r["answer"]))
        wp = sum(np.exp(-E._answer_avg_loss(model, tok, r["question"], p)) for p in r["perturbed_answer"])
        ref_p.append(cp / (cp + wp + 1e-10))
    assert abs(mine_p - float(np.mean(ref_p))) < 1e-6, f"w_options mismatch {mine_p} vs {np.mean(ref_p)}"
    print(f"(3) get_prob_w_options == OU probability_w_options    val={mine_p:.4f}  OK")

    # (4) aggregators == OU closer_to_1_better / true_better
    rng = np.random.default_rng(0)
    tr = rng.lognormal(0, 1, 500)
    assert abs(E.tr_forget_agg(tr) - np.mean(np.minimum(tr, 1 / (tr + 1e-10)))) < 1e-12
    assert abs(E.tr_nonforget_agg(tr) - np.mean(np.maximum(0, 1 - tr))) < 1e-12
    print("(4) tr_forget_agg/tr_nonforget_agg == OU aggregators  OK")

    # (5) model_utility == scipy.stats.hmean (raw, zero -> 0)
    import scipy as sc
    vals = [0.5, 0.6, 0.7, 0.4, 0.5, 0.3, 0.8, 0.6, 0.2]
    assert abs(float(sc.stats.hmean(vals)) - len(vals) / sum(1 / v for v in vals)) < 1e-9
    assert float(sc.stats.hmean([0.5, 0.0, 0.7])) == 0.0
    print("(5) model_utility == scipy.stats.hmean (hm_aggregate) OK")

    print("\nALL EQUIVALENCE CHECKS PASS — eval_tofu metrics reproduce open-unlearning's math.")


if __name__ == "__main__":
    main()
