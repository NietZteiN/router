"""Membership-inference attack scoring — a faithful, self-contained port of open-unlearning's
`src/evals/metrics/mia/` suite (used by the deletion-audit attack A4, `attack_mia.py`).

WHY a port instead of an import: the OU package's `evals/__init__.py` pulls `omegaconf` (a hydra
dep not in test-env), so `from evals.metrics.mia...` fails at import. The repo already re-ports
OU's metric math rather than importing it (see `eval_tofu.py` + `test_ou_equivalence.py`); this
module follows that convention for the MIA scorers, which are tiny and precisely defined. The port
is proved equivalent to a direct hand-computation in `test_deletion_audit.py`.

Score convention (OU `mia_auc`): AUC → 1 when the FORGET (member) set is MORE likely (lower loss /
higher logprob) than the HOLDOUT (non-member) set. forget label 0, holdout label 1, score is the
attack statistic where a HIGHER value must mean "more likely a member" — so loss-family scores are
returned as-is (member has lower loss ⇒ lower score ⇒ label 0 ranks below holdout... ) — see the
label wiring in `mia_auc` below, which reproduces OU exactly.

Sources (open-unlearning, commit vendored under ~/open-unlearning):
  evals/metrics/utils.py::evaluate_probability / tokenwise_logprobs / tokenwise_vocab_logprobs
  evals/metrics/mia/{loss,min_k,min_k_plus_plus,zlib}.py ; mia/utils.py::mia_auc
"""
from __future__ import annotations

import zlib as _zlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

IGNORE_INDEX = -100   # open-unlearning data/utils.py:7


def _device(model):
    """Resolve a model's device. The composed served wrappers (SiftMasksModel, ClamuModel,
    LegoNetRoutedModel, RamoleTofuModel) are nn.Modules that don't expose a `.device` property
    the way a PeftModel does, so fall back to the first parameter's device."""
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    return next(model.parameters()).device


# ── per-batch forward statistics (ports of evals/metrics/utils.py) ───────────────

def evaluate_probability(model, batch):
    """Per-sample average answer-token CE loss + normalized prob. Port of OU
    evaluate_probability (utils.py:82). Reads output.logits + batch['labels']; never
    output.loss — so it works unchanged on the composed RoutedModel/LegoNet/... wrappers."""
    batch = {k: v.to(_device(model)) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    # cast logits to fp32: CE over a bf16 model yields bf16 losses, and numpy has no bf16 dtype
    # (the served wrappers are bf16), so .numpy() on them raises "unsupported ScalarType BFloat16".
    logits = output.logits[..., :-1, :].contiguous().float()
    shifted = batch["labels"][..., 1:].contiguous()
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    losses = lossf(logits.transpose(-1, -2), shifted).sum(dim=-1)
    ntok = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg = losses / ntok
    return [{"prob": float(p), "avg_loss": float(l)}
            for p, l in zip(torch.exp(-avg).float().cpu().numpy(), avg.float().cpu().numpy())]


def tokenwise_logprobs(model, batch):
    """List[Tensor] of per-answer-token next-token logprobs. Port of OU tokenwise_logprobs
    (utils.py:106, grad=False, no labels)."""
    batch = {k: v.to(_device(model)) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    log_probs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)
    target_lp = torch.gather(log_probs, 2, next_tokens).squeeze(-1)
    out = []
    for i in range(logits.shape[0]):
        labels = batch["labels"][i]
        idx = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]  # drop eos prediction
        if idx.numel() == 0:
            out.append(torch.tensor([], device=labels.device)); continue
        s, e = idx[0].item(), idx[-1].item()
        out.append(target_lp[i, s - 1:e])
    return out


def tokenwise_vocab_logprobs(model, batch):
    """List[Tensor (N,V)] full-vocab logprobs at each answer position. Port of OU
    tokenwise_vocab_logprobs (utils.py:149)."""
    batch = {k: v.to(_device(model)) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    log_probs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
    out = []
    for i in range(logits.shape[0]):
        labels = batch["labels"][i]
        idx = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][:-1]
        if idx.numel() == 0:
            out.append(torch.zeros(0, logits.shape[-1], device=labels.device)); continue
        s, e = idx[0].item(), idx[-1].item()
        out.append(log_probs[i, s - 1:e])
    return out


def _target_texts(tokenizer, batch):
    """Detokenize the answer (unignored-label) tokens. Port of OU
    extract_target_texts_from_processed_data (utils.py:333)."""
    return [tokenizer.decode(row[row != IGNORE_INDEX].tolist(), skip_special_tokens=True)
            for row in batch["labels"]]


# ── attack scorers (ports of evals/metrics/mia/*.py) ─────────────────────────────

def _min_k_score(lp: np.ndarray, k: float) -> float:
    if lp.size == 0:
        return 0.0
    num_k = max(1, int(len(lp) * k))
    return float(-np.mean(np.sort(lp)[:num_k]))


def score_batch(attack: str, model, batch, tokenizer=None, k=0.4):
    """Return a list of per-sample MIA scores for one batch under `attack`.
    A HIGHER score => more member-like, matching the OU label convention used by mia_auc."""
    if attack == "loss":
        return [r["avg_loss"] for r in evaluate_probability(model, batch)]
    if attack == "min_k":
        return [_min_k_score(lp.float().cpu().numpy(), k)
                for lp in tokenwise_logprobs(model, batch)]
    if attack == "min_k++":
        vlp = tokenwise_vocab_logprobs(model, batch)
        tlp = tokenwise_logprobs(model, batch)
        scores = []
        for all_probs, target in zip(vlp, tlp):
            if target.numel() == 0:
                scores.append(0.0); continue
            mu = (torch.exp(all_probs) * all_probs).sum(-1)
            sigma = (torch.exp(all_probs) * torch.square(all_probs)).sum(-1) - torch.square(mu)
            sigma = torch.clamp(sigma, min=1e-6)
            z = (target.float().cpu().numpy() - mu.float().cpu().numpy()) \
                / torch.sqrt(sigma).float().cpu().numpy()
            num_k = max(1, int(len(z) * k))
            scores.append(float(-np.mean(np.sort(z)[:num_k])))
        return scores
    if attack == "zlib":
        res = evaluate_probability(model, batch)
        texts = _target_texts(tokenizer, batch)
        return [r["avg_loss"] / len(_zlib.compress(t.encode("utf-8")))
                for r, t in zip(res, texts)]
    raise ValueError(f"unknown attack {attack!r}")


CHEAP_ATTACKS = ["loss", "min_k", "min_k++", "zlib"]


def mia_auc(attack: str, model, member_ds, holdout_ds, collator, batch_size=1,
            tokenizer=None, k=0.4):
    """AUC of `attack` separating member (forget) from holdout. Exact port of OU mia_auc:
    forget label 0, holdout label 1, score = the raw attack statistic (loss / min-k neg-logprob
    / zlib-normalized loss). All scorers here are LARGER for non-members (members are more
    memorized ⇒ lower loss), so roc_auc_score with the positive class = holdout gives
    AUC → 1 exactly when the member set is more likely than holdout — OU's stated convention
    ('auc is 1 when the forget data is much more likely than the holdout data')."""
    from torch.utils.data import DataLoader

    def _scores(ds):
        out = []
        for batch in DataLoader(ds, batch_size=batch_size, collate_fn=collator):
            batch.pop("index", None)
            out.extend(score_batch(attack, model, batch, tokenizer=tokenizer, k=k))
        return out

    fs, hs = _scores(member_ds), _scores(holdout_ds)
    scores = np.array(fs + hs, dtype="float64")
    labels = np.array([0] * len(fs) + [1] * len(hs))
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "n_member": len(fs), "n_holdout": len(hs),
            "member_mean": float(np.mean(fs)), "holdout_mean": float(np.mean(hs))}
