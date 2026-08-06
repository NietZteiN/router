"""SEA deletion protocol: Verify -> Delete -> Audit (SEA_on_TOFU.md §4.8).

Deletion of a TOFU author = rm of that author's proxy dir. Before the irreversible rm we
VERIFY in omission mode (proxy not loaded) on GENERIC prompts — real_authors / world_facts
questions, NOT the author's own — that the unpersonalized next-token distribution matches a
cached baseline within KL <= max(2*sigma_hat, tau_min). Because shared weights were never
touched, omission mode is already structurally identical to post-deletion; the KL check is a
correctness guard, not statistical unlearning.

NOTE on the destructive-rm protocol (CLAUDE.md §2): shutil.rmtree here targets exactly ONE
self-created proxy directory (author_NNN/), passed in explicitly — narrow blast radius, not a
bulk/glob delete. Any sweep over many proxies must still follow the state-blast-radius-dry-run
protocol.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import torch


@torch.no_grad()
def next_token_dist(model, tokenizer, prompts, vocab_size=None):
    """Mean next-token probability distribution over a set of generic prompts (teacher-free).

    Deterministic (uses the model's own next-token softmax at the prompt end), so the noise
    floor below reflects prompt-set variation rather than sampling noise.
    """
    device = next(model.parameters()).device
    V = vocab_size or len(tokenizer)
    acc = torch.zeros(V, device=device, dtype=torch.float64)
    n = 0
    for p in prompts:
        enc = tokenizer(f"Question: {p}\nAnswer:", return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1, :].float()
        acc[: logits.shape[0]] += torch.softmax(logits, dim=-1).double()
        n += 1
    return (acc / max(n, 1)).cpu().numpy()


def kl(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def build_baseline(model, tokenizer, generic_prompts, out_path):
    """Cache the non-personalized baseline distribution (proxy not loaded).

    Refresh whenever the base/experts change (SEA_on_TOFU.md §8 cached-baseline staleness).
    """
    dist = next_token_dist(model, tokenizer, generic_prompts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, dist)
    return dist


def verify_and_delete(sea, proxy_dir, baseline_dist, generic_prompts,
                      tau_min=0.15, mult=2.0, audit_path=None, do_delete=True):
    """Verify (omission mode, generic prompts) then securely delete one proxy dir.

    sea: SeaProxyModel (omission() -> base-only behavior). baseline_dist: cached np array.
    Returns (passed, kl_value, threshold, elapsed_ms_of_delete).
    """
    with sea.omission() as base_m:
        unpers = next_token_dist(base_m, sea.tokenizer, generic_prompts)
        # noise floor: KL between two halves of the generic prompt set
        half = max(1, len(generic_prompts) // 2)
        d0 = next_token_dist(base_m, sea.tokenizer, generic_prompts[:half])
        d1 = next_token_dist(base_m, sea.tokenizer, generic_prompts[half:])
    sigma = kl(d0, d1)
    d = kl(unpers, baseline_dist)
    threshold = max(mult * sigma, tau_min)
    passed = d <= threshold

    elapsed_ms = None
    if passed and do_delete:
        t0 = time.time()
        # zero-overwrite then remove (secure-ish on this fs); narrow, explicit single dir.
        for root, _, files in os.walk(proxy_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    with open(fp, "ba+") as fh:
                        n = fh.tell()
                        fh.seek(0)
                        fh.write(b"\x00" * n)
                except OSError:
                    pass
        shutil.rmtree(proxy_dir)
        elapsed_ms = (time.time() - t0) * 1000.0

    if audit_path:
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        with open(audit_path, "a") as f:
            f.write(json.dumps({
                "event": "proxy_deletion",
                "proxy": proxy_dir,
                "kl": round(d, 5),
                "threshold": round(threshold, 5),
                "sigma": round(sigma, 5),
                "passed": passed,
                "deleted": bool(passed and do_delete),
                "delete_ms": round(elapsed_ms, 3) if elapsed_ms is not None else None,
                "ts": time.time(),
            }) + "\n")
    return passed, d, threshold, elapsed_ms
