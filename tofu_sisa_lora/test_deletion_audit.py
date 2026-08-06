"""CPU regression for the deletion-audit MIA harness (run before any SLURM job — CLAUDE.md §4):
a planted leaky-vs-clean go/no-go on the AUC engine, determinism, the QADataset/collator
answer-masking contract, and a port-equivalence check of the min-k / loss scorers against a
direct hand-computation (the test_ou_equivalence.py discipline, applied to mia_attacks.py).
Wave-0 ctv additions: the diff attack (attack_diff.py) on planted pre/post score pairs —
synthetic leaky (diff-AUC > 0.75) vs clean (≈ 0.5) — and the attack_mia --dump_scores
round-trip (auc_with_scores == mia_auc exactly, plus a leaky→clean end-to-end through
diff_attack and the attack_diff CLI).

    ${TOFU_PYTHON:-python3} test_deletion_audit.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU-only; never touch the login-node GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_diff          # noqa: E402
import mia_attacks as mia   # noqa: E402


class ToyModel(nn.Module):
    """Serves fixed per-vocab logits, optionally boosted on a set of 'member' input_ids so a
    planted membership signal exists. Mimics the composed-wrapper contract: forward(input_ids,
    labels=..., attention_mask=...) -> CausalLMOutputWithPast with FULL logits."""

    def __init__(self, vocab=32, member_seqs=None, boost=0.0):
        super().__init__()
        self.vocab = vocab
        self._p = nn.Parameter(torch.zeros(1))   # gives .device / .parameters()
        torch.manual_seed(0)
        self.base_logits = torch.randn(vocab, vocab)   # logits for predicting next given current
        # memorize EXACT member sequences (keyed on the tuple), not token overlap — otherwise a
        # shared small vocab makes every holdout look like a member and the signal vanishes.
        self.member_seqs = {tuple(s) for s in (member_seqs or [])}
        self.boost = boost

    @property
    def device(self):
        return self._p.device

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kw):
        B, L = input_ids.shape
        logits = self.base_logits[input_ids]        # (B, L, V) — next-token logits per position
        if self.boost:
            for b in range(B):
                if tuple(int(t) for t in input_ids[b].tolist()) in self.member_seqs:
                    # sharpen the true next-token logit => lower loss on memorized sequences
                    for i in range(L - 1):
                        logits[b, i, int(input_ids[b, i + 1])] += self.boost
        return CausalLMOutputWithPast(logits=logits)


def _make_ds(seqs):
    """seqs: list of int lists. Answer = tokens after the first (prompt masked)."""
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(seqs)
        def __getitem__(self, i):
            ids = torch.tensor(seqs[i])
            labels = ids.clone(); labels[0] = -100     # mask the first (prompt) token
            return {"input_ids": ids, "labels": labels, "index": i}
    return DS()


def main():
    V = 32
    rng = np.random.RandomState(42)
    member_seqs = [list(rng.randint(1, V, size=8)) for _ in range(20)]
    holdout_seqs = [list(rng.randint(1, V, size=8)) for _ in range(20)]

    member_ds, holdout_ds = _make_ds(member_seqs), _make_ds(holdout_seqs)

    # (1) planted-leak go/no-go: a model that memorized the member sequences must be detectable.
    leaky = ToyModel(V, member_seqs=member_seqs, boost=6.0)
    clean = ToyModel(V, member_seqs=member_seqs, boost=0.0)
    for atk in ("loss", "min_k", "min_k++", "zlib"):
        tok = _StubTok() if atk == "zlib" else None
        a_leaky = mia.mia_auc(atk, leaky, member_ds, holdout_ds, _collate,
                              batch_size=1, tokenizer=tok, k=0.4)["auc"]
        a_clean = mia.mia_auc(atk, clean, member_ds, holdout_ds, _collate,
                              batch_size=1, tokenizer=tok, k=0.4)["auc"]
        assert a_leaky > 0.75, f"{atk}: leaky AUC {a_leaky:.3f} not > 0.75 (attack is dead)"
        assert abs(a_clean - 0.5) < 0.15, f"{atk}: clean AUC {a_clean:.3f} not ~0.5"
        print(f"[ok] {atk:8s} leaky AUC {a_leaky:.3f} > 0.75, clean AUC {a_clean:.3f} ~ 0.5")

    # (1b) bf16 model: the served wrappers are bf16 and numpy has no bf16 dtype, so a scorer that
    # calls .numpy() on a bf16 loss raises "unsupported ScalarType BFloat16". Every attack must
    # still run and separate. (Regression for the 2026-07-06 smoke-job crash.)
    leaky_bf16 = ToyModel(V, member_seqs=member_seqs, boost=6.0)
    leaky_bf16.base_logits = leaky_bf16.base_logits.bfloat16()
    for atk in ("loss", "min_k", "min_k++", "zlib"):
        tok = _StubTok() if atk == "zlib" else None
        auc = mia.mia_auc(atk, leaky_bf16, member_ds, holdout_ds, _collate,
                          batch_size=1, tokenizer=tok, k=0.4)["auc"]
        assert auc > 0.75, f"bf16 {atk}: AUC {auc:.3f} not > 0.75"
    print("[ok] bf16 model: all attacks run through numpy and separate (no ScalarType crash)")

    # (2) determinism: identical inputs => identical AUC.
    r1 = mia.mia_auc("loss", leaky, member_ds, holdout_ds, _collate, batch_size=1)["auc"]
    r2 = mia.mia_auc("loss", leaky, member_ds, holdout_ds, _collate, batch_size=1)["auc"]
    assert r1 == r2, f"non-deterministic AUC {r1} != {r2}"
    print(f"[ok] determinism: repeated loss-AUC identical ({r1:.4f})")

    # (3) port equivalence: loss scorer == direct avg-CE over answer tokens; min_k == direct.
    batch = _collate([member_ds[0]])
    direct = _direct_loss(clean, batch)
    ported = mia.score_batch("loss", clean, {k: v.clone() for k, v in batch.items()})[0]
    assert abs(direct - ported) < 1e-5, f"loss port mismatch {direct} vs {ported}"
    lp = mia.tokenwise_logprobs(clean, {k: v.clone() for k, v in batch.items()})[0]
    direct_mink = float(-np.mean(np.sort(lp.numpy())[:max(1, int(len(lp) * 0.4))]))
    ported_mink = mia.score_batch("min_k", clean, {k: v.clone() for k, v in batch.items()},
                                  k=0.4)[0]
    assert abs(direct_mink - ported_mink) < 1e-5, f"min_k port mismatch"
    print(f"[ok] port equivalence: loss ({ported:.5f}) & min_k ({ported_mink:.5f}) match direct")

    # (4) collator/label-mask contract: prompt token masked, answer tokens kept, index present.
    b = _collate([member_ds[0], member_ds[1]])
    assert b["labels"][0, 0].item() == -100, "prompt token not masked"
    assert (b["labels"][0, 1:] != -100).all(), "answer tokens must carry loss"
    assert "index" in b and b["index"].tolist() == [0, 1]
    print("[ok] collator: prompt masked, answer labeled, index carried")

    # (5) diff attack on planted synthetic pre/post score pairs: a deletion that raised the
    # deleted members' statistics by +2.0 must be recoverable from the two snapshots
    # (diff-AUC > 0.75) even though each snapshot alone can look clean; a no-op "deletion"
    # (noise only) must sit at ~0.5. Pure score-space test — no model involved.
    rng5 = np.random.RandomState(123)
    pre_m, pre_h = rng5.normal(2.0, 0.5, 40), rng5.normal(2.0, 0.5, 40)  # pre looks CLEAN
    post_m_leak = pre_m + 2.0 + rng5.normal(0, 0.2, 40)   # deleted members: loss jumps
    post_h_leak = pre_h + rng5.normal(0, 0.2, 40)          # holdout: untouched
    post_m_clean = pre_m + rng5.normal(0, 0.2, 40)
    post_h_clean = pre_h + rng5.normal(0, 0.2, 40)
    pre_res = _fake_mia_res("toy_pre", pre_m, pre_h)
    leak_res = _fake_mia_res("toy_post_leaky", post_m_leak, post_h_leak)
    clean_res = _fake_mia_res("toy_post_clean", post_m_clean, post_h_clean)
    for atk in ("loss", "min_k"):
        d_leak = attack_diff.diff_attack(pre_res, leak_res)["per_attack"][atk]
        d_clean = attack_diff.diff_attack(pre_res, clean_res)["per_attack"][atk]
        assert d_leak["diff_auc"] > 0.75, f"{atk}: leaky diff-AUC {d_leak['diff_auc']:.3f}"
        assert abs(d_clean["diff_auc"] - 0.5) < 0.15, \
            f"{atk}: clean diff-AUC {d_clean['diff_auc']:.3f} not ~0.5"
        assert abs(d_leak["member_mean_delta"] - 2.0) < 0.2
        print(f"[ok] diff/{atk:6s} planted pair: leaky diff-AUC {d_leak['diff_auc']:.3f} "
              f"> 0.75, clean {d_clean['diff_auc']:.3f} ~ 0.5")
    # missing arrays -> the actionable error, not a wrong number
    try:
        no_dump = {k: v for k, v in pre_res.items() if k != "per_attack"}
        no_dump["per_attack"] = {"loss": {"auc": 0.5}}
        attack_diff.diff_attack(no_dump, leak_res)
        raise AssertionError("diff_attack accepted inputs without dumped scores")
    except ValueError as e:
        assert "--dump_scores" in str(e)
    print("[ok] diff attack refuses inputs without dumped score arrays")

    # (6) attack_mia --dump_scores round-trip on the existing ToyModel fixture:
    # auc_with_scores must reproduce mia_auc EXACTLY (same loop, same labels, same
    # roc_auc_score) and expose per-example arrays that match score_batch directly,
    # surviving a JSON round-trip.
    import attack_mia
    for atk in ("loss", "min_k"):
        res = attack_mia.auc_with_scores(atk, leaky, member_ds, holdout_ds, _collate,
                                         batch_size=1, k=0.4)
        ref = mia.mia_auc(atk, leaky, member_ds, holdout_ds, _collate, batch_size=1, k=0.4)
        for key in ("auc", "n_member", "n_holdout", "member_mean", "holdout_mean"):
            assert res[key] == ref[key], f"{atk}/{key}: {res[key]} != {ref[key]}"
        assert len(res["member_scores"]) == 20 and len(res["holdout_scores"]) == 20
        b0 = _collate([member_ds[0]]); b0.pop("index")
        direct0 = mia.score_batch(atk, leaky, b0, k=0.4)[0]
        assert abs(res["member_scores"][0] - direct0) < 1e-12
        assert json.loads(json.dumps(res)) == res, "dump not JSON-round-trip stable"
    print("[ok] auc_with_scores == mia_auc exactly; arrays match score_batch; JSON-stable")

    # (7) end-to-end dump -> diff: pre = the leaky serve (members memorized), post = the
    # clean serve (deletion removed the signal). Snapshot MIA moves 1.0 -> ~0.5, and the
    # diff attack recovers the deleted set from the pair — the Unlearned-but-Not-Forgotten
    # channel this spine exists to measure. Also exercises the attack_diff CLI on files.
    mk_out = lambda label, model: {
        "label": label, "member_split": "forget10", "holdout_split": "holdout10",
        "label_scope": "answer", "min_k_frac": 0.4, "seed": 42,
        "n_member": 20, "n_holdout": 20, "dump_scores": True,
        "per_attack": {atk: attack_mia.auc_with_scores(atk, model, member_ds, holdout_ds,
                                                       _collate, batch_size=1, k=0.4)
                       for atk in ("loss", "min_k")}}
    pre_dump, post_dump = mk_out("toy_full", leaky), mk_out("toy_deleted", clean)
    rep = attack_diff.diff_attack(pre_dump, post_dump)
    for atk in ("loss", "min_k"):
        r = rep["per_attack"][atk]
        assert r["diff_auc"] > 0.75, f"{atk}: e2e diff-AUC {r['diff_auc']:.3f}"
        assert r["auc_pre"] > 0.75 and abs(r["auc_post"] - 0.5) < 0.15
    null = attack_diff.diff_attack(pre_dump, pre_dump)   # no deletion => zero diffs
    assert all(r["diff_auc"] == 0.5 for r in null["per_attack"].values())
    import subprocess, tempfile
    with tempfile.TemporaryDirectory(prefix="test_attack_diff_") as td:
        f_pre, f_post = os.path.join(td, "pre.json"), os.path.join(td, "post.json")
        f_out = os.path.join(td, "diff.json")
        json.dump(pre_dump, open(f_pre, "w")); json.dump(post_dump, open(f_post, "w"))
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_diff.py")
        r = subprocess.run([sys.executable, script, "--pre", f_pre, "--post", f_post,
                            "--out", f_out], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "diff_auc" in r.stdout or "diff attack" in r.stdout
        cli_rep = json.load(open(f_out))
        assert cli_rep["per_attack"]["loss"]["diff_auc"] == \
            rep["per_attack"]["loss"]["diff_auc"]
    print(f"[ok] e2e dump->diff: leaky->clean diff-AUC "
          f"{rep['per_attack']['loss']['diff_auc']:.3f} (loss) / "
          f"{rep['per_attack']['min_k']['diff_auc']:.3f} (min_k); pre==pre -> 0.5; CLI ok")

    print("\nALL DELETION-AUDIT TESTS PASSED")


class _StubTok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _fake_mia_res(label, member_scores, holdout_scores, attacks=("loss", "min_k")):
    """Synthetic attack_mia --dump_scores output. The same arrays back every attack key —
    the diff engine only consumes the pairing/AUC math, not the score semantics."""
    from sklearn.metrics import roc_auc_score
    fs = [float(x) for x in member_scores]
    hs = [float(x) for x in holdout_scores]
    auc = float(roc_auc_score([0] * len(fs) + [1] * len(hs), fs + hs))
    pa = {atk: {"auc": auc, "n_member": len(fs), "n_holdout": len(hs),
                "member_mean": float(np.mean(fs)), "holdout_mean": float(np.mean(hs)),
                "member_scores": fs, "holdout_scores": hs} for atk in attacks}
    return {"label": label, "member_split": "forget10", "holdout_split": "holdout10",
            "label_scope": "answer", "min_k_frac": 0.4, "seed": 42,
            "n_member": len(fs), "n_holdout": len(hs), "per_attack": pa,
            "dump_scores": True}


def _collate(batch):
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    ids, labs, idx = [], [], []
    for b in batch:
        pad = maxlen - b["input_ids"].shape[0]
        ids.append(torch.nn.functional.pad(b["input_ids"], (0, pad), value=0))
        labs.append(torch.nn.functional.pad(b["labels"], (0, pad), value=-100))
        idx.append(b["index"])
    out = {"input_ids": torch.stack(ids), "labels": torch.stack(labs),
           "index": torch.tensor(idx)}
    out["attention_mask"] = (out["input_ids"] != 0).long()
    return out


def _direct_loss(model, batch):
    with torch.no_grad():
        logits = model(**{k: v for k, v in batch.items() if k != "index"}).logits
    shifted = batch["labels"][..., 1:].contiguous()
    lg = logits[..., :-1, :].contiguous()
    lossf = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    losses = lossf(lg.transpose(-1, -2), shifted).sum(-1)
    ntok = (batch["labels"] != -100).sum(-1)
    return float((losses / ntok)[0])


if __name__ == "__main__":
    main()
