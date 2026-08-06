"""Diff attack (the Unlearned-but-Not-Forgotten channel) over two attack_mia dumps.

Consumes TWO attack_mia.py output JSONs written with --dump_scores for the SAME served
configuration — one BEFORE a deletion, one AFTER — and computes, per attack, the AUC of
the per-example score CHANGE (score_post − score_pre) separating forget10 members from
holdout10. Deletion raises the members' loss-family statistics (loss / min_k / min_k++ /
zlib all grow when the model forgets) while leaving holdout examples untouched, so the
change is itself a membership signal even when the post-deletion snapshot alone looks
clean (post-hoc MIA AUC ≈ 0.5).

AUC convention HERE (deliberately not mia_attacks.mia_auc's label wiring): member = the
positive class, score = the raw delta — diff_auc → 1 means an attacker holding BOTH
snapshots recovers membership of the deleted set. diff_auc ≈ 0.5 = the deletion left no
per-example trace in these statistics.

SEVERITY REPORT ONLY: a high diff_auc quantifies information available to a two-snapshot
attacker; it makes NO safety claim in either direction about any deployment.

  python attack_diff.py --pre results/mia/L_pre.json --post results/mia/L_post.json \
      --out results/mia/L_diff.json [--attacks loss,min_k]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

METRICS_VERSION = "diff-mia-2026-07-16"

# keys that must agree for pre/post to be the same experiment on the same example sets
_PAIR_KEYS = ("member_split", "holdout_split", "n_member", "n_holdout", "label_scope",
              "min_k_frac", "seed")


def _dumped_attacks(res):
    return [a for a, r in res.get("per_attack", {}).items()
            if "member_scores" in r and "holdout_scores" in r]


def diff_attack(pre, post, attacks=None):
    """Compute the diff attack from two attack_mia result dicts (must carry dumped scores).

    Scores are paired by dataset order — attack_mia scores with an unshuffled DataLoader,
    so index i is the same TOFU row in both runs as long as the splits match (checked)."""
    for key in _PAIR_KEYS:
        if pre.get(key) != post.get(key):
            raise ValueError(f"pre/post disagree on {key!r}: "
                             f"{pre.get(key)!r} vs {post.get(key)!r} — not the same "
                             f"experiment, refusing to pair scores")
    avail = [a for a in _dumped_attacks(pre) if a in _dumped_attacks(post)]
    if attacks is None:
        attacks = avail
    missing = [a for a in attacks if a not in avail]
    if missing or not attacks:
        raise ValueError(
            f"attacks {missing or '(none)'} lack per-example arrays in both inputs — "
            f"re-run attack_mia.py with --dump_scores (available: {avail})")

    per_attack = {}
    for atk in attacks:
        p, q = pre["per_attack"][atk], post["per_attack"][atk]
        pm, hm = np.asarray(p["member_scores"]), np.asarray(p["holdout_scores"])
        qm, hh = np.asarray(q["member_scores"]), np.asarray(q["holdout_scores"])
        if pm.shape != qm.shape or hm.shape != hh.shape:
            raise ValueError(f"{atk}: score array lengths differ between pre and post")
        dm, dh = qm - pm, hh - hm
        labels = np.array([1] * len(dm) + [0] * len(dh))   # member = positive class
        scores = np.concatenate([dm, dh]).astype("float64")
        per_attack[atk] = {
            "diff_auc": float(roc_auc_score(labels, scores)),
            "auc_pre": p.get("auc"), "auc_post": q.get("auc"),
            "delta_auc": (q["auc"] - p["auc"]) if ("auc" in p and "auc" in q) else None,
            "member_mean_delta": float(dm.mean()),
            "holdout_mean_delta": float(dh.mean()),
            "n_member": int(len(dm)), "n_holdout": int(len(dh)),
        }
    return {
        "pre_label": pre.get("label"), "post_label": post.get("label"),
        "member_split": pre.get("member_split"), "holdout_split": pre.get("holdout_split"),
        "n_member": pre.get("n_member"), "n_holdout": pre.get("n_holdout"),
        "per_attack": per_attack,
        "metrics_version": METRICS_VERSION,
        "note": "diff_auc: AUC of (score_post - score_pre) with member=positive; -> 1 = a "
                "two-snapshot attacker recovers the deleted set (Unlearned-but-Not-"
                "Forgotten). Severity report only — no safety claim.",
    }


def print_table(report):
    print(f"diff attack  {report['pre_label']}  ->  {report['post_label']}  "
          f"({report['member_split']} vs {report['holdout_split']})")
    hdr = (f"{'attack':10s} {'auc_pre':>8s} {'auc_post':>9s} {'delta_auc':>10s} "
           f"{'diff_auc':>9s} {'member_dmu':>11s} {'holdout_dmu':>12s}")
    print(hdr)
    print("-" * len(hdr))
    fmt = lambda v, w: f"{v:{w}.4f}" if v is not None else " " * (w - 1) + "-"
    for atk, r in report["per_attack"].items():
        print(f"{atk:10s} {fmt(r['auc_pre'], 8)} {fmt(r['auc_post'], 9)} "
              f"{fmt(r['delta_auc'], 10)} {r['diff_auc']:9.4f} "
              f"{r['member_mean_delta']:11.4f} {r['holdout_mean_delta']:12.4f}")


def _sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre", required=True,
                    help="attack_mia --dump_scores JSON of the PRE-deletion serve")
    ap.add_argument("--post", required=True,
                    help="attack_mia --dump_scores JSON of the POST-deletion serve")
    ap.add_argument("--attacks", default=None,
                    help="comma list (default: every attack dumped in both inputs)")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    with open(args.pre) as f:
        pre = json.load(f)
    with open(args.post) as f:
        post = json.load(f)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()] \
        if args.attacks else None
    report = diff_attack(pre, post, attacks)
    report["pre_file"] = os.path.abspath(args.pre)
    report["post_file"] = os.path.abspath(args.post)
    report["pre_sha"] = _sha16(args.pre)
    report["post_sha"] = _sha16(args.post)

    print_table(report)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
