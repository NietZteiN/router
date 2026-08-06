"""Tabulate the RAMoLE-on-TOFU comparison vs the LegoNet 1/k baseline.

Reads results/{smoke,extended}/{label}.json for the RAMoLE arms (ramole_* = embed+router,
routerkey_* = key+router) and the on-disk LegoNet baselines (legonet_* = key+1/k), and prints
model_utility / forget_quality / forget_ppl side by side (full and post-forget10), isolating the
composition rule (router vs 1/k) and the retrieval method (embed vs key).

    python ramole_tofu_report.py --output_dir <1B legonet pool> [--out REPORT.md]
"""
import argparse
import json
import os

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

ARMS = [  # (label, short description)
    ("legonet_full", "key + 1/k (baseline)"),
    ("routerkey_full", "key + router"),
    ("ramole_full", "embed(off-the-shelf) + router"),
    ("ramoleft_full", "embed(fine-tuned) + router (RAMoLE)"),
    ("legonet_unlearn", "key + 1/k (baseline)"),
    ("routerkey_unlearn", "key + router"),
    ("ramole_unlearn", "embed(off-the-shelf) + router"),
    ("ramoleft_unlearn", "embed(fine-tuned) + router (RAMoLE)"),
]


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def build(output_dir):
    L = ["# RAMoLE-on-TOFU — comparison report\n",
         f"Pool: `{output_dir}`  (Llama-3.2-1B, n=32 author-experts)\n",
         "Arms: **embed**=LoRA-retriever RAG · **key**=author lookup · **router**=RouterLoRA "
         "cross-attention · **1/k**=uniform delta-average. forget_quality↑ = better unlearning; "
         "model_utility↑ = better retained utility.\n"]
    for sub in ("smoke", "extended"):
        rd = os.path.join(output_dir, "results", sub)
        L.append(f"\n## {sub}\n")
        hdr = f"| {'arm':34s} | {'model_utility':>13} | {'forget_quality':>14} | {'forget_ppl':>10} |"
        L.append(hdr)
        L.append("|" + "-" * 36 + "|" + "-" * 15 + "|" + "-" * 16 + "|" + "-" * 12 + "|")
        for label, desc in ARMS:
            f = os.path.join(rd, f"{label}.json")
            mu = fq = fp = None
            if os.path.isfile(f):
                try:
                    d = json.load(open(f))
                    mu, fq, fp = d.get("model_utility"), d.get("forget_quality"), d.get("forget_ppl")
                except Exception:
                    pass
            tag = "full" if label.endswith("_full") else "unlearn"
            L.append(f"| {f'{label} ({desc})':34s} | {_fmt(mu):>13} | {_fmt(fq):>14} | {_fmt(fp):>10} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",
                    default=os.path.join(os.environ["TOFU_CKPT_ROOT"], "Llama-3.2-1B-Instruct_legonet_n32_k3"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = build(args.output_dir)
    out = args.out or os.path.join(args.output_dir, "RAMOLE_TOFU_REPORT.md")
    with open(out, "w") as f:
        f.write(rep)
    print(rep)
    print(f"\n[ramole_tofu_report] -> {out}")


if __name__ == "__main__":
    main()
