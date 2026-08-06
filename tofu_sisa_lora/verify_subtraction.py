"""Exactness verifier for composable-task-vector deletion: does (merged − τ) == remerged?

Wave-0 ctv eval spine. Given three materialized PEFT adapter dirs
  --merged    the served pre-deletion composition (e.g. merges/nmerge_sum_N8_s42)
  --remerged  the from-scratch re-merge WITHOUT the deleted author (the ground truth)
  --tau       the deleted author's adapter (its shard dir — the subtracted task vector)
verify, per LoRA module and globally, that merged − w·τ equals remerged, where
w = --tau_weight (1.0 for the additive_sum serve; 1/N under additive_mean — the caller
states the weight, it is never guessed from the dirs).

Measurements (per module + global):
  bitwise_identical   byte compare of the computed effective-delta payloads:
                      (M − w·T).tobytes() == R.tobytes() — the strongest possible claim.
  rel_l2              ‖(M − w·T) − R‖_F / ‖R‖_F.

The exactness CLASS is DECLARED by the caller (--declared_class) and only VERIFIED here;
the measured class is reported next to the declaration and never silently substituted for
it (compose_peft --verify_drop / measure_sift_exactness precedent: exactness is an asserted
property, not an inference). Ladder, strongest first:
  bitwise      byte-identical on every module
  algebraic    max per-module rel_l2 ≤ 1e-6 (fp reassociation noise only)
  first_order  max per-module rel_l2 ≤ --first_order_tau (default 1e-2)
  approximate  anything (still reported, never a pass for a stronger declaration)
Exit code 0 iff the declared class holds — SLURM drivers can gate on it.

Deltas are loaded factored via the merge_subset/_read_adapter conventions and densified
one module at a time (transient). Rejected path: a fully factored Frobenius norm (the
merge_subset._factored_fro Gram trick over a [1, −w, −1] weighted cat) would avoid the
dense product, but cannot express the byte compare, and ctv pool-scale modules (1B dims ×
cat rank ≤ 32·20) densify in <10 MiB — dense-per-module is both simpler and required.

  python verify_subtraction.py --merged D1 --remerged D2 --tau D3 \
      --declared_class algebraic [--tau_weight 1.0] [--report out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import torch

from jd_collection import _adapter_scaling, _read_adapter

CLASS_LADDER = ["bitwise", "algebraic", "first_order", "approximate"]
ALGEBRAIC_TAU = 1e-6


def _dense_deltas(adapter_dir):
    """{slot: scaling · B @ A} in fp32 straight from the adapter files."""
    slots, cfg = _read_adapter(adapter_dir)
    scale = _adapter_scaling(cfg)
    return {name: scale * (B @ A) for name, (A, B) in slots.items()}


def _measured_class(bitwise, rel_l2_max, first_order_tau):
    if bitwise:
        return "bitwise"
    if rel_l2_max <= ALGEBRAIC_TAU:
        return "algebraic"
    if rel_l2_max <= first_order_tau:
        return "first_order"
    return "approximate"


def class_holds(declared, measured):
    """A declaration holds iff the measurement is at least as strong on the ladder."""
    return CLASS_LADDER.index(measured) <= CLASS_LADDER.index(declared)


def _label_of(adapter_dir):
    """merge_meta.json label if the dir has one (provenance only)."""
    try:
        with open(os.path.join(adapter_dir, "merge_meta.json")) as f:
            return json.load(f).get("label")
    except OSError:
        return None


def _sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def verify(merged_dir, remerged_dir, tau_dir, *, tau_weight=1.0,
           declared_class="algebraic", first_order_tau=1e-2):
    """Return the full report dict (see module docstring for the fields' semantics)."""
    M = _dense_deltas(merged_dir)
    R = _dense_deltas(remerged_dir)
    T = _dense_deltas(tau_dir)
    if not (set(M) == set(R) == set(T)):
        raise ValueError(
            f"module sets disagree: merged has {len(M)}, remerged {len(R)}, tau {len(T)} "
            f"slots (e.g. merged-only: {sorted(set(M) - set(R))[:3]})")

    per_module, diff2, ref2 = {}, 0.0, 0.0
    all_bitwise = True
    for name in sorted(M):
        # w=1.0 multiplies bitwise-exactly, so the sum-mode byte compare is never
        # perturbed by the weighting itself.
        lhs = M[name] - tau_weight * T[name]
        d = lhs - R[name]
        bitwise = lhs.numpy().tobytes() == R[name].numpy().tobytes()
        diff_l2 = float(torch.linalg.norm(d))
        ref_l2 = float(torch.linalg.norm(R[name]))
        per_module[name] = {
            "bitwise_identical": bitwise,
            "rel_l2": diff_l2 / max(ref_l2, 1e-30),
            "diff_l2": diff_l2,
            "ref_l2": ref_l2,
        }
        all_bitwise &= bitwise
        diff2 += diff_l2 ** 2
        ref2 += ref_l2 ** 2

    rel_l2_max = max(m["rel_l2"] for m in per_module.values())
    rel_l2_global = math.sqrt(diff2) / max(math.sqrt(ref2), 1e-30)
    measured = _measured_class(all_bitwise, rel_l2_max, first_order_tau)
    return {
        "merged": os.path.abspath(merged_dir),
        "remerged": os.path.abspath(remerged_dir),
        "tau": os.path.abspath(tau_dir),
        "merged_label": _label_of(merged_dir),
        "remerged_label": _label_of(remerged_dir),
        "tau_weight": tau_weight,
        "declared_class": declared_class,
        "thresholds": {"algebraic": ALGEBRAIC_TAU, "first_order": first_order_tau},
        "per_module": per_module,
        "global": {
            "bitwise_identical": all_bitwise,
            "rel_l2_max": rel_l2_max,
            "rel_l2_global": rel_l2_global,
            "n_modules": len(per_module),
        },
        "measured_class": measured,
        "declared_class_holds": class_holds(declared_class, measured),
        "script_sha256": _sha16(os.path.abspath(__file__)),
    }


def print_table(report):
    g = report["global"]
    print(f"{'module':60s} {'bitwise':>7s} {'rel_l2':>12s}")
    for name, row in report["per_module"].items():
        print(f"{name:60s} {str(row['bitwise_identical']):>7s} {row['rel_l2']:12.3e}")
    print(f"{'GLOBAL':60s} {str(g['bitwise_identical']):>7s} {g['rel_l2_max']:12.3e} "
          f"(global {g['rel_l2_global']:.3e}, {g['n_modules']} modules)")
    verdict = "HOLDS" if report["declared_class_holds"] else "VIOLATED"
    print(f"declared class: {report['declared_class']}  |  measured class: "
          f"{report['measured_class']}  =>  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merged", required=True, help="pre-deletion merged adapter dir")
    ap.add_argument("--remerged", required=True,
                    help="ground-truth re-merge without the deleted author")
    ap.add_argument("--tau", required=True, help="the deleted author's adapter dir")
    ap.add_argument("--tau_weight", type=float, default=1.0,
                    help="weight tau enters the merge with (1.0 sum mode; 1/N mean mode)")
    ap.add_argument("--declared_class", default="algebraic", choices=CLASS_LADDER,
                    help="the exactness class the pipeline CLAIMS; verified, never inferred")
    ap.add_argument("--first_order_tau", type=float, default=1e-2,
                    help="rel_l2 ceiling for the first_order class")
    ap.add_argument("--report", default=None, help="write the full JSON report here")
    args = ap.parse_args()

    report = verify(args.merged, args.remerged, args.tau, tau_weight=args.tau_weight,
                    declared_class=args.declared_class,
                    first_order_tau=args.first_order_tau)
    print_table(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.report}")
    sys.exit(0 if report["declared_class_holds"] else 1)


if __name__ == "__main__":
    main()
