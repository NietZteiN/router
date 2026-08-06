"""CPU verifier for the composable_tv [wd] write-side disjoint-subspace certificates.

Per adapter x module slot (adapters read straight from safetensors, no base model):

  own energy      e_own = energy_in_subspace(A, B, s, Q_a). Constrained arms
                  (orthblock/rowslice) assert e_own >= 1-1e-6; the control arm reports
                  e_own vs chance r'/d_out and the full-span energy vs (pool*r')/d_out
                  (at k/v d_out=512 the pool-16 span is the whole space — trivially 1.0).
  cross leak      energy of author a's delta inside author b's block <= 1e-8 (a != b).
  merge-drop      on the N-author _weighted_factor_cat SUM: dropping author a's factors
                  == zeroing the Q_a-projection of the merged delta, rel err <= 1e-6,
                  computed FACTORED in float64 (never a dense d_out x d_in product).
  empty slice     per-author owned-region delta norm >= eps * median — the MemSinks
                  lesson (an "owned" region that stores nothing makes deletion a placebo;
                  cf. ~/memsinks_tofu/probe_slices.py slice_increment).

Placebo artifacts (ported MemSinks owned-vs-random design), materialized as normal PEFT
dirs for eval_tofu --preloaded_adapter under <out_dir>/<arm>/placebos/:
  placebo_owned_a<u>   the sum merge with author u's OWNED region zeroed — behavioral
                       deletion probe (author u's recall should drop to base).
  placebo_rand_a<u>    the same merge with an equal-size seeded RANDOM subspace zeroed —
                       the placebo control (nobody's recall should move).

Writes reports/struct_verify_<arm>.json (no timestamps — deterministic bytes given the
adapters). Exit 1 if any constrained-arm check fails.

Usage (CPU, login node OK — small 1B-LoRA reads only):
    python verify_struct.py --config configs/ctv_1b_wd.json --arm orthblock
    python verify_struct.py --config configs/ctv_1b_ctrl.json --arm control
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from jd_collection import _adapter_scaling, _read_adapter
from merge_subset import subset_authors, write_effective_adapter, _weighted_factor_cat
from struct_bases import (
    author_basis,
    basis_sha256,
    delta_fro2,
    delta_fro2_in,
    energy_in_subspace,
    module_basis,
    seeded_subspace,
)
from train_struct_tv import (
    REPO_DIR,
    STRUCT_ARMS,
    arm_dir,
    check_arm,
    derive_pool,
    load_ctv_config,
    _sha256_file,
)

OWN_ENERGY_MIN = 1.0 - 1e-6
LEAK_MAX = 1e-8
DROP_REL_MAX = 1e-6


def _fro2_cat(pairs):
    """||sum_i B_i A_i||_F^2 for a list of factored terms, via one cat + Grams (float64)."""
    B = torch.cat([b.to(torch.float64) for b, _ in pairs], dim=1)
    A = torch.cat([a.to(torch.float64) for _, a in pairs], dim=0)
    return float(torch.sum((B.t() @ B) * (A @ A.t())))


def empty_slice_check(own_norms, eps):
    """Flag authors whose owned-region delta norm < eps * median (the MemSinks lesson).
    Returns (flagged_authors_sorted, median)."""
    vals = list(own_norms.values())
    med = float(np.median(vals)) if vals else float("nan")
    flagged = sorted(a for a, v in own_norms.items() if v < eps * med)
    return flagged, med


def _geometry(cfg, arm):
    """(struct_seed, pool_size, r_prime, mode) for the verification basis. The control
    arm has no constraint of its own — it is measured against the wd geometry pinned in
    its config's struct_ref block (orthblock reference; a random subspace's chance level
    is the honest null for an unconstrained delta)."""
    if arm in STRUCT_ARMS:
        return cfg["struct_seed"], cfg["pool_size"], cfg["r_prime"], arm
    ref = cfg.get("struct_ref")
    if not ref:
        raise KeyError("control verify needs a struct_ref block "
                       "({struct_seed, pool_size, r_prime}) in the ctrl config")
    return ref["struct_seed"], ref["pool_size"], ref["r_prime"], ref.get("mode", "orthblock")


def verify_arm(cfg, arm, authors=None, drop_authors=None, placebo_author=None,
               out_path=None, eps_empty=1e-3, write_placebos=True):
    """Run every check for one arm; returns the report dict (also written to out_path).

    Check failures never raise — they set pass=False so one run reports everything
    (main() turns a constrained-arm failure into exit 1)."""
    constrained = arm in STRUCT_ARMS
    struct_seed, sp, rp, mode = _geometry(cfg, arm)
    struct_pool = subset_authors(cfg["pool_seed"], sp)
    adir = arm_dir(cfg, arm)

    if authors is None:
        authors = [a for a in derive_pool(cfg)
                   if os.path.isdir(os.path.join(adir, f"shard_{a}"))]
    if not authors:
        raise FileNotFoundError(f"no shard_<author> adapter dirs under {adir}")

    slots_by_author, scale_by_author, cfg_by_author, meta_by_author = {}, {}, {}, {}
    for a in authors:
        d = os.path.join(adir, f"shard_{a}")
        slots, acfg = _read_adapter(d)
        slots_by_author[a] = slots
        scale_by_author[a] = _adapter_scaling(acfg)
        cfg_by_author[a] = acfg
        mp = os.path.join(d, "struct_meta.json")
        meta_by_author[a] = json.load(open(mp)) if os.path.exists(mp) else None
    slot_names = sorted(slots_by_author[authors[0]].keys())

    # Shared per-slot bases, rebuilt from the seed string (author-free — see struct_bases).
    Q_full = {}
    for slot in slot_names:
        d_out = slots_by_author[authors[0]][slot][1].shape[0]
        Q_full[slot] = module_basis(struct_seed, slot, d_out, sp, rp, mode)

    in_pool = [a for a in authors if a in struct_pool]
    idx_of = {a: struct_pool.index(a) for a in in_pool}

    # ---- own energy + owned-region norms (+ control chance report) ----
    own_min, own_per_author, own_norms = 1.0, {}, {}
    span_report = {"per_slot_chance": {}, "mean_e_own": None, "mean_e_span": None} \
        if not constrained else None
    e_own_all, e_span_all = [], []
    for a in in_pool:
        worst, norm2 = 1.0, 0.0
        for slot in slot_names:
            A, B = slots_by_author[a][slot]
            Qa = author_basis(Q_full[slot], idx_of[a], rp)
            e = energy_in_subspace(A, B, scale_by_author[a], Qa)
            worst = min(worst, e)
            norm2 += delta_fro2_in(A, B, scale_by_author[a], Qa)
            e_own_all.append(e)
            if not constrained:
                e_span_all.append(energy_in_subspace(A, B, scale_by_author[a], Q_full[slot]))
        own_per_author[a] = worst
        own_norms[a] = math.sqrt(norm2)
        own_min = min(own_min, worst)
    if not constrained:
        for slot in slot_names:
            d_out = Q_full[slot].shape[0]
            span_report["per_slot_chance"][slot] = {
                "block": rp / d_out, "span": min(1.0, sp * rp / d_out)}
        span_report["mean_e_own"] = float(np.mean(e_own_all)) if e_own_all else None
        span_report["mean_e_span"] = float(np.mean(e_span_all)) if e_span_all else None
        span_report["chance_block_mean"] = float(np.mean(
            [v["block"] for v in span_report["per_slot_chance"].values()]))
        span_report["chance_span_mean"] = float(np.mean(
            [v["span"] for v in span_report["per_slot_chance"].values()]))

    # ---- cross-author leak ----
    # NOT energy_in_subspace: its zero-delta convention (-> 1.0) is right for the own
    # certificate but a zero delta must contribute ZERO leak, so divide explicitly.
    leak_max = 0.0
    for a in in_pool:
        for slot in slot_names:
            A, B = slots_by_author[a][slot]
            tot = delta_fro2(A, B, scale_by_author[a])
            if tot <= 0.0:
                continue
            for b in in_pool:
                if b == a:
                    continue
                Qb = author_basis(Q_full[slot], idx_of[b], rp)
                leak_max = max(leak_max,
                               delta_fro2_in(A, B, scale_by_author[a], Qb) / tot)

    # ---- merge-drop identity on the _weighted_factor_cat SUM ----
    # Full sum via the same function the driver uses; drop/zero variants built in memory
    # from the identical scaled factors (w=1.0, so w*s*B == s*B bitwise).
    drop_report = {"per_author": {}, "max_rel_err": None}
    merged = None
    if len(authors) >= 2:
        dirs = [os.path.join(adir, f"shard_{a}") for a in authors]
        merged, ref_cfg, out_rank, _ = _weighted_factor_cat(dirs, [1.0] * len(authors))
        if drop_authors is None:
            drop_authors = [a for a in cfg["probe_authors"] if a in in_pool] or in_pool[:5]
        for u in drop_authors:
            worst = 0.0
            for slot in slot_names:
                A_m, B_m = merged[slot]
                Qu = author_basis(Q_full[slot], idx_of[u], rp).to(torch.float64)
                B_md = B_m.to(torch.float64)
                B_zero = B_md - Qu @ (Qu.t() @ B_md)
                drop_pairs = [(scale_by_author[b] * slots_by_author[b][slot][1],
                               slots_by_author[b][slot][0])
                              for b in authors if b != u]
                den = _fro2_cat(drop_pairs)
                if den <= 0.0:
                    continue
                diff_pairs = [(B_zero, A_m.to(torch.float64))] + \
                             [(-b, a) for b, a in drop_pairs]
                worst = max(worst, math.sqrt(max(_fro2_cat(diff_pairs), 0.0) / den))
            drop_report["per_author"][u] = worst
        errs = list(drop_report["per_author"].values())
        drop_report["max_rel_err"] = max(errs) if errs else None

    # ---- empty-slice detector ----
    flagged, med = empty_slice_check(own_norms, eps_empty)

    # ---- basis provenance (report-only: byte drift across BLAS builds must not gate) ----
    sha_match = None
    if constrained and any(meta_by_author[a] for a in in_pool):
        sha_match = all(
            meta_by_author[a]["basis_sha256"].get(slot)
            == basis_sha256(author_basis(Q_full[slot], idx_of[a], rp))
            for a in in_pool if meta_by_author[a] and meta_by_author[a].get("basis_sha256")
            for slot in slot_names)

    # ---- placebo adapters (MemSinks owned-vs-random port) ----
    placebos = None
    if write_placebos and merged is not None and in_pool:
        u = placebo_author
        if u is None:
            cands = [a for a in cfg["probe_authors"] if a in in_pool] or in_pool
            u = cands[0]
        if u not in idx_of:
            raise ValueError(f"placebo author {u} not in the struct pool {struct_pool}")
        owned_dir = os.path.join(adir, "placebos", f"placebo_owned_a{u}")
        rand_dir = os.path.join(adir, "placebos", f"placebo_rand_a{u}")
        owned_slots, rand_slots = {}, {}
        e_owned_before, e_owned_after, e_rand_after = [], [], []
        for slot in slot_names:
            A_m, B_m = merged[slot]
            d_out = B_m.shape[0]
            Qu = author_basis(Q_full[slot], idx_of[u], rp).contiguous()
            R = seeded_subspace(f"{struct_seed}:placebo_rand:a{u}:{slot}", d_out, rp)
            B_owned = B_m - Qu @ (Qu.t() @ B_m)
            B_rand = B_m - R @ (R.t() @ B_m)
            owned_slots[slot] = (A_m, B_owned.contiguous())
            rand_slots[slot] = (A_m, B_rand.contiguous())
            e_owned_before.append(energy_in_subspace(A_m, B_m, 1.0, Qu))
            e_owned_after.append(energy_in_subspace(A_m, B_owned, 1.0, Qu))
            e_rand_after.append(energy_in_subspace(A_m, B_rand, 1.0, R))
        ref_cfg = cfg_by_author[authors[0]]
        write_effective_adapter(owned_dir, owned_slots, ref_cfg, out_rank)
        write_effective_adapter(rand_dir, rand_slots, ref_cfg, out_rank)
        placebos = {
            "author": u, "owned_dir": owned_dir, "rand_dir": rand_dir,
            "merged_authors": authors, "out_rank": out_rank,
            "owned_energy_before_mean": float(np.mean(e_owned_before)),
            "owned_energy_after_max": float(np.max(e_owned_after)),
            "rand_energy_after_max": float(np.max(e_rand_after)),
        }
        for d, target in ((owned_dir, "owned"), (rand_dir, "random")):
            with open(os.path.join(d, "placebo_meta.json"), "w") as f:
                json.dump({"type": target, "author": u, "arm": arm, "mode": mode,
                           "merged_authors": authors, "struct_seed": struct_seed,
                           "pool_size": sp, "r_prime": rp, "out_rank": out_rank,
                           "script_sha256": _sha256_file(os.path.abspath(__file__))},
                          f, indent=2)

    checks = {
        "own_energy": {"min": own_min, "per_author": own_per_author,
                       "threshold": OWN_ENERGY_MIN,
                       "pass": (own_min >= OWN_ENERGY_MIN) if constrained else None},
        "cross_leak": {"max": leak_max, "threshold": LEAK_MAX,
                       "pass": (leak_max <= LEAK_MAX) if constrained else None},
        "merge_drop": {**drop_report, "threshold": DROP_REL_MAX,
                       "pass": (drop_report["max_rel_err"] is not None
                                and drop_report["max_rel_err"] <= DROP_REL_MAX)
                       if constrained else None},
        "empty_slice": {"eps": eps_empty, "median_norm": med,
                        "per_author_norm": own_norms, "flagged": flagged,
                        "pass": (not flagged) if constrained else None},
    }
    if span_report is not None:
        checks["control_chance"] = span_report
    ok = all(c["pass"] for c in checks.values()
             if isinstance(c, dict) and c.get("pass") is not None) if constrained else True

    report = {
        "arm": arm, "mode": mode, "constrained": constrained,
        "geometry": {"struct_seed": struct_seed, "pool_size": sp, "r_prime": rp,
                     "d_out_per_slot": {s: Q_full[s].shape[0] for s in slot_names}},
        "pool": struct_pool, "authors_verified": authors,
        "authors_in_struct_pool": in_pool,
        "basis_sha_match": sha_match,
        "checks": checks,
        "placebos": placebos,
        "ok": ok,
        "out_dir": adir,
        "script_sha256": _sha256_file(os.path.abspath(__file__)),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[verify] wrote {out_path}")
    print(f"[verify] {arm}: authors={len(authors)} own_min={own_min:.2e} "
          f"leak_max={leak_max:.2e} drop_max={drop_report['max_rel_err']} "
          f"empty_flagged={flagged} ok={ok}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--arm", required=True, choices=("control",) + STRUCT_ARMS)
    p.add_argument("--authors", default=None,
                   help="comma-separated author ids (default: every pool author on disk)")
    p.add_argument("--drop_authors", default=None,
                   help="authors for the merge-drop identity (default: probes on disk)")
    p.add_argument("--placebo_author", type=int, default=None,
                   help="author whose owned region gets the placebo pair (default: first probe)")
    p.add_argument("--eps_empty", type=float, default=1e-3)
    p.add_argument("--no_placebos", action="store_true")
    p.add_argument("--out", default=None,
                   help="report path (default reports/struct_verify_<arm>.json)")
    args = p.parse_args()

    cfg = load_ctv_config(args.config)
    check_arm(cfg, args.arm)
    authors = [int(x) for x in args.authors.split(",")] if args.authors else None
    drops = [int(x) for x in args.drop_authors.split(",")] if args.drop_authors else None
    out = args.out or os.path.join(REPO_DIR, "reports", f"struct_verify_{args.arm}.json")

    report = verify_arm(cfg, args.arm, authors=authors, drop_authors=drops,
                        placebo_author=args.placebo_author, out_path=out,
                        eps_empty=args.eps_empty, write_placebos=not args.no_placebos)
    if args.arm in STRUCT_ARMS and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
