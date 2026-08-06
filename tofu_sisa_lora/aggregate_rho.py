"""Assemble the residual-fact-recall curve ρ vs R from three eval_entangled_probe JSONs.

ρ = clip((post − floor) / (ceiling − floor), 0, 1), per (R, mode) group and per
channel×surface signal (the same formula unit-tested in test_entangled_facts.py).
  ceiling — planted experts, --drop_shard none
  post    — planted experts, --drop_shard 9 (any policy)
  floor   — clean oracle-B experts, --drop_shard 9

Previously this was a manual report step (ENTANGLED_FACTS_REPORT_2026-07-06.md);
this script closes the gap so every new probe arm (e.g. embed tombstone) gets an
identical, mechanical ρ table.

  python aggregate_rho.py --ceiling C.json --post P.json --floor F.json \
      [--signals served_embedsim_prob_orig ...] --out reports/rho_<arm>.json
"""
from __future__ import annotations

import argparse
import json


def rho(post: float, floor: float, ceiling: float) -> float:
    denom = ceiling - floor
    if abs(denom) < 1e-9:
        return 0.0
    return max(0.0, min(1.0, (post - floor) / denom))


def rho_table(ceiling: dict, post: dict, floor: dict, signals=None) -> dict:
    """{signal: {R{r}_{mode}: {rho, post, floor, ceiling, n}}} over signals present in all
    three runs (or the explicit subset)."""
    common = set(ceiling["aggregates"]) & set(post["aggregates"]) & set(floor["aggregates"])
    keys = sorted(common if signals is None else (common & set(signals)))
    if signals is not None and set(signals) - common:
        raise SystemExit(f"signals missing from one of the runs: {sorted(set(signals) - common)}")
    out = {}
    for sk in keys:
        groups = (set(ceiling["aggregates"][sk]) & set(post["aggregates"][sk])
                  & set(floor["aggregates"][sk]))
        out[sk] = {}
        for g in sorted(groups):
            c = ceiling["aggregates"][sk][g]["mean"]
            p = post["aggregates"][sk][g]["mean"]
            f = floor["aggregates"][sk][g]["mean"]
            out[sk][g] = {"rho": rho(p, f, c), "post": p, "floor": f, "ceiling": c,
                          "n": post["aggregates"][sk][g]["n"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ceiling", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--signals", nargs="*", default=None,
                    help="restrict to these aggregate keys (default: all common)")
    ap.add_argument("--ceiling_channel", default=None,
                    help="take the ceiling from a DIFFERENT channel than post/floor, e.g. "
                         "`expert_max` — its aggregate keys are remapped onto the post channel. "
                         "Use this for a router-INDEPENDENT ceiling (max answer-prob over experts, "
                         "no routing): a magnet router misroutes even with no drop, which collapses "
                         "ceiling≈floor and makes the (post−floor)/(ceiling−floor) ratio degenerate. "
                         "A shared ceiling also makes rho comparable across router families.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {}
    for name in ("ceiling", "post", "floor"):
        with open(getattr(args, name)) as f:
            runs[name] = json.load(f)

    if args.ceiling_channel:
        post_ch = runs["post"].get("channels") or []
        if len(post_ch) != 1:
            raise SystemExit(f"--ceiling_channel needs the post run to declare exactly one "
                             f"channel; got {post_ch}")
        src, dst = args.ceiling_channel, post_ch[0]
        remapped = {k.replace(src, dst, 1): v
                    for k, v in runs["ceiling"]["aggregates"].items() if k.startswith(src)}
        if not remapped:
            raise SystemExit(f"ceiling run has no '{src}_*' aggregates "
                             f"(has: {sorted(runs['ceiling']['aggregates'])[:5]})")
        runs["ceiling"] = dict(runs["ceiling"], aggregates=remapped)

    table = rho_table(runs["ceiling"], runs["post"], runs["floor"], args.signals)
    out = {"ceiling": args.ceiling, "post": args.post, "floor": args.floor,
           "ceiling_channel": args.ceiling_channel,
           "post_drop_shard": runs["post"].get("drop_shard"),
           "post_embed_policy": runs["post"].get("embed_policy"), "rho": table}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate_rho] -> {args.out}")
    for sk, groups in table.items():
        print(f"  {sk}: " + "  ".join(f"{g}={d['rho']:.3f}" for g, d in sorted(groups.items())))


if __name__ == "__main__":
    main()
