"""Autonomous collector for router_leak Waves 4+5 (CPU, no GPU, no inference).

Runs as a SLURM job chained after every Wave-4/5 producer so the NUMBERS are assembled by the
cluster itself — independent of any interactive session. Emits one summary JSON + a markdown
table; the narrative write-up (report prose, log entry) is done afterwards by a human/agent from
this summary, so nothing is lost if the driving session goes away.

Collects:
  rho_by_family   — Mode-B residual-fact-recall per router family, recomputed against the SHARED
                    router-independent `expert_max` ceiling (fixes the magnet-router degeneracy
                    where ceiling≈floor; also makes rho comparable ACROSS families).
  ppl_seal        — the router-native abstain seal: rho + abstain rate (route == -1).
  disclosure      — per-RUNG deletion-disclosure AUC (shard / author / name).
  mia             — composed-model MIA on the embed-routed arms (sibling = leak, tombstone = seal).

  python collect_wave45.py --rl_dir ... --ent_out ... --out_json ... --out_md ...
"""
from __future__ import annotations

import argparse
import json
import os

FAMILIES = ["centroid_sbert", "centroid_lm", "ppl", "activation_norm", "attn_norm", "logit_div"]


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _rho_vs_shared_ceiling(ceiling, post, floor):
    """rho table using the shared expert_max ceiling remapped onto the served_embedsim channel
    (the same remap `aggregate_rho.py --ceiling_channel` performs)."""
    from aggregate_rho import rho_table
    post_ch = (post.get("channels") or ["served_embedsim"])[0]
    remapped = {k.replace("expert_max", post_ch, 1): v
                for k, v in ceiling["aggregates"].items() if k.startswith("expert_max")}
    if not remapped:
        return None
    return rho_table(dict(ceiling, aggregates=remapped), post, floor)


def _r8_verbatim(tbl):
    """The headline cell: verbatim-surface rho at R=8 (the replicated-fact privacy channel)."""
    if not tbl:
        return None
    for sk in tbl:
        if "prob" in sk and "orig" in sk:
            for g, v in tbl[sk].items():
                if g.startswith("R8") and "verbatim" in g:
                    return round(v["rho"], 4)
    return None


def _abstain_rate(run):
    """Fraction of probes the seal diverted to base (route == -1)."""
    rows = (run or {}).get("per_fact") or []
    vals = [v for r in rows for k, v in r.items() if k.startswith("served_embedsim_route_")]
    return round(sum(1 for v in vals if v == -1) / len(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rl_dir", required=True)
    ap.add_argument("--ent_out", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    a = ap.parse_args()

    out = {"missing": [], "rho_by_family": {}, "ppl_seal": {}, "disclosure": {}, "mia": {}}

    ceiling = _load(f"{a.ent_out}/ceiling_expert_max.json")
    if ceiling is None:
        out["missing"].append("ceiling_expert_max.json (shared ceiling) — rho not recomputable")

    # ---- Mode-B rho per family, against the shared ceiling -------------------------------
    for fam in FAMILIES:
        post = _load(f"{a.ent_out}/postdrop_embedsim_fam_{fam}.json")
        floor = _load(f"{a.ent_out}/floor_embedsim_fam_{fam}.json")
        if not (post and floor):
            out["missing"].append(f"{fam}: post/floor")
            continue
        if ceiling is None:
            continue
        tbl = _rho_vs_shared_ceiling(ceiling, post, floor)
        out["rho_by_family"][fam] = {"rho_R8_verbatim": _r8_verbatim(tbl), "table": tbl}

    # ---- ppl-native seal ------------------------------------------------------------------
    sp = _load(f"{a.ent_out}/postdrop_embedsim_pplseal.json")
    sf = _load(f"{a.ent_out}/floor_embedsim_pplseal.json")
    if sp and sf and ceiling is not None:
        tbl = _rho_vs_shared_ceiling(ceiling, sp, sf)
        out["ppl_seal"] = {"rho_R8_verbatim": _r8_verbatim(tbl),
                           "abstain_rate": _abstain_rate(sp), "table": tbl}
    else:
        out["missing"].append("ppl seal post/floor")

    # ---- per-rung deletion disclosure ------------------------------------------------------
    d = _load(f"{a.rl_dir}/rl_centroid_k10_rungs.json")
    if d and "disclosure" in d:
        b = d["disclosure"]
        out["disclosure"] = {"shard": b.get("auc_forget_vs_holdout"),
                             "author": b.get("auc_forget_vs_holdout_author"),
                             "name": b.get("auc_forget_vs_holdout_name")}
    else:
        out["missing"].append("rl_centroid_k10_rungs.json (per-rung disclosure)")

    # ---- composed MIA on the embed-routed arms ---------------------------------------------
    for arm in ("sibling", "tombstone"):
        m = _load(f"{a.rl_dir}/mia_embedrouted_{arm}_del9.json")
        if not m:
            out["missing"].append(f"mia_embedrouted_{arm}_del9.json")
            continue
        pa = m.get("per_attack") or {}
        out["mia"][arm] = {k: (v.get("auc") if isinstance(v, dict) else v) for k, v in pa.items()}

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2)

    L = ["# router_leak Waves 4+5 — collected numbers", "",
         "Assembled by `collect_wave45.py` (CPU SLURM job, no inference). rho is recomputed "
         "against the SHARED router-independent `expert_max` ceiling.", "",
         "## Mode-B rho per router family (verbatim surface, R=8)", "",
         "| family | rho@R8 (shared ceiling) |", "|---|---|"]
    for fam in FAMILIES:
        v = out["rho_by_family"].get(fam, {}).get("rho_R8_verbatim")
        L.append(f"| {fam} | {'—' if v is None else v} |")
    s = out["ppl_seal"]
    L += ["", "## ppl-native seal (H-SEAL-PPL)", "",
          f"- sealed rho@R8 verbatim: **{s.get('rho_R8_verbatim', '—')}**",
          f"- abstain rate (probes diverted to base): **{s.get('abstain_rate', '—')}**",
          "- bar: CONFIRM (seal misses replicated facts) if rho >= 0.5; REFUTE if it collapses "
          "toward the author-tombstone's 0.047.",
          "", "## Deletion-disclosure AUC per rung (H-DISC-RUNG)", "",
          "| rung | AUC | prior catch |", "|---|---|---|",
          f"| shard | {out['disclosure'].get('shard','—')} | 0.605 |",
          f"| author | {out['disclosure'].get('author','—')} | 0.963 |",
          f"| name | {out['disclosure'].get('name','—')} | 0.703 |",
          "", "## Composed MIA on the embed-routed arms (H-MIA-ROUTER)", "",
          "| arm | " + " | ".join(["loss", "min_k", "min_k++", "zlib"]) + " |",
          "|---|---|---|---|---|"]
    for arm in ("sibling", "tombstone"):
        r = out["mia"].get(arm, {})
        L.append(f"| {arm} | " + " | ".join(str(r.get(k, "—")) for k in
                                            ("loss", "min_k", "min_k++", "zlib")) + " |")
    L += ["", "Reference points: oracle floor 0.379; exact module-drop routerkey 0.375; "
              "ramole-embed 0.353 (leaked yet MIA-blind).", ""]
    if out["missing"]:
        L += ["## Missing inputs", ""] + [f"- {m}" for m in out["missing"]]
    with open(a.out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[collect_wave45] -> {a.out_json} + {a.out_md}")
    print(f"  families with rho: {sorted(out['rho_by_family'])}")
    print(f"  missing: {out['missing'] or 'none'}")


if __name__ == "__main__":
    main()
