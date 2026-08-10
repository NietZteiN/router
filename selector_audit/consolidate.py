"""One report for the whole selector_audit campaign — CPU, offline, missing inputs are fine.

Runs unattended as the last stage of the overnight chain, so whatever has landed by morning is
summarized in one file and whatever has not is listed as pending rather than silently absent.
Every number it prints is read from a producer's own JSON; it computes nothing new except the
question-type breakdown (H15), which needs no model.

  python consolidate.py --pool_dir <k200 e25 pool> --repo <repo root> --out_md STATUS.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# A question that asks for an identity invites a NAME, and a wrong expert supplies one. The
# CSAR pilot's 0.460-vs-0.290 gap tracked exactly this, so the breakdown is worth having
# explicitly rather than as a q0-4 proxy.
_IDENTITY_RE = re.compile(
    r"\b(full name|what is the name|who is the author|name of the author|"
    r"what is the (?:real )?name|born in .* on |what is the full)", re.IGNORECASE)


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def csar_rows(rl_dir: str) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(rl_dir, "csar_*.json"))):
        d = _load(p)
        if not d or "strategies" not in d:
            continue
        stem = os.path.basename(p)[len("csar_"):-len(".json")]
        for s, v in d["strategies"].items():
            out.append({"run": stem, "strategy": s, "n": v.get("n"),
                        "CSAR": v.get("CSAR"),
                        "refusal": (v.get("rates") or {}).get("refusal"),
                        "base_generic": (v.get("rates") or {}).get("base_generic"),
                        "unattributable": (v.get("rates") or {}).get("unattributable")})
    return out


def question_type_breakdown(rl_dir: str) -> list:
    """H15: is CSAR really 'the router supplies the wrong name when asked for a name'?"""
    rows = []
    for p in sorted(glob.glob(os.path.join(rl_dir, "csar_*.json"))):
        c = _load(p)
        stem = os.path.basename(p)[len("csar_"):-len(".json")]
        # The dump is named for the SAME stem as its csar output. Preferring an approximate
        # match over the exact one silently paired `..._centroid_sbert-random` with the
        # gold-form dump, whose questions belong to a different run — the H15 rows then
        # duplicated the gold-form row and the `random` strategy vanished from the table
        # because none of its rows could be found. Exact name first, always.
        dump = os.path.join(rl_dir, "sibling_content_" + stem + ".json")
        if not os.path.exists(dump):
            dump = os.path.join(rl_dir,
                                "sibling_content_" + stem.split("_centroid")[0] + ".json")
        a = _load(dump)
        if not c or not a:
            continue
        # and refuse to score a pair whose strategy sets disagree: that means the wrong dump,
        # not a partial one, and it would produce numbers computed from another run's text
        if set(c["strategies"]) != set(a.get("strategies", {})):
            print(f"[consolidate] SKIP question-type for {stem}: strategies "
                  f"{sorted(c['strategies'])} != dump {sorted(a.get('strategies', {}))} "
                  f"({os.path.basename(dump)})")
            continue
        qtext = {(s, r["row"]): r.get("question", "")
                 for s, b in a.get("strategies", {}).items() for r in b.get("per_question", [])}
        for s, v in c["strategies"].items():
            ident = [0, 0]
            other = [0, 0]
            for x in v.get("rows", []):
                q = qtext.get((s, x["row"]), "")
                bucket = ident if _IDENTITY_RE.search(q) else other
                bucket[0] += (x["category"] == "cross_source")
                bucket[1] += 1
            if ident[1] and other[1]:
                rows.append({"run": stem, "strategy": s,
                             "identity_q": ident[0] / ident[1], "n_identity": ident[1],
                             "other_q": other[0] / other[1], "n_other": other[1]})
    return rows


def eval_rows(smoke_dir: str) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(smoke_dir, "routed_*.json"))):
        if p.endswith(".progress.json"):
            continue
        d = _load(p)
        if not d or "model_utility" not in d:
            continue
        ra = d.get("route_audit") or {}
        out.append({"arm": os.path.basename(p)[:-len(".json")],
                    "mu": d.get("model_utility"), "fq": d.get("forget_quality"),
                    "forget_rouge": d.get("forget_rouge"), "retain_rouge": d.get("retain_rouge"),
                    "real_rouge": d.get("real_rouge"), "world_rouge": d.get("world_rouge"),
                    "audit_ok": ra.get("ok") if ra else None,
                    "stats": d.get("route_stats")})
    return out


def mia_rows(mia_dir: str) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(mia_dir, "*.json"))):
        d = _load(p)
        if not d or "per_attack" not in d:
            continue
        per = d["per_attack"]
        aucs = {k: (v.get("auc") if isinstance(v, dict) else v) for k, v in per.items()}
        out.append({"arm": os.path.basename(p)[:-len(".json")], "aucs": aucs,
                    "n_member": d.get("n_member"), "n_holdout": d.get("n_holdout")})
    return out


def family_rows(pool_dirs: list) -> list:
    """Behavioral / feature-space audit cells, if the wave has landed."""
    out = []
    for pd in pool_dirs:
        for p in sorted(glob.glob(os.path.join(pd, "results", "router_leak", "rl_family_*.json"))):
            d = _load(p)
            if not d or "strategies" not in d:
                continue
            for s, e in d["strategies"].items():
                sc = (e.get("self_check") or {})
                out.append({"pool": os.path.basename(pd), "file": os.path.basename(p),
                            "strategy": s,
                            "self_check": f"{sc.get('passed')}/{sc.get('n')}"
                                          if sc else "—",
                            "full_top1_acc": e.get("full_top1_acc"),
                            "cells": sorted((e.get("cells") or {}).keys())})
    return out


def write_md(res: dict, path: str) -> None:
    L = ["# selector_audit — overnight status", "",
         f"Generated by `selector_audit/consolidate.py`. Inputs that had not landed are listed "
         f"under **Pending** rather than omitted.", ""]

    L += ["## Serving arms (TOFU metrics)", "",
          "| arm | mu | fq | forget_R | retain_R | real_R | world_R | route audit |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["eval"]:
        L.append(f"| `{r['arm']}` | {_f(r['mu'],4)} | {_f(r['fq'],4)} | {_f(r['forget_rouge'])} | "
                 f"{_f(r['retain_rouge'])} | {_f(r['real_rouge'])} | {_f(r['world_rouge'])} | "
                 f"{'ok' if r['audit_ok'] else ('FAILED' if r['audit_ok'] is False else '—')} |")
    if not res["eval"]:
        L.append("| _none yet_ | | | | | | | |")

    L += ["", "## CSAR", "",
          "| run | strategy | n | **CSAR** | refusal | base-generic | unattributable |",
          "|---|---|---|---|---|---|---|"]
    for r in res["csar"]:
        L.append(f"| `{r['run']}` | {r['strategy']} | {r['n']} | **{_f(r['CSAR'])}** | "
                 f"{_f(r['refusal'])} | {_f(r['base_generic'])} | {_f(r['unattributable'])} |")
    if not res["csar"]:
        L.append("| _none yet_ | | | | | | |")

    if res["qtype"]:
        L += ["", "### H15 — is it 'the wrong name when asked for a name'?", "",
              "| run | strategy | identity questions | other questions |",
              "|---|---|---|---|"]
        for r in res["qtype"]:
            L.append(f"| `{r['run']}` | {r['strategy']} | {_f(r['identity_q'])} "
                     f"(n={r['n_identity']}) | {_f(r['other_q'])} (n={r['n_other']}) |")

    L += ["", "## Privacy (composed-model MIA)", "",
          "| arm | " + " | ".join(sorted({a for r in res["mia"] for a in r["aucs"]})) + " |"]
    attacks = sorted({a for r in res["mia"] for a in r["aucs"]})
    L.append("|---" * (len(attacks) + 1) + "|")
    for r in res["mia"]:
        L.append(f"| `{r['arm']}` | " + " | ".join(_f(r["aucs"].get(a)) for a in attacks) + " |")
    if not res["mia"]:
        L.append("| _none yet_ | |")

    L += ["", "## Router-family audits", "",
          "| pool | file | strategy | self_check | full top-1 acc | cells |",
          "|---|---|---|---|---|---|"]
    for r in res["family"]:
        L.append(f"| {r['pool']} | `{r['file']}` | {r['strategy']} | {r['self_check']} | "
                 f"{_f(r['full_top1_acc'])} | {len(r['cells'])} |")
    if not res["family"]:
        L.append("| _none yet_ | | | | | |")

    if res["pending"]:
        L += ["", "## Pending", ""] + [f"- `{p}`" for p in res["pending"]]
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--extra_pool", nargs="*", default=[])
    ap.add_argument("--expect", nargs="*", default=[],
                    help="paths that SHOULD exist; any that do not are listed under Pending")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rl = os.path.join(args.pool_dir, "results", "router_leak")
    res = {
        "eval": eval_rows(os.path.join(args.pool_dir, "results", "smoke")),
        "csar": csar_rows(rl),
        "qtype": question_type_breakdown(rl),
        "mia": mia_rows(os.path.join(args.pool_dir, "results", "mia")),
        "family": family_rows([args.pool_dir] + list(args.extra_pool)),
        "pending": [p for p in args.expect if not os.path.exists(p)],
    }
    write_md(res, args.out_md)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
    print(f"[consolidate] eval={len(res['eval'])} csar={len(res['csar'])} "
          f"mia={len(res['mia'])} family={len(res['family'])} "
          f"pending={len(res['pending'])} -> {args.out_md}")
    for p in res["pending"]:
        print(f"  PENDING {p}")


if __name__ == "__main__":
    main()
