#!/usr/bin/env python3
"""H15 — is CSAR mostly "the router supplied the wrong NAME when asked for a name"?

That objection is the first one a referee raises about §4.3: if cross-source attribution is just
the served answer emitting the survivor's name in place of the deleted author's, the harm is a
naming glitch, not the transfer of a stranger's biography.

`csar.py` already records WHICH survivor facts it matched in each answer (`hits`), so the question
can be answered directly rather than inferred from a question-type classifier: split every
cross_source row by whether its hits are exhausted by name-forms of the routed survivor.

  substantive  at least one hit that is NOT a name-form of the survivor -- a title, place,
               award, occupation. The conservative CSAR, and the number §4.3 should quote.
  name_only    every hit is a name-form. Still misattribution of IDENTITY, so it is reported as
               its own row rather than folded in or discarded.

Also splits by question POSITION, because TOFU orders each author's 20 questions with the
identity-seeking ones first (the q0-q4 sampling bias of 2026-08-07). If the harm were a naming
artifact it would live in q0-q4 and vanish after.

⚠ DIRECTIONAL LIMIT. `router._extract_author_names` yields nothing for 18 of 200 authors, and a
hit on such a survivor cannot be classified -- it falls through to `substantive`. So substantive
is an UPPER bound and name_only a LOWER bound, and `unclassifiable_frac` is printed per cell so a
reader can see when that slack is large enough to matter. It is small gold-form (0.08-0.16) and
NOT small under name_stripped (0.27-0.33), where these numbers should not be quoted.

This decomposes the classifier's OWN output and inherits its errors. It does not substitute for
the hand labels the pre-registration requires.

Usage:
  python csar_decompose.py --csar_json A.json [B.json ...] --hf_home DIR --out_json J --out_md M
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tofu_sisa_lora")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def author_name_forms(hf_home: str, n_authors: int = 200) -> dict:
    """{author_id: [lowercased extracted name forms]} — the same extractor the routers use."""
    from datasets import load_dataset
    from router import _extract_author_names
    full = load_dataset("locuslab/TOFU", "full")["train"]
    return {a: [n.lower() for n in
                _extract_author_names([full[a * 20 + w]["question"] for w in range(20)])]
            for a in range(n_authors)}


def is_name_hit(hit: str, survivor: int, names: dict) -> bool:
    """Is this matched fact just a form of the routed survivor's own name?

    Token-wise and substring-either-direction because the fact index stores a multiword name AND
    its parts AND their possessives ("hsiao yun-hwa", "yun-hwa", "yun-hwa's"), so an equality test
    would count most of them as substantive.
    """
    h = hit.lower().strip()
    if h.endswith("'s"):
        h = h[:-2].strip()
    for nm in names.get(survivor, []):
        if not nm:
            continue
        for tok in [nm] + nm.split():
            if len(tok) > 2 and (h == tok or h in tok or tok in h):
                return True
    return False


def decompose(block: dict, names: dict, noname: set, position=None) -> dict:
    rows = block["rows"]
    if position is not None:
        rows = [r for r in rows if position(r["row"] - r["author"] * 20)]
    n = len(rows)
    cs = [r for r in rows if r["category"] == "cross_source"]
    subst = [r for r in cs if any(not is_name_hit(h, r["survivor"], names) for h in r["hits"])]
    nameonly = [r for r in cs if r["hits"]
                and all(is_name_hit(h, r["survivor"], names) for h in r["hits"])]
    unclass = [r for r in cs if r["survivor"] in noname]
    f = lambda x: (len(x) / n) if n else float("nan")
    return {"n": n, "csar": f(cs), "substantive": f(subst), "name_only": f(nameonly),
            "substantive_share_of_csar": (len(subst) / len(cs)) if cs else float("nan"),
            "unclassifiable_frac": (len(unclass) / len(cs)) if cs else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csar_json", nargs="+", required=True)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    names = author_name_forms(args.hf_home)
    noname = {a for a, v in names.items() if not v}
    print(f"[csar_decompose] {len(noname)} of {len(names)} authors have no extractable name — "
          f"hits on those survivors cannot be classified and count as substantive.")

    out = {"n_authors_without_name": len(noname), "arms": []}
    for path in args.csar_json:
        j = json.load(open(path))
        arm = os.path.basename(path)[:-5]
        for strat, block in j["strategies"].items():
            rec = {"arm": arm, "strategy": strat, "all": decompose(block, names, noname)}
            rec["q0_q4"] = decompose(block, names, noname, position=lambda w: w < 5)
            rec["q5_q19"] = decompose(block, names, noname, position=lambda w: w >= 5)
            out["arms"].append(rec)
            a = rec["all"]
            print(f"  {arm:44s} {strat:15s} CSAR={a['csar']:.4f} "
                  f"substantive={a['substantive']:.4f} name_only={a['name_only']:.4f} "
                  f"unclassifiable={a['unclassifiable_frac']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# H15 — is CSAR mostly a swapped name?", "",
         "`substantive` = at least one matched fact that is NOT a name-form of the routed "
         "survivor. It is the conservative CSAR and the number §4.3 should quote. `name_only` is "
         "still misattribution of identity, reported separately rather than folded in.", "",
         f"⚠ {len(noname)}/200 authors have no extractable name; a hit on such a survivor falls "
         "through to `substantive`, so **substantive is an upper bound and name_only a lower "
         "bound**. Read `unclassifiable` before quoting a cell — above ~0.2 the cell is not "
         "quotable at this precision.", "",
         "| arm | strategy | slice | n | CSAR | substantive | name-only | unclassifiable |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out["arms"]:
        for slc in ("all", "q0_q4", "q5_q19"):
            c = r[slc]
            L.append(f"| `{r['arm']}` | {r['strategy']} | {slc} | {c['n']} | {c['csar']:.4f} | "
                     f"**{c['substantive']:.4f}** | {c['name_only']:.4f} | "
                     f"{c['unclassifiable_frac']:.3f} |")
    open(args.out_md, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
