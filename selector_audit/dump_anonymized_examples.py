#!/usr/bin/env python3
"""Twenty (original, name-stripped, gold) triples, for judging whether anonymised TOFU questions
are still natural and answerable in principle.

This is a VIEWER, not a new transform. The questions come from `analyze_router_shift`'s
`build_eval_rows` + `build_conditions` — the same 800 rows, the same `strip_names`, the same
attacker and seed that produced findings 4 and 5. If this file ever disagrees with what those
findings were measured on, it is this file that is wrong.

The 18 authors from whom no name can be extracted are the reason a `no-op` column exists: for
those rows `name_stripped` IS the original, so an eyeball sample that happened to draw them would
suggest the transform does nothing. They are marked rather than dropped.

  python dump_anonymized_examples.py --out outputs/anonymized_examples.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "tofu_sisa_lora"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


def _full_name_spans(q: str, nms: list) -> list:
    """The author's name as it appears in `q`, extended through adjacent hyphenated letters.

    `router._extract_author_names` yields "Hsiao Yun" for *Hsiao Yun-Hwa* and "Aisha Al" for
    *Aisha Al-Hamad* — it splits on the hyphen. `strip_names` then removes exactly what it was
    given, so the other half survives. Recovering the FULL span is what makes that measurable.
    """
    import re
    spans = []
    for nm in sorted([n for n in nms if n], key=len, reverse=True):
        for m in re.finditer(re.escape(nm), q, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            while s > 0 and (q[s - 1].isalpha() or q[s - 1] == "-"):
                s -= 1
            while e < len(q) and (q[e].isalpha() or q[e] == "-"):
                e += 1
            spans.append(q[s:e])
    return spans


def residual_fragments(original: str, stripped: str, nms: list) -> list:
    """Name parts that survive stripping — [] when the row is genuinely anonymised."""
    import re
    if original == stripped:
        return []
    lower = {n.lower() for n in (nms or [])}
    resid = set()
    for fn in _full_name_spans(original, nms):
        for part in re.split(r"[^A-Za-z]+", fn):
            if len(part) >= 2 and part.lower() not in lower and part in stripped:
                resid.add(part)
    return sorted(resid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "outputs",
                                                  "anonymized_examples.md"))
    ap.add_argument("--n_forget", type=int, default=10)
    ap.add_argument("--n_retain", type=int, default=10)
    ap.add_argument("--attacker_id", type=int, default=0,
                    help="Kept only so build_conditions matches finding 5 exactly; unused here.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"))
    args = ap.parse_args()
    if not args.hf_home:
        raise SystemExit("--hf_home or $HF_HOME is required")

    from analyze_router_shift import build_eval_rows, build_conditions

    full, rows, authors, paras = build_eval_rows(args.hf_home)
    cond, names, attacker_name = build_conditions(full, rows, authors, paras,
                                                  args.attacker_id, args.hf_home)
    is_forget = np.isin(authors, np.arange(180, 200))
    print(f"[examples] {len(rows)} rows ({int(is_forget.sum())} forget / "
          f"{int((~is_forget).sum())} retain); attacker={attacker_name!r}", flush=True)

    # One row per author, so twenty examples are twenty people rather than one person's twenty
    # questions. Sampling rows directly would draw ~5 rows from the same author at this size.
    rng = np.random.default_rng(args.seed)

    def _pick(mask, n):
        pool = np.flatnonzero(mask)
        by_author: dict[int, list[int]] = {}
        for i in pool:
            by_author.setdefault(int(authors[i]), []).append(int(i))
        chosen_authors = rng.choice(sorted(by_author), size=min(n, len(by_author)),
                                    replace=False)
        return [int(rng.choice(by_author[int(a)])) for a in sorted(chosen_authors)]

    picks = _pick(is_forget, args.n_forget) + _pick(~is_forget, args.n_retain)

    n_noop = sum(1 for i in picks if cond["name_stripped"][i] == cond["original"][i])
    total_noop = sum(1 for i in range(len(rows))
                     if cond["name_stripped"][i] == cond["original"][i])

    # How often does stripping leave an identifying fragment behind? This is not cosmetic:
    # "-Hamad" is a whole surname, and a row that keeps it is not anonymised.
    frag = {i: residual_fragments(cond["original"][i], cond["name_stripped"][i],
                                  names.get(int(authors[i]), []))
            for i in range(len(rows))}
    total_frag = sum(1 for v in frag.values() if v)
    para_frag = sum(1 for i in range(len(rows))
                    if residual_fragments(cond["paraphrase"][i], cond["para_stripped"][i],
                                          names.get(int(authors[i]), [])))
    para_noop = sum(1 for i in range(len(rows))
                    if cond["para_stripped"][i] == cond["paraphrase"][i])
    print(f"[examples] name_stripped: {total_noop} no-op, {total_frag} keep a fragment "
          f"=> {(total_noop + total_frag) / len(rows):.1%} not anonymised", flush=True)

    lines = [
        "# Anonymised TOFU questions — 20 examples",
        "",
        "The `name_stripped` transform used throughout finding 4, applied to the same 800-row "
        "evaluation set (400 forget / 400 retain) that produced it.",
        "",
        f"- Source: `tofu_sisa_lora/analyze_router_shift.py` — `build_eval_rows` + "
        f"`build_conditions` -> `strip_names` (line 90).",
        f"- Sample: {args.n_forget} forget + {args.n_retain} retain rows, one question per author, "
        f"seed {args.seed}.",
        f"- **No-op rows in this sample: {n_noop}/20.**",
        "",
        "## The transform does not fully anonymise",
        "",
        f"Across all 800 rows, **{(total_noop + total_frag) / len(rows):.1%} still carry a name**:",
        "",
        "| | rows | share |",
        "|---|---|---|",
        f"| unchanged (no name extractable) | {total_noop} | {total_noop / len(rows):.1%} |",
        f"| stripped but a name fragment survives | {total_frag} | {total_frag / len(rows):.1%} |",
        f"| cleanly stripped | {len(rows) - total_noop - total_frag} | "
        f"{(len(rows) - total_noop - total_frag) / len(rows):.1%} |",
        "",
        f"`para_stripped` inherits the same defect ({para_noop} no-op + {para_frag} fragment = "
        f"{(para_noop + para_frag) / len(rows):.1%}).",
        "",
        "The cause is upstream of `strip_names`: `router._extract_author_names` splits hyphenated "
        "names, yielding `\"Hsiao Yun\"` for *Hsiao Yun-Hwa*, `\"Aisha Al\"` for *Aisha Al-Hamad*, "
        "`\"Yeon Park\"` for *Ji-Yeon Park*. `strip_names` removes exactly what it was handed and "
        "leaves `-Hwa`, `-Hamad`, `Ji-` behind — and `-Hamad` is a complete surname. So "
        "\"no *extracted* name form remains\" is not the same claim as \"no name remains\".",
        "",
        "Two consequences for any name-free number: it is an **upper bound** (residual signal "
        "still helps the selector), and the stripped questions are ungrammatical stubs that "
        "models complete arbitrarily — asked `\"Are the details of 's birth documented?\"` the "
        "frozen base answers about *Jesus'* birth — so part of the measured drop is broken "
        "grammar rather than lost identity.",
        "",
        "`gold` is the reference answer, unchanged by the transform; it is what a correct answer "
        "would have to recover from the anonymised question.",
        "",
    ]

    for section, sel in (("Forget authors (180–199, deleted)", picks[:args.n_forget]),
                         ("Retain authors (0–179, kept)", picks[args.n_forget:])):
        lines += [f"## {section}", ""]
        for i in sel:
            row = int(rows[i])
            a = int(authors[i])
            noop = cond["name_stripped"][i] == cond["original"][i]
            nm = ", ".join(names.get(a, [])) or "(no name extracted)"
            if noop:
                flag = "  — **transform is a no-op**"
            elif frag.get(i):
                flag = f"  — **leaves `{'`, `'.join(frag[i])}` behind**"
            else:
                flag = ""
            lines += [
                f"### author {a} · row {row}{flag}",
                "",
                f"- names extracted: {nm}",
                f"- **original:** {cond['original'][i]}",
                f"- **name_stripped:** {cond['name_stripped'][i]}",
                f"- *gold:* {full[row]['answer']}",
                "",
            ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[examples] -> {args.out}  ({n_noop}/20 no-op, {total_noop}/{len(rows)} overall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
