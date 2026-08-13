# paper/followup — "Deleted from the Router, Not from the Model"

**Status: DRAFT.** This is the manuscript for the MUSR follow-up, written from
[`../../SELECTOR_AUDIT_REPORT.md`](../../SELECTOR_AUDIT_REPORT.md). Unlike the AAAI submission
(which is deliberately absent from this repo — see [`../README.md`](../README.md) and the
`paper/pdf/` + `paper/tex/` entries in [`../../.gitignore`](../../.gitignore)), this draft is our
own unsubmitted work and carries no distribution notice. It is committed so the numbers and the
prose stay in one history.

> If this repo's public visibility makes an in-tree draft undesirable, moving this directory to
> `paper/tex/` makes it gitignored with no other change.

| file | what it is |
|---|---|
| `main.tex` | the draft. `article` class on purpose, so it builds with no venue `.sty` present |
| `refs.bib` | 23 entries transcribed from [`../../papers/RELATED_WORK.md`](../../papers/RELATED_WORK.md) — edit both or they drift |

## Building

**Build out-of-tree.** `test_repo_selfcontained.py` bans `.pdf` *everywhere* under `paper/`,
including this directory — that ban is what keeps the LaTeX exemption from becoming a hole the
AAAI artifact could slip through (see `test_manuscript_absent`). So a build that writes `main.pdf`
next to `main.tex` will fail the repo gate. Write it somewhere else:

```bash
OUT=$(mktemp -d)
cp refs.bib "$OUT/"
pdflatex -output-directory="$OUT" main.tex \
  && (cd "$OUT" && bibtex main) \
  && pdflatex -output-directory="$OUT" main.tex \
  && pdflatex -output-directory="$OUT" main.tex
echo "$OUT/main.pdf"
```

**No TeX distribution is installed on this cluster**, so the draft has never been compiled. It has
been checked structurally instead — environments balanced, braces balanced, every `\cite` key
present in `refs.bib`, every `\ref` backed by a `\label`, no unescaped `_`, and every `tabular` row
matching its column spec. Re-run that check on any machine with Python:

```bash
python3 tools/check_tex.py main.tex refs.bib     # see below; kept with the draft
```

Expect first-compile warnings that the checker cannot see: overfull boxes, and `natbib`/`hyperref`
option clashes if a venue class is swapped in.

## Draft conventions that are not decoration

- **`\blocked{...}` renders in red.** It marks a claim not yet supported by a completed
  measurement. There is currently one, and it gates the whole of §5: the ~300 hand labels
  validating the CSAR classifier. **No CSAR number may go to a venue until those labels exist**,
  and the macro is loud so that the caveat cannot quietly fall off during editing — which is the
  exact failure mode the campaign kept catching in itself.
- **`\todonote{...}` renders in blue** for editorial gaps (artifact URL, the withheld MIA column).
- The paper states what it does **not** claim (§11) as a numbered list, because four of the
  campaign's hypotheses were pre-registered and then refuted, and two published readings were
  corrected by the campaign's own re-checks.

## Claim → artifact map

Every number in the draft traces to a file in this repo. `<pool>` =
`Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` under `$TOFU_CKPT_STORE`.

| §  | claim | artifact |
|---|---|---|
| 5 (`sec:blind`) | 8-arm destination table, utility identical at 0.8009, route audits | `<pool>/results/extended/routed_{reroute_f10_s*,oracle_del_f10}.json` |
| 5 | paired spread CI [0.2245, 0.6975], P(spread>0.25)=0.961, ±0.35 per cell, 1/lcm lattice | `tofu_sisa_lora/reports/h29_forget_quality_ci.{json,md}` ← `selector_audit/bootstrap_fq.py` |
| 6 (`sec:csar`) | CSAR / own-disclosure / refusal per router × transform, random floor | `<pool>/results/router_leak/csar_k200_f10_qpa20*.{json,md}` ← `submit_csar_audit.sh` |
| 6 | substantive vs name-only decomposition, non-identity slice | `tofu_sisa_lora/reports/h15/csar_decompose.{json,md}` ← `selector_audit/csar_decompose.py` |
| 7 (`sec:destinations`) | 400/400 reassigned, busiest share, n_eff | `tofu_sisa_lora/reports/orphan_destinations.md` |
| 7 | magnet refutation, key_tfidf/indirect saturation 0.902, RDR 0.000 vs 0.092 | `tofu_sisa_lora/analyze_sequential_deletion.py` · [log](../../log/selector_audit/2026-08-10_magnet-saturation-and-rdr.md) |
| 8 (`sec:detect`) | granularity ladder, name-stripped flattening, probe AUC 0.990 / +0.001 lift | `tofu_sisa_lora/reports/{granularity_ladder,router_probe_*}.{json,md}` |
| 8 | transforms, `para_stripped`, shuffle control, key_exact 0.025 | `tofu_sisa_lora/reports/h30/router_shift_h30.{json,md}` |
| 8 | name-injection steering (97.7 / 31.7 / 3.5 %) | [log](../../log/selector_audit/2026-08-07_h3-is-a-lexical-artifact.md) |
| 9 (`sec:defense`) | AUC-vs-m frontier, 45–90×, catch/false-refusal operating points | `tofu_sisa_lora/reports/h26/cost_*.{json,md}` ← `analyze_selector_cost.py` |
| 10 (`sec:duration`) | the three-point epochs axis | `tofu_sisa_lora/reports/h21/epochs_axis.{json,md}` |

Full artifact index with regeneration commands and per-cell warnings:
[`../../tofu_sisa_lora/reports/selector_audit/INDEX.md`](../../tofu_sisa_lora/reports/selector_audit/INDEX.md).

## What the draft is missing

1. **Figures.** The draft is tables-only. Three plots would earn their space: AUC vs. prefilter
   size `m` (the monotone-decreasing `indirect` curve), the catch/false-refusal trade-off curve,
   and the epochs axis.
2. **The 300 hand labels** — the only human-blocked item in the campaign.
3. **A privacy column.** The MIA battery returned byte-identical AUCs across arms serving different
   models; §11 reports its absence rather than the number.
4. **Behavioral family under `para_stripped`** — filed, needs the transform wired into
   `router_family_audit`'s query-transform set plus one GPU wave. Would let §9's frontier be stated
   on the honest hard surface rather than on `name_stripped`.
