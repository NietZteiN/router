# paper/workshop — "Deletion Without Absence" (4 pages + appendices)

**Status: DRAFT.** The NeurIPS-workshop-length cut of
[`../followup/main.tex`](../followup/main.tex) — same campaign, same numbers, one argument. The
long draft (14 pages) reports the whole audit; this one puts **only the main argument in the body**
and pushes everything else into appendices, which do not count against a workshop page limit.

| file | what it is |
|---|---|
| `main.tex` | the draft. `\usepackage[dblblindworkshop]{neurips_2026}` |
| `main.pdf` | the built draft: **body ends on page 4**, 13 pages with references and appendices |
| `refs.bib` | copied from `../followup/refs.bib` — **edit both or they drift** |
| `neurips_2026.sty` | the NeurIPS 2026 style file, unmodified upstream |
| `neurips_2026.tex` | the unmodified upstream shell, kept for reference. Not part of the build |
| `checklist.tex` | the NeurIPS checklist, kept for reference. **Not** `\input` — workshop tracks do not require it. If the target workshop does, uncomment the `\input` and fill it in |

## Before submitting

1. **`\workshoptitle{}` is a PLACEHOLDER.** The style file requires it for both workshop options
   and prints an empty footnote rather than erroring, so it is easy to ship broken.
2. **Check the track.** `dblblindworkshop` vs. `sglblindworkshop`; add `final` behind it for
   camera-ready.
3. **Check the page limit.** The body is built for 4 pages. If the workshop says 5, the material
   to promote back is in the appendices, in this order: the name-stripping table (`tab:lexical`,
   App. F) back into §5, then the CSAR decomposition table (`tab:decompose`, App. B) into §4.
4. **The CSAR numbers are still blocked.** See below.

## What is in the body, and what is not

Body: the orphan question (§1), D1/D2/D3 and the three metrics (§2), the blind-forget-metric
construction (§3), what the system actually says (§4), the lexical-artifact result (§5), the
defense frontier (§6). Three tables total, two of them in the body.

Appendices A–K: full setup and threat model, metric definitions, query transforms, the E5
construction with its paired intervals, orphan destinations, the granularity ladder and steering
attack, the defense frontier, the training-duration axis, the measurement constraints, extended
related work, limitations.

## Building

**There is no TeX distribution on this cluster.** Both drafts were built with a
[Tectonic](https://tectonic-typesetting.github.io/) static binary fetched into a scratch
directory, which needs no install and pulls packages on demand:

```bash
curl -sL -o tectonic.tar.gz \
  'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
tar xzf tectonic.tar.gz
./tectonic -X compile main.tex          # runs bibtex and reruns to convergence by itself
```

Current build: **13 pages**, body ending on page 4, 0 undefined citations, 0 undefined references,
0 overfull boxes, four cosmetic underfull ones.

### Checking the 4-page limit after any edit

Nothing enforces it automatically. `main.tex` carries `\label{end-of-body}` at the end of the body
plus a `\typeout`, so one build tells you both whether you fit and by how much:

```bash
grep end-of-body main.aux                    # -> {{7}{4}...}  the 4 is the page. Must be 4.
./tectonic -X compile main.tex --print 2>&1 | grep BODYEND
# -> BODYEND pagetotal=593.64966pt of goal 650.43pt   (~57pt of slack at time of writing)
```

`pagetotal` is how far down page 4 the body ends and `goal` is the full text height, so the
difference is the headroom you have left. Adding roughly 13pt per line of prose, ~57pt is about
four lines.

Structural check on a machine with no TeX (shares the long draft's checker):

```bash
python3 ../followup/tools/check_tex.py main.tex refs.bib
```

## Conventions that are not decoration

- **The long draft's red `\blocked{...}` marker is not used here** — dropped deliberately, so this
  draft has no loud in-text warning. The claim it guarded is unchanged and still binding: **the
  ~300 hand labels validating the CSAR classifier are outstanding, and no CSAR number should go to
  a venue until they exist.** In this draft that survives only as one plain-prose line in the
  limitations appendix, so it is easy to lose track of — check it before submitting.
  [`../followup/main.tex`](../followup/main.tex) still carries the loud version.
- **The title is load-bearing for the repo gate.** `\hypersetup{pdftitle={Deletion Without
  Absence}}` is read back out of the built PDF's bytes by `test_repo_selfcontained.py`, which
  allows exactly one PDF per draft directory and only if its own metadata agrees. It is a
  *different* string from the long draft's title on purpose, so the two PDFs cannot be swapped.
  Change it here and change `DRAFT_PDFS` in `../../test_repo_selfcontained.py` to match; the gate
  fails until you do, deliberately.

## Claim → artifact map

Every number in this draft also appears in the long draft, whose README is the per-claim map from
each number to the file that produced it:
[`../followup/README.md`](../followup/README.md#claim--artifact-map). Full artifact index with
regeneration commands and per-cell warnings:
[`../../tofu_sisa_lora/reports/selector_audit/INDEX.md`](../../tofu_sisa_lora/reports/selector_audit/INDEX.md).

## What this draft is missing

Same four gaps as the long draft — figures, the 300 hand labels, the privacy/MIA column, and the
behavioural family under `para_stripped` — plus one of its own: **no figures at all.** At 4 pages
a single figure would have to displace a table, and the two body tables are the two headline
results.
