# reproduce/ — reverse-engineering the merge-vs-routing master report

Everything needed to take
[`../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md`](../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md)
apart: where every number came from, how to re-derive it, and what will silently mislead you if
you try.

That report argues one thing — **every separable weight-space merge caps at mu ≈ 0.41–0.48, and
only serve-time selection (routing / inference masks) clears it, reaching 0.51–0.82.** It was
assembled by hand from ~3,000 per-run result JSONs. No script produced it, and it printed no
per-cell provenance. This directory supplies that, and makes the whole thing checkable without a
GPU.

## Three tiers, by how much you want to spend

| Tier | Cost | What you get | Command |
|---|---|---|---|
| **1. Verify** | seconds, laptop | Recompute every cell from the snapshotted JSONs and diff against the report | `python reproduce/verify_report.py` |
| **2. Rebuild** | seconds, laptop | Regenerate the report's tables from the JSONs alone and diff the markdown | `python reproduce/rebuild_tables.py` |
| **3. Re-run** | ~GPU-days, cluster | Retrain the pools and recompute the JSONs from scratch | see [PIPELINES.md](PIPELINES.md) |

Tier 1 and 2 need **only the Python standard library** — no torch, no `/storage2`, no HF token, no
cluster. Clone the repo and run them.

```console
$ python reproduce/verify_report.py
verify_report.py -- 214 cells of reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md

  178 verified against the result JSONs
  36 recorded from an out-of-snapshot source (not checkable here)
```

Exit code is non-zero if any cell disagrees. Useful flags: `-v` (show passing cells), `--table F`,
`--run a40-2026-07-24`, `--cross-hardware`.

## What is here

| File | What it is |
|---|---|
| [`cells.tsv`](cells.tsv) | **The core artifact.** One row per (table, row, metric): the value the report prints, the exact JSON field it came from, and a note wherever the report and the data diverge. |
| [`CELL_PROVENANCE.md`](CELL_PROVENANCE.md) | The same map in prose, organised by table, with the producing script and command for each pool. |
| [`verify_report.py`](verify_report.py) | Recomputes each cell and compares to the report at the precision the report printed. |
| [`rebuild_tables.py`](rebuild_tables.py) | Emits the report's tables from the JSONs alone, for a markdown diff. |
| [`snapshot_results.py`](snapshot_results.py) | Builds/audits `results_snapshot/` from a checkpoint store. `--check` re-hashes against the manifest. |
| [`results_snapshot/`](results_snapshot/) | 947 result JSONs (~4.5 MB) across the 38 pools the report cites, plus `MANIFEST.tsv` (sha256 + source path per file). |
| [`CAVEATS.md`](CAVEATS.md) | **Read before trusting any re-derivation.** Twelve traps, each confirmed against the data — including two rows of the report that are spliced across files or tiers. |
| [`PIPELINES.md`](PIPELINES.md) | Per-pool build recipe: train → prepare_eval → eval → collect, with the CPU gate for each. |
| [`METHODS.md`](METHODS.md) | The Table H counterpart of PIPELINES: per-method CPU gate → build → eval chain for every serve-time-selection method, in-tree and `external/`, plus the orphan-battery stages. |
| [`LLAMA2_7B.md`](LLAMA2_7B.md) | **Llama-2-7B only**: mu and orphan behavior per method, in numbers, with the coverage gaps named. Answers "what do we actually know at 7B?" |
| [`CROSS_HARDWARE.md`](CROSS_HARDWARE.md) | The A40 study vs the CISPA A100 rebuild: what replicated and how closely. |
| [`VENDOR_DRIFT.md`](VENDOR_DRIFT.md) | The two `tofu_sisa_lora/` trees have diverged **both ways**, and the repo copy is also the CISPA port — read before syncing anything. |

## How a cell is verified

`cells.tsv` gives each metric its own row, because the report's rows are not always sourced from a
single file. Five cell kinds:

- **direct** — one field of one JSON. Most cells.
- **mean** — arithmetic mean of a field over a glob (Table F′'s own-prob/own-rouge).
- **single** — one probe row standing in for a whole-table cell (Table F′'s solo mu is author 15
  alone; every other probe is NaN — see CAVEATS #4).
- **reband** — a recombination. Tables A, G and I re-present numbers from B–F; those cells resolve
  to the source cell rather than re-reading a file, so a fix propagates.
- **external** — produced outside this repo (Table H's memsinks / sepmlp / sea / memory-adapters
  rows, and the whole CISPA A100 run). Recorded with a citation; reported as `rec`, never as a
  pass. A `reband` pointing at an external cell inherits `rec` rather than counting as MISSING.
- **audit** — a dotted path into a `router_family_audit.py` blob, for the orphan-behavior cells of
  [`LLAMA2_7B.md`](LLAMA2_7B.md), e.g. `strategies.centroid_lm.cells.d199.adequacy.mean`. A final
  `@n_eff` / `@max_share` / `@top3` segment is **derived** from `orphan_capture.top1_hist` rather
  than read — recomputing the concentration in stdlib, independently of the analyzer that produced
  the published table, is the check.

Tolerance comes from the report's own precision: a cell printed as `0.419` is accepted within
±0.0005, because three decimals is all the report committed to. `~4e5` means order-of-magnitude
(±25%).

## Scope of the snapshot

`results_snapshot/` holds the **A40** pools — the original sprint-cluster study, which is what the
master report measures. The CISPA A100 rebuild
([`MERGE_METHODS_7B_K200_2026-07-25.md`](../tofu_sisa_lora/reports/MERGE_METHODS_7B_K200_2026-07-25.md))
recomputed the 7B k=200 numbers on different hardware; its JSONs live under `$TOFU_CKPT_ROOT` on
that cluster and are **not** here. Those cells are recorded in `cells.tsv` with `run =
a100-2026-07-25` and reported as `rec`. To fold them in, run the snapshotter there:

```bash
python reproduce/snapshot_results.py --ckpt-root "$TOFU_CKPT_ROOT"
```

## Verifying the snapshot itself

```bash
python reproduce/snapshot_results.py --check     # re-hash every file against MANIFEST.tsv
```

## If a cell fails

A `FAIL` means the report and the JSONs disagree — not necessarily that the JSON is right. Check
[CAVEATS.md](CAVEATS.md) first: several report rows legitimately mix tiers or directories, and two
are documented splices. If it is not one of those, the ledger entry behind the number is in
[`../log/`](../log/) — `log/merge_mechanism/2026-07-24_merge-ceiling-vs-routing-master.md` is the
entry for this report.
