# merge-tables-7b

Self-contained repo for the **Llama-2-7B, one-author-per-shard (k=200) model-merging results** build
(TOFU exact-unlearning project). It carries the **plan, results, and the full runnable codebase** —
everything needed to run the plan *as if on the cluster*, minus the model weights (those live on
`/storage2` and are far too large for git).

## Contents
- **[PLAN.md](PLAN.md)** — the execution plan (budget-bounded to ≈1 GPU-day): phases M/P/C, commands, estimate.
- **[STATUS.md](STATUS.md)** — coverage inventory (computed vs TODO on the 7B k=200 pool), runtime evidence,
  the memory-wall constraint, and the priority-ordered next actions.
- **[RESULTS_TABLES.md](RESULTS_TABLES.md)** — consolidated results tables (merge operators, dilution,
  N-ladder, PEFT bake-off, full-FT SIFT/ClAMU, ctv arms) with base/finetuned anchors per row.
- **[SETUP.md](SETUP.md)** — how to run after a `git pull`: env, the `checkpoints` symlink, HF token, config paths.
- **[tofu_sisa_lora/](tofu_sisa_lora/)** — the full source tree: all `*.py` (trainers, eval, merge, analysis,
  test gates), `submit_*.sh` SLURM drivers, `configs/*.json`, `reports/` (writeups + result CSV/JSON the
  analysis reads), `requirements.txt`, and `CLAUDE.md` (repo map).
- **[reproduce/](reproduce/)** — reverse-engineer the merge-vs-routing master report: per-cell provenance,
  a snapshot of the result JSONs, and `verify_report.py` (no GPU, no cluster — see below).
- **[log/](log/)** — the dated research ledger, 155 entries across 19 threads: hypothesis before the run,
  exact command, SLURM job ID, verdict after.
- **[external/](external/)** — the six out-of-tree projects Table H of the master report cites
  (`memsinks_tofu` is a hard import dependency of `eval_tofu.py --memsinks_config`).

## Reverse-engineering the master report
[`tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md`](tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md)
is the thesis report — every separable weight-space merge caps at mu ≈ 0.41–0.48; only serve-time
selection clears it. It was assembled by hand from ~3,000 result JSONs. [`reproduce/`](reproduce/) makes
every cell checkable **with the Python standard library alone** — no torch, no GPU, no `/storage2`, no
HF token:

```bash
python reproduce/verify_report.py       # recompute each cell from the snapshot, diff vs the report
python reproduce/rebuild_tables.py      # regenerate the report's tables from the JSONs alone
```

Start at [`reproduce/README.md`](reproduce/README.md); read [`reproduce/CAVEATS.md`](reproduce/CAVEATS.md)
before trusting any re-derivation. The 7B k=200 cells the report left open are filled in by
[the P3 addendum](tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24_ADDENDUM_P3.md).

**Two entry points added 2026-07-26:**
- [`reproduce/LLAMA2_7B.md`](reproduce/LLAMA2_7B.md) — the **Llama-2-7B** view: mu *and* orphan
  behavior per method, in specific numbers, with the coverage gaps named rather than implied. Only
  3 of Table H's 12 methods have a 7B `model_utility` verifiable from the snapshot; this says which,
  and what blocks each of the rest.
- [`reproduce/METHODS.md`](reproduce/METHODS.md) — reproduce **every** Table H method: CPU gate →
  build → eval chain, in-tree and `external/`, plus the orphan-battery stages.

⚠ Before syncing this repo's `tofu_sisa_lora/` against the sprint-cluster tree, read
[`reproduce/VENDOR_DRIFT.md`](reproduce/VENDOR_DRIFT.md) — they have diverged in **both** directions
and the copy here is also the CISPA A100 port.

## What's NOT in git
Model weights, checkpoints, and datasets — the `tofu_sisa_lora/checkpoints` symlink to
`/storage2/jack/checkpoints/tofu_sisa_lora` is excluded (recreate it per `SETUP.md`). Python caches are
excluded too. Everything else — code, configs, and result data — is here, so the plan runs without hunting
for missing files.

## Quickstart
```bash
git clone git@github.com:NietZteiN/merge-tables-7b.git && cd merge-tables-7b
# 1) env + symlink + HF token — see SETUP.md
pip install -r tofu_sisa_lora/requirements.txt
cd tofu_sisa_lora && ln -s /storage2/jack/checkpoints/tofu_sisa_lora checkpoints
# 2) run the plan — priority order + commands in STATUS.md, e.g.:
python test_merge_extra.py && bash submit_ctv.sh configs/sparsify_7b.json
```
Running actual jobs requires the cluster (GPUs, `/storage2`, SLURM) and the global 4-GPU cap (see `STATUS.md`).

## Update workflow
```bash
git pull --rebase          # before you start
git add -A && git commit -m "…" && git push
```
