# VENDOR_DRIFT — the two `tofu_sisa_lora/` trees have diverged, in both directions

Recorded 2026-07-26. **Nothing here has been reconciled** — this is a map so the sync can be done
deliberately, because the naive `rsync` in either direction destroys real work.

## The two trees

| | Path | What it is |
|---|---|---|
| **home** | `/home/jack/tofu_sisa_lora/` | The working tree on the original **sprint** cluster (A40s). Where the reports are authored. |
| **repo** | `merge-tables-7b/tofu_sisa_lora/` | The vendored copy in this repo — **and also the CISPA A100 port**. |

There is no sync script; vendoring has been manual (`git log`: *"Vendor the master report + the 7
files missing since the last sync"*).

## ⚠ The repo copy is not a stale snapshot — it is a fork

This is the part that makes a blind sync dangerous. The repo copy was **adapted to run on the CISPA
A100 cluster**, where the sprint node names, the `all` partition and the 4-GPU policy do not exist:

- `cluster_env.sh` (repo-only) is the single source of truth there for interpreter, partition,
  account, checkpoint root and GPU cap. At least `submit_eval.sh`, `submit_ctv.sh` and
  `submit_pool_7b.sh` source it.
- `slurm_nodes.sh` **differs by design**: home pins `sprint1/2/3`, `--exclude=sprint4`,
  `TOFU_ARRAY_CAP=4`; the repo version is a compatibility shim that defers to `cluster_env.sh` and
  says so in its header.
- Repo-only operational scripts: `badnode_reaper.sh`, `campaign_orchestrator.sh`, `anchor_gate.sh`,
  `overnight_finalizer.sh`, `run_pair.sh`, `stage_hf_cache.sh`, `finalize_report.py`,
  `requirements.txt`.

**Copying home → repo wholesale would overwrite the CISPA port and break every driver there.**

## Current divergence (2026-07-26)

52 files differ · 16 exist only in the repo · 220 exist only in home. `log/` is **identical** in
both. Nothing under `reproduce/` exists in home at all.

### Newer in **home** (would be lost by a repo → home sync)

| File | What home has that the repo does not |
|---|---|
| `reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md` | **Table C′** (the 7B k=200 operator battery), **Appendix C glossary**, the 2026-07-26 coverage update, the PEFT-linear rank caveat — and, as of today, the **orphan-column legend + Table H′**. |
| `reports/POST_DELETION_ROUTING_FULL_REPORT_2026-07-24.md` | 565 lines vs 364. |
| `submit_router_family.sh` | stages `j7`–`j11` (the 7B + de-confound orphan batteries). |
| `submit_7b_routed_fill.sh` | home-only; new. |

### Newer in the **repo** (would be lost by a home → repo sync)

| File | What the repo has that home does not |
|---|---|
| `reports/MERGE_METHODS_7B_K200_2026-07-25.md` | The CISPA A100 rebuild of the 7B k=200 merge battery. |
| `reports/MERGE_VS_ROUTING_MASTER_2026-07-24_ADDENDUM_P3.md` | The P3 addendum folding that run into the master report's frame. |
| `reports/nmerge_mu.csv` | 104 rows vs home's 70 (home's is from 2026-07-16). |
| `reports/interference/` | Directory absent from home. |
| the cluster port | everything in the section above. |

## Consequence for `reproduce/`

`verify_report.py` parses the **repo** copy of the master report. So:

> **Until someone syncs the report, the Table H′ / `H7B` cells in [`cells.tsv`](cells.tsv) describe a
> section that does not exist in the vendored `MERGE_VS_ROUTING_MASTER_2026-07-24.md`.**

That is harmless to the checker — cells resolve against `results_snapshot/`, not against the report
prose, so `verify_report.py` passes either way (267 cells, 222 verified, 0 FAIL, 0 MISSING). But a
reader diffing the report against `rebuild_tables.py` output will not find Table H′ in the repo copy
until the sync happens.

## Suggested sync, if and when you want one

Not performed here. The safe shape is **file-by-file, one direction each**:

1. **home → repo, reports only** — the master report, `POST_DELETION_ROUTING_FULL_REPORT`, and the
   two new drivers. Do **not** sync `slurm_nodes.sh` or any `submit_*.sh` that sources
   `cluster_env.sh`; `submit_router_family.sh` and `submit_7b_routed_fill.sh` are sprint-flavored
   (they hardcode `--partition=all` and `--exclude=sprint4`) and need the `cluster_env.sh` treatment
   before they run on CISPA.
2. **repo → home** — `MERGE_METHODS_7B_K200_2026-07-25.md`, the P3 addendum, `nmerge_mu.csv`,
   `reports/interference/`.
3. Then add a `sync_vendor.sh` with an explicit allow-list, so the next drift is visible rather than
   discovered.

The 220 home-only files are mostly `paper/`, `.pytest_cache/` and per-thread scratch; they are not
all intended for the repo, which is why an allow-list beats `rsync -a`.
