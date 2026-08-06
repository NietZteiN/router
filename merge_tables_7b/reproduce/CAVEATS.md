# CAVEATS — what will silently mislead you

Fourteen traps, every one confirmed against the data in `results_snapshot/` (not inferred from
prose). Numbers 1, 4, 5, 6 and 12 are cases where the master report itself is imprecise; 13 is two
bugs in this directory's own tooling; the rest are properties of the pipeline that make an honest
re-derivation come out wrong.

---

## 1. Two different "base" anchors, both labelled Llama-3.2-1B

The anchor legend gives **P1 base = 0.398** and **P4 base = 0.380**. Both pools are
`meta-llama/Llama-3.2-1B-Instruct`. They are different evals:

| Printed as | Actual | File |
|---|---|---|
| P1 / P5 base 0.398 | 0.3984 | `Llama-3.2-1B-Instruct_ctv_ctrl_r32_e25/results/smoke/base_model__own15.json` — the ctv **single-author probe** (author 15) |
| P4 base 0.380 | 0.3796 | `Llama-3.2-1B-Instruct/results/smoke/base_model.json` — the P1 pool's **population** base |

Consequence: **Table A carries the identical row twice with different Δbase.** "Uniform mean
λ=1/k" (P1) reads mu 0.419, Δbase **+0.021**; "LoRA (additive mean)" (P4) reads the same mu 0.419
from the same file, Δbase **+0.039**. Both are arithmetically correct against their own anchor.
If you recompute Δbase from a pool's own `base_model.json`, you will reproduce the P4 column and
disagree with the P1 column by 0.019 everywhere.

## 2. Band 2 is a property of the pool, not of the operator

Table B/A file `PEFT linear` (0.050) and `TSV-M` (0.051) under "Broke downward". At 7B k=200 r8 the
*same* operators are the **top of the band** — linear **0.4498**, TSV-M **0.4344**
([`MERGE_METHODS_7B_K200_2026-07-25.md`](../tofu_sisa_lora/reports/MERGE_METHODS_7B_K200_2026-07-25.md) §2).

The cause is the scale convention, not the algorithm. Shards train with **rsLoRA** (scaling α/√r),
and PEFT's stock factor-space merges apply `√(w·scaling)`, double-counting the √r ≈ 2.83. At k=10
the per-adapter weight is 1/10 and the inflated delta blows the model up; at k=200 the 1/200 weight
absorbs it. **Compare within a convention.** Do not read "linear is degenerate" as a general fact.

## 3. Mis-filed r8 gapfill JSONs

Ten of the k=200 r8 results sit in `Llama-2-7B-chat-hf_k200_r8_e5_lr1e4/results/*.json` —
**not** `results/smoke/`. An `EVAL_MANIFEST=` override passed `--out` one level up. A
`results/{smoke,extended}/` glob silently misses them: `merged_{fisher,regmean,ties,della_ties,tsv,
lorahub,breadcrumbs_s0.00177,breadcrumbs_s0.005,subtract_orth}.json` and `tree_root_slerp.json`.
`snapshot_results.py` walks the whole `results/` tree for exactly this reason.

## 4. Table F′ solo mu is a single author, and the rest are NaN

The "solo mu" column comes from **author 15 only**. The other four probes (76, 82, 111, 177) have
`model_utility = NaN`: `--retain_author_ids` restricts the retain split to one author, and TOFU's
`*_perturbed` splits cover only ~2 rows per author, so the retain truth-ratio subset comes out
empty and the harmonic mean is undefined. Nothing in the report says this.

The **own-prob / own-rouge** columns of the same table are means over *all* probe rows (20 rows;
32 for `[wd]`), which do not go NaN. So one row of Table F′ mixes an n=1 statistic with two n=20
statistics.

`[wd]` additionally averages its **two variants** — `orthblock` (solo mu 0.4563) and `rowslice`
(0.4638) → the printed 0.460. Its own-rouge 0.404 is likewise the both-variant mean; orthblock
alone is 0.3995.

## 5. Table F's ClAMU K=16 row is spliced across two directories

`_clamu/` and `_clamu_K16/` are **both** K=16 — the default config's `num_clusters` is 16 — but
they differ in mask-training budget (`mask_epochs`). The report's row takes two cells from one and
one from the other:

| Cell | Printed | Comes from | That file's other values |
|---|---|---|---|
| mu | 0.647 | `_clamu/clamu_full.json` (0.6469) | fq 0.3929, f_rouge 0.5254 |
| fq | 0.239 | `_clamu_K16/clamu_full.json` (0.2391) | mu 0.6620 |
| f_rouge | 0.593 | `_clamu_K16/clamu_full.json` (0.5934) | — |

Read as one model, the row does not exist on disk. The K=200 row by contrast is internally
consistent (all three cells from `_clamu_K200/clamu_full.json`).

## 6. Table F's `merge_full` row mixes tiers

mu 0.407 and f_rouge 0.383 are the **smoke** tier; fq 0.099 is the **extended** tier. The smoke fq
is 0.594, and the extended mu/f_rouge are 0.4051/0.3638. Neither tier alone produces the printed
row.

## 7. `fq` is low-information — do not use it to identify a file

`forget_quality` is a KS p-value over a small discrete sample, so the same handful of values recur
across completely unrelated runs. `0.2391` appears in at least 17 files in the snapshot, spanning
five pools and both model scales. Matching a row by its fq will find the wrong file. Match on
`model_utility` + `forget_rouge`.

## 8. `routed_*` JSONs carry no self-identifying metadata

`routed_scaffold_ood.json`, `routed_oracle_full.json` and friends are written by
`eval_routed_scaffold.py`, not `eval_tofu.py`. Every one of them stores the internal `label` field
as `routed_scaffold_ood` regardless of filename, and leaves `model_name`, `adapter` and `k` unset.
**The filename and the pool directory are the only provenance.** Do not dedupe or group these by
their `label` field.

## 9. Table H rows are not all the same metric or tier

- `sepmlp ≈0.795` is a **recall** number, not `model_utility`. `memory_adapters 0.869` is tagged
  `[OU]`, `s3t 0.581` is tagged `[F(d)]`. None is comparable to the mu column.
- The two legonet rows are the **post-deletion** label (`legonet_unlearn`), not `legonet_full`
  (which reads 0.6277 / 0.4947) — and they are at **different tiers**: 7B smoke, 1B extended.
- `routing_scaffold k=10` (0.7509) is extended tier while most neighbours are smoke.

## 10. `reports/nmerge_mu.csv` is a half-table

`analyze_nmerge.py` writes to `--out_prefix`, default `reports/nmerge`. Running it for the
interference config and then the centered config overwrites the first with the second. Current
state:

| Copy | Holds | Missing |
|---|---|---|
| `reports/nmerge_mu.csv` (committed, from the A100 rebuild) | 44 `cpool_svd1024` + 39 `cr16` + 15 `cr16_svd1024` | the entire additive ladder, **and** the `base_model`/`ft_r32` anchors |
| `reports/interference/nmerge_mu.csv` (added here) | 49 `nmerge_add` + 11 `nmerge_add_svd1024` + anchors | the centered ladder |
| `reports/centered/`, `reports/e25/` | the A40 centered and e25 ladders | — |

Always pass a distinct `--out_prefix` per config.

## 11. Operational traps

- **0-byte adapter placeholders.** The vendored k=200 pools arrived on the new cluster as 0-byte
  files, which satisfy every driver's skip-if-exists test. Check file **sizes**, not existence,
  before trusting a resumed pool.
- **KnOTS-TIES does not complete at k=200** — the shared-basis construction over 200 adapters hangs
  (>50 min, 0% GPU). Excluded upstream, not missing by accident.
- **High-k eval memory law.** PEFT casts adapters to fp32 on load, so eval memory ≈ 13.5 GiB +
  k·n_params(rank)·4B. k=200 × r32 (~65 GiB) does not fit one 40–46 GiB card: use two GPUs with
  `device_map="auto"`, or `--preloaded_adapter`, or `--lazy_adapter_cache`.
- **`reports/all_metrics_smoke.csv` is stale** — a pre-metrics-fix snapshot (`merged_dare_ties`
  reads 0.17 there vs the live 0.424). The report says so itself. Regenerate with
  `collect_results.py --root ./checkpoints --smoke`.
- **`STUB=1` previews any `submit_*.sh`** without submitting. Use it before every SLURM job.
- **`submit_peft_bakeoff.sh` has diverged.** The repo copy is the CISPA A100 port; the
  sprint-cluster tree has its own newer edits. The two were deliberately not merged — the repo
  keeps the CISPA version, since reverting it would break the cluster the results were produced on.
  Full drift map: [VENDOR_DRIFT.md](VENDOR_DRIFT.md).

## 12. A Table H row's orphan numbers may come from a different model than its mu

The "Model used" column names the model behind the **utility**. The orphan cell in the same row is
not guaranteed to match it, and two rows definitely do not:

- **legonet** — mu **0.6371 is 7B**, but the two-magnet `n_eff 6.1` (e5/e11/e30) figure quoted
  everywhere is the **1B** n=32 pool. No 7B legonet routing audit exists.
- **sea** — mu **0.711 is 4-bit 7B**, but its routing is borrowed from the **1B** SIFT per-author
  centroid measurement (n_eff 27.1, magnet author 88), and the Group-A verdict is argued from
  mechanism rather than an end-to-end serve.

Two further orphan cells are **predicted, not measured**: **memsinks** (routing measured, but the
H-GB3 serve run was never executed at any scale) and **sea** (as above).

Related trap when comparing orphan batteries: **pool provenance**. The 1B k=10 battery
(`_experts_scaf_k10`) is the *scaffolded* pool while the 7B k=10 battery is the *plain* pool, so a
naive 1B-vs-7B read conflates model scale with scaffolding for every router that reads the LLM. The
routers that do not read it come out bit-identical, which is the tell. See
[LLAMA2_7B.md §3.3](LLAMA2_7B.md#33-is-the-magnet-a-property-of-the-model-or-of-the-embedding-space).

## 13. Two analyzer traps, both hit and confirmed on 2026-07-26

**(a) `snapshot_results.py` with no `--ckpt-root`, run from inside this repo, silently produces a
0-file snapshot and overwrites `MANIFEST.tsv` with an empty one.** Its default resolution order
prefers `tofu_sisa_lora/checkpoints`, which in a fresh clone is an absent/empty gitignored symlink.
It only ever *copies*, so the 955 JSONs survive — but the manifest is destroyed, and `--check` then
trivially "passes" on nothing. Always pass the store explicitly:

```bash
python reproduce/snapshot_results.py --ckpt-root /home/jack/tofu_sisa_lora/checkpoints
python reproduce/snapshot_results.py --check      # must report 955/955
```

A run that prints `[snapshot] 0 files` plus a 38-pool WARNING is this failure, not an empty store.

**(b) `analyze_router_family.py` keys rows by `<strategy>@k<k>` and cannot distinguish two pools at
the same `k`.** Feeding it the 1B-scaffolded, 1B-plain and 7B-plain k=10 batteries at once renders
all three as `(k=10)` in `rl_family_leak_table.md` — indistinguishable — and its H-ARCH verdict
deduplicates them by file mtime, keeping only the newest. It does emit `WARN: duplicate
<strategy>@k10 entries`; heed it. **Feed this analyzer one pool per `k`.** The pool-keyed reduction
is `analyze_orphan_destinations.py`, which labels every row with its pool directory and handles the
multi-pool case correctly.

## 14. `n_eff` is not comparable across pool granularities

Dropping "one shard" removes 20 authors at k=10 but only 1 at k=200, so the orphan count falls
400 → 80 → 20 while survivors rise 9 → 49 → 199 — and `n_eff` is bounded by both. The non-monotone
7B dial (`centroid_lm` 2.5 → 7.3 → 1.9) is mostly that bound moving. Compare **adequacy** instead,
which is a similarity ratio and therefore scale-free.
