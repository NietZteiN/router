# CELL_PROVENANCE — every table, and where its numbers come from

Prose companion to [`cells.tsv`](cells.tsv). For each table of
[`MERGE_VS_ROUTING_MASTER_2026-07-24.md`](../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md):
the pool, the script that produced it, the eval label, and the JSON the number is read from.

Common to every eval: interpreter `/home/jack/anaconda3/envs/test-env/bin/python` (or
`$TOFU_PYTHON` on CISPA), cwd `tofu_sisa_lora/`, `HF_HOME` set, **seed 42**, `--smoke` tier unless
noted, metrics version `ou-2026-06-10`. `$CK` = the checkpoint store (`tofu_sisa_lora/checkpoints`
symlink, or `$TOFU_CKPT_ROOT`). Paths below are relative to `$CK` and to
[`results_snapshot/`](results_snapshot/) alike — the snapshot preserves the store's layout.

Metric fields in the JSONs: `model_utility` (mu), `forget_quality` (fq, alias `ks_pval`),
`forget_rouge` (f_rouge), `retain_ppl` (r_ppl).

---

## Table A — grand master

**Not independently produced.** Table A re-bands the rows of B (P1), C/P2, D (P3), E (P4) and F
(P5) into "in-band / broke downward / partial win". Every row resolves to one of those tables;
`cells.tsv` encodes them as `reband` so a correction propagates rather than being duplicated.

Its `Δbase` / `ΔFT` columns are arithmetic against the anchor legend — and the P1 anchor is not the
P1 pool's own base eval. See [CAVEATS #1](CAVEATS.md#1-two-different-base-anchors-both-labelled-llama-32-1b).

The P3 column was deliberately left almost empty ("partly queued"); it is filled in by
[the P3 addendum](../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24_ADDENDUM_P3.md).

## Table B — P1, Llama-3.2-1B, k=10 LoRA shards

**Pool** `Llama-3.2-1B-Instruct/` · **anchors** `base_model.json` 0.3796, `ft_all.json` 0.5302.

Train (no config file — CLI only, flag-free defaults *are* the frozen recipe r32/α64/e5/lr1e-4):

```bash
python train_lora_shard.py --shard_id $i --k 10 \
    --model_name meta-llama/Llama-3.2-1B-Instruct --output_dir ./checkpoints
# driver: bash submit_overnight.sh 10 meta-llama/Llama-3.2-1B-Instruct
```

The forget-quality KS reference needs the retain90 oracle first, at the **legacy** recipe:

```bash
python train_lora_shard.py --retain90 --k 10 --rank 8 --alpha 16 --epochs 3 --lr 2e-4 \
    --model_name meta-llama/Llama-3.2-1B-Instruct --output_dir ./checkpoints
python prepare_eval.py --smoke --k 10 \
    --output_dir ./checkpoints/Llama-3.2-1B-Instruct \
    --model_name meta-llama/Llama-3.2-1B-Instruct
```

Eval — one label per row, all through `activate_label` in `merge_lora.py`:

```bash
python eval_tofu.py --model_name meta-llama/Llama-3.2-1B-Instruct \
    --output_dir ./checkpoints/Llama-3.2-1B-Instruct \
    --label <LABEL> --k 10 --forget_shard_id 9 --smoke \
    --out ./checkpoints/Llama-3.2-1B-Instruct/results/smoke/<LABEL>.json
# driver: bash submit_eval.sh ./checkpoints/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B-Instruct 10
```

| Report row | Label | JSON |
|---|---|---|
| Naive sum λ=1 | `merged_additive_s1` | `results/smoke/merged_additive_s1.json` |
| Uniform mean λ=1/k | `merged_additive_mean` | `merged_additive_mean.json` |
| Tuned-λ sum λ=0.05 | `merged_additive_s0.05` | `merged_additive_s0.05.json` |
| DARE-TIES | `merged_dare_ties` | `merged_dare_ties.json` |
| DELLA-TIES | `merged_della_ties` | `merged_della_ties.json` |
| Fisher-weighted | `merged_fisher` | `merged_fisher.json` |
| KnOTS | `merged_knots_ties` | `merged_knots_ties.json` |
| Breadcrumbs λ=1/(n√r) | `merged_breadcrumbs_s0.0354` | `merged_breadcrumbs_s0.0354.json` |
| Breadcrumbs λ=1/n | `merged_breadcrumbs_s0.1` | `merged_breadcrumbs_s0.1.json` |
| PEFT linear | `merged_linear` | `merged_linear.json` |
| TSV-M | `merged_tsv` | `merged_tsv.json` |
| SLERP | **`tree_root_slerp`** | `tree_root_slerp.json` |
| Subtract-orth | `subtract_orth` | `subtract_orth.json` |
| Task-arith subtraction | `subtract_linear` | `subtract_linear.json` |
| *Routing key-exact* | `routed_key_exact` | `routed_key_exact.json` |
| *Routing + scaffold, OOD-aware* | — | `routed_scaffold_ood.json` |

Two rows are not plain `eval_tofu` merges:

- **SLERP** is pairwise-only, so the label is a **tree** merge (`tree_root_slerp`), not
  `merged_slerp` — there is no k=10 SLERP merge.
- **Routing + scaffold** is produced by a different script entirely:

  ```bash
  python train_scaffold.py                 # public-Alpaca scaffold LoRA
  python make_scaffolded_base.py           # bake it into theta_0
  python eval_routed_scaffold.py --shards_dir $CK/Llama-3.2-1B-Instruct_experts_scaf_k10 ...
  ```

  It writes into the P1 pool's results dir but sets no `model_name`/`adapter`/`k`
  ([CAVEATS #8](CAVEATS.md#8-routed_-jsons-carry-no-self-identifying-metadata)).

`merged_fisher` is a **data-required** merge — `eval_tofu.py` builds the dataloader itself
(`build_merge_dataloader`); `merge_shards` raises without it. `merged_lorahub` additionally needs
`nevergrad`.

## Table C — the dilution law (Llama-2-7B, one pool per k)

One pool per shard count, same `dare_ties` operator swept over k.

| k | Pool | JSON |
|---|---|---|
| 4 | `Llama-2-7B-chat-hf_k4_r32_e5_lr1e4/` | `results/smoke/merged_dare_ties.json` |
| 10 / 20 | `..._k{10,20}_r32_e5_lr1e4/` | same filename |
| 50 / 100 | `..._k{50,100}_r32_e5_lr1e4/` | same filename |
| 200 | `..._k200_r8_e5_lr1e4/` | same filename — **r8**, because r32 × 200 exceeds a 46 GiB A40 |

Drivers: k ∈ {4,10,20} via `submit_llama2_grid_overnight.sh` / `submit_shard_grid.sh`; k ∈
{50,100,200} via `bash submit_scale_grid.sh`, which chains train+prep → `gate_scale_load.py` (the
memory go/no-go gate) → eval arrays → collect, and auto-submits the r8 fallback when the r32 gate
fails. The k=4 pool also trains a `retain75` oracle (`--retain_authors 150`) so its fq is valid —
forget shard 3 at k=4 is authors 150–199.

The routing row is `routed_key_exact.json` in the same pools; k=4/10/20 were never routed, hence
the report's em-dashes.

## Table D — per-author N-merge ladder (P3)

**Pools** `Llama-2-7B-chat-hf_nmerge_r32/` (additive) and `..._nmerge_r32_centered/` (centered),
both merging subsets of the 200 single-author LoRAs in `..._k200_r32_e5_lr1e4/`.
**Configs** `configs/nmerge_interference_7b.json`, `configs/nmerge_centered_7b.json`.

```bash
bash submit_nmerge.sh configs/nmerge_interference_7b.json plan    # subset plan
bash submit_nmerge.sh configs/nmerge_interference_7b.json merge   # CPU array: materialize adapters
bash submit_nmerge.sh configs/nmerge_interference_7b.json eval    # GPU array
python analyze_nmerge.py --config configs/nmerge_interference_7b.json \
       --out_prefix reports/interference/nmerge                   # NOTE the distinct prefix
```

Merges are materialized on **CPU** by `merge_subset.py` into `<pool>/merges/<label>/`, then served
with `eval_tofu.py --preloaded_adapter <dir> --k 200 --forget_shard_id 199 --eval_shard_id 82`.
That is what dodges the high-k memory law — one merged adapter resident, not 200.

**Headline probe = author 82** = `numpy.random.RandomState(42).permutation(199)[0]`. Result files:
`results/smoke/nmerge_{add,cr16}[_svd1024]_N{n}_s42__own82.json`. The `_svd1024` infix appears for
N ≥ 128 (additive) and N ≥ 64 (centered) — above the exact cap the merge is SVD-compressed to rank
1024, and the filename records it.

⚠ `--out_prefix` is mandatory here: see
[CAVEATS #10](CAVEATS.md#10-reportsnmerge_mucsv-is-a-half-table).

## Table E — PEFT parameterization bake-off (P4)

**Pools** `Llama-3.2-1B-Instruct_peft_{dora,ia3,vera,prefix}_k10/` · **config**
`configs/peft_bakeoff_1b.json` (per-method *standard* learning rates, deliberately not
recipe-matched).

```bash
python test_compose_peft.py                                   # CPU gate; also gates the dora arm
bash submit_peft_bakeoff.sh configs/peft_bakeoff_1b.json all  # smoke -> train -> compose -> eval
```

Each parameterization is served differently — this is the part that is easy to get wrong:

| Method | How the composed model is served | JSON |
|---|---|---|
| LoRA | the P1 pool's `merged_additive_mean` (no `_peft_lora_` pool exists) | `Llama-3.2-1B-Instruct/results/smoke/merged_additive_mean.json` |
| DoRA | standard merge label `merged_additive_mean` | `_peft_dora_k10/results/smoke/merged_additive_mean.json` |
| IA³ | `compose_peft.py` → `--preloaded_adapter` | `ia3_composed_full.json`, `ia3_geo_full.json` |
| VeRA | `compose_peft.py` (shared frozen basis) → `--preloaded_adapter` | `vera_composed_full.json` |
| Prefix | `prefix_concat.py` via `--prefix_pool_dir` | `prefixcat_full.json` |

All evaluated `--k 10 --forget_shard_id 9`. Routed rows are `routed_key_exact.json` in each pool;
isolated probes are `{method}_iso_s9.json`.

## Table F — full-parameter task vectors (P5)

**SIFT-Masks** — pool `Llama-3.2-1B-Instruct_sift_masks/`, config `configs/sift_masks_tofu_1b.json`:

```bash
python test_sift_masks.py                                          # CPU exactness gate
bash submit_sift_masks_tofu.sh configs/sift_masks_tofu_1b.json all # build -> unlearn -> eval
python eval_tofu.py --sift_masks_config configs/sift_masks_tofu_1b.json \
    --label sift_full --k 10 --forget_shard_id 9 --smoke --out .../sift_full.json
```

Labels: `sift_full` (masked), `merge_full` (the same sum with **no** mask), `sift_unlearn`,
`merge_unlearn`.

**ClAMU** — pools `_clamu/` and `_clamu_K{1,4,16,50,100,200}/`, configs `configs/clamu_tofu_1b.json`
(+ `_K{K}.json`):

```bash
python test_clamu.py
bash submit_clamu_tofu.sh configs/clamu_tofu_1b.json all   # setup -> build -> localize -> unlearn -> eval
```

Labels: `merge_full` (Global, no mask), `emr_full`, `tall_full`, `clamu_full` (optimized mask).

⚠ Two rows of this table are not single-file rows:
[CAVEATS #5](CAVEATS.md#5-table-fs-clamu-k16-row-is-spliced-across-two-directories) (ClAMU K=16
splices `_clamu/` and `_clamu_K16/`) and
[CAVEATS #6](CAVEATS.md#6-table-fs-merge_full-row-mixes-tiers) (`merge_full` mixes smoke and
extended).

## Table F′ — ctv training-time constructions

**Pools** `Llama-3.2-1B-Instruct_ctv_{ctrl,lin,wd}_r32_e25/`, `..._ctv_ds_e25/` · **configs**
`configs/ctv_1b_{ctrl,wd,lin,ds}.json` · driver `bash submit_ctv.sh CONFIG [gate|prep|train|verify|merge|eval|collect]`.

| Arm | Trainer | Served by |
|---|---|---|
| ctrl, wd | `train_struct_tv.py --arm {control,orthblock,rowslice}` | `--preloaded_adapter` |
| lin | `train_linear_tv.py` | `eval_tofu.py --linear_tv_config` |
| ds | `train_ds_support.py` | `eval_tofu.py --ds_config` |

Every row is an isolated single-author probe:
`eval_tofu.py … --eval_shard_id <a> --retain_author_ids <a>` → `results/smoke/iso_a{a}[_{variant}]__own{a}.json`.

**Read the aggregation rule before re-deriving**: solo mu is author 15 alone (the others are NaN),
own-prob/own-rouge are means over all probe rows, and `[wd]` averages two variants —
[CAVEATS #4](CAVEATS.md#4-table-f-solo-mu-is-a-single-author-and-the-rest-are-nan). `analyze_ctv.py`
exists but left no CSV on disk, so this table was assembled directly from the JSONs;
`rebuild_tables.py --table "F'"` reproduces it.

## Tables G and I — re-bands

Both recombine numbers already established elsewhere; `cells.tsv` encodes every cell as a `reband`
pointing at its source. Table I is the cleanest control in the report — same weights, selector
off vs on:

| Underlying weights | no-select | + select | Source cells |
|---|---|---|---|
| SIFT spine | `merge_full` 0.407 | `sift_full` 0.737 | F |
| ClAMU spine | Global 0.351 | opt-mask 0.647 / 0.672 | F |
| LoRA 1B k=10 | `additive_mean` 0.419 | routed+scaffold 0.556 | B |
| LoRA 7B k=200 | `additive_mean` 0.460 | oracle route 0.824 | D, and `_k200_r32_e25_lr1e4/results/smoke/routed_oracle_full.json` |

## Table H — routing master

Lifted from [`ROUTING_MASTER_2026-07-23.md`](../tofu_sisa_lora/reports/ROUTING_MASTER_2026-07-23.md)
Table 1, and it spans several projects. In-tree rows verify from the snapshot; the rest are
`external`.

| Row | Home | Source |
|---|---|---|
| routing_scaffold k=200 e25 (**0.8236**, repo best) | in-tree | `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4/results/smoke/routed_oracle_full.json` |
| routing_scaffold k=10 (0.7509) | in-tree | `Llama-3.2-1B-Instruct_experts_scaf_k10/results/**extended**/routed_scaffold_strong.json` |
| legonet 7B / 1B | in-tree legonet arm | `*_legonet_n32_k3/results/{smoke,extended}/legonet_**unlearn**.json` |
| sisa_lora routed | in-tree | `..._k50_r32_e5_lr1e4/results/smoke/routed_key_exact.json` |
| clamu, sift_masks, peft_compose | in-tree | Tables F and E |
| s3t | in-tree | `Llama-2-7B-chat-hf_s3t_m5_L4_armB/F_curve.json` — an `[F(d)]` curve, not mu |
| **memsinks** | [`../external/memsinks_tofu/`](../external/memsinks_tofu/) | its own store; `eval_tofu.py --memsinks_config` imports `memsinks_routed_model.py` from there |
| **sepmlp** | [`../external/sepmlp_tofu/`](../external/sepmlp_tofu/) | `reports/MUSR_EVIDENCE_FULL_REPORT_2026-07-22.md` — a **recall** number |
| **sea** | [`../external/sea_tofu/`](../external/sea_tofu/) | `reports/SEA_UNLEARNING_REPORT.md` |
| **memory_adapters** | [`../external/memadapt_tofu/`](../external/memadapt_tofu/) | `REPORT_2026-07-15.md` — tagged `[OU]` |

⚠ The rows are not a like-for-like column:
[CAVEATS #9](CAVEATS.md#9-table-h-rows-are-not-all-the-same-metric-or-tier). And a row's *orphan*
numbers are not always from the same model as its *mu*:
[CAVEATS #12](CAVEATS.md#12-a-table-h-rows-orphan-numbers-may-come-from-a-different-model-than-its-mu).

## Table H′ — the Llama-2-7B slice (`H7B`)

The 7B-only view: mu per method **and** orphan behavior in numbers. Prose version with the readings:
[`LLAMA2_7B.md`](LLAMA2_7B.md). Cells: [`cells.tsv`](cells.tsv) rows tagged `H7B`.

| Group | Kind | Source |
|---|---|---|
| mu — routing_scaffold k=200, sisa_lora k=50/100/200, legonet | `reband` | Re-present Tables C and H, so a fix upstream propagates rather than being re-typed |
| mu — legonet pre-deletion (0.6277) | `direct` | `..._legonet_n32_k3/results/smoke/legonet_full.json` — Table H quotes the *unlearn* label; this is the other one |
| mu — peft_compose IA³ 7B (0.6473) | `reband` → `external` | `E-a100/ia3_routed/mu`. CISPA A100; those JSONs are not in this snapshot, so the cell reports `rec`, never a pass |
| mu — sea (4-bit), s3t (`[F(d)]`) | `external` | Not `model_utility`; recorded with a citation |
| orphan — 4 routers × 2 drop sets | **`audit`** | `..._k200_r32_e25_lr1e4/results/router_leak/rl_family_k200.json`, dotted paths into `strategies.<s>.cells.<drop>.{orphan_capture.*,adequacy.mean,retain_shift_top1}` |
| orphan — `n_eff` | **`audit` + derived** | `...orphan_capture.@n_eff` — recomputed as 1/HHI from `top1_hist` by `verify_report.py`, in stdlib, independently of `analyze_orphan_destinations.py` which produced the published table. The reader also asserts the histogram sums to `n`. |
| orphan — self-detect AUC, tombstone rungs | `external` | Need the `.npz` score sidecars (~18 MB, deliberately not snapshotted). Recorded with the `analyze_router_family.py` command; sidecars sit beside each audit JSON in the checkpoint store |

Producer for every orphan cell: `router_family_audit.py`, driven by `submit_router_family.sh`
stages `j3` (7B k=200), `j7`/`j8` (7B k=10 feature/behavioral), `j9` (7B k=50), `j10`/`j11` (1B
plain k=10, the de-confound arm). Per-stage commands: [`METHODS.md`](METHODS.md).

## Table J — anchors

`Llama-3.2-1B-Instruct/results/smoke/{base_model,ft_all,ft_strong_scaf}.json` (0.3796 / 0.5302 /
0.6372); 7B base and `ft_r32` from `Llama-2-7B-chat-hf_nmerge_r32/results/smoke/{base_model,ft_r32}__own82.json`
(0.426 / 0.7563); locuslab's released full fine-tune via `eval_ft_minimal.py` / `verify_eval_tofu.py`
→ `tofu_ft_llama2-7b/results/`. 3B / TinyLlama / phi-2 base rows from their own pools.

## The ledger

Each thread keeps a dated research ledger under [`../log/`](../log/) recording the hypothesis, the
exact command, the SLURM job ID and often a script sha256 — the strongest provenance anchor
available for the routing cells. The entry behind this report is
[`../log/merge_mechanism/2026-07-24_merge-ceiling-vs-routing-master.md`](../log/merge_mechanism/2026-07-24_merge-ceiling-vs-routing-master.md).
