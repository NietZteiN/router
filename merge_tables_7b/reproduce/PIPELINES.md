# PIPELINES — rebuilding the pools from scratch (tier 3)

Tiers 1 and 2 ([README](README.md)) need nothing but Python. This is tier 3: regenerating the
result JSONs themselves, which needs GPUs, the TOFU dataset, and gated Llama weights.

Environment setup — Python, the `checkpoints` symlink, `HF_HOME`, the HF token, and which config
fields to repoint off-cluster — is already covered in [`../SETUP.md`](../SETUP.md); it is not
repeated here. On the CISPA build, every driver instead sources
[`../tofu_sisa_lora/cluster_env.sh`](../tofu_sisa_lora/cluster_env.sh), which is the single source
of truth for interpreter, partition, account, checkpoint root and the GPU cap.

**Run the CPU gate before every SLURM job.** They are fast, they catch the failure modes that
otherwise burn GPU-hours, and each pool below names its own.

**`STUB=1` on any `submit_*.sh`** prints the sbatch scripts it would submit, without submitting.

---

## Order of operations

The pools are independent except for two shared prerequisites:

1. **The retain90 oracle + KS reference.** `forget_quality` is a KS test against an adapter trained
   only on authors 0–179. Without `results/{tier}/retain_tr_scores.npy`, every `fq` is NaN. The
   oracle keeps the **legacy** recipe (r8/α16/e3/lr2e-4) explicitly — the newer defaults would
   invalidate existing KS references.
2. **The scaffolded base**, for every routing arm: `train_scaffold.py` → `make_scaffolded_base.py`.

```bash
python train_lora_shard.py --retain90 --k 10 --rank 8 --alpha 16 --epochs 3 --lr 2e-4 \
    --model_name <MODEL> --output_dir ./checkpoints
python prepare_eval.py --smoke --k 10 --output_dir ./checkpoints/<SLUG> --model_name <MODEL>
```

At k=4 the oracle needs `--retain_authors 150` (forget shard 3 = authors 150–199); the directory is
still named `retain90/` so `prepare_eval.py` finds it.

---

## P1 — Llama-3.2-1B, k=10 (Tables A, B; anchors for E)

No config file; the flag-free defaults of `train_lora_shard.py` **are** the frozen recipe
(rank 32 / α 64 / 5 epochs / lr 1e-4, winner of the 2026-06-11 grid).

```bash
bash submit_overnight.sh 10 meta-llama/Llama-3.2-1B-Instruct        # train 10 shards
# oracle + prepare_eval as above
bash submit_eval.sh ./checkpoints/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B-Instruct 10
python collect_results.py --root ./checkpoints --smoke
```

Gates: `python test_ou_equivalence.py` (metric math vs open-unlearning) and
`python test_merge_extra.py` (every merge method's closed-form identities). Run both after touching
metric or merge code — they are what keep old result JSONs comparable.

The two routing rows need the scaffold arm as well:

```bash
python train_scaffold.py && python make_scaffolded_base.py
python train_lora_shard.py --shard_id $i --k 10 --model_name <scaffolded base path> \
    --output_dir ./checkpoints/Llama-3.2-1B-Instruct_experts_scaf_k10
cp .../results/smoke/retain_tr_scores.npy <that pool>/results/smoke/   # KS ref must be present
python eval_routed_scaffold.py --shards_dir ./checkpoints/Llama-3.2-1B-Instruct_experts_scaf_k10 ...
```

## P2 / Table C — Llama-2-7B, one pool per k

```bash
bash submit_llama2_grid_overnight.sh          # k = 4, 10, 20
bash submit_scale_grid.sh                     # k = 50, 100, 200 (chained, capped at 4 GPUs)
bash submit_merge_methods_eval.sh             # the k=4 merge battery + its retain75 oracle
```

`submit_scale_grid.sh` wires `gate_scale_load.py` between training and eval: it loads all k shard
adapters plus one k-way merge on one GPU, asserts the adapter count (the loader skips missing dirs
*silently*), and prints peak CUDA memory. A non-zero exit blocks the dependent eval array and
auto-submits the r8 fallback. Because `kill_invalid_depend` is off cluster-wide, the gates
`scancel` their own dependents on failure — do not remove that, or a failed chain hangs pending
forever.

**Budget:** the memory law is `13.5 GiB (7B bf16) + k · n_params(rank) · 4B` — PEFT casts adapters
to fp32 on load. Measured at k=200: r1 = 14.1 GiB, r8 = 24.9 GiB, r32 ≈ 65 GiB. Hence the r8
fallback at k=200 on a 46 GiB A40, or two 40 GB A100s with `device_map="auto"` on CISPA.

## P3 / Table D — per-author N-merge ladders

```bash
bash submit_nmerge.sh configs/nmerge_interference_7b.json all   # plan -> merge (CPU) -> eval (GPU)
bash submit_nmerge.sh configs/nmerge_centered_7b.json all
python analyze_nmerge.py --config configs/nmerge_interference_7b.json \
       --out_prefix reports/interference/nmerge
python analyze_nmerge.py --config configs/nmerge_centered_7b.json \
       --out_prefix reports/centered/nmerge
python plot_nmerge.py                        # base anaconda python only -- matplotlib is not in test-env
```

Gate: `python test_merge_subset.py`.

The merge stage is a **CPU** array (no gres) needing ~160 GB RAM and 32 CPUs — N=200 r32 factors
plus the SVD workspace. Only the eval stage wants a GPU, and it serves one materialized adapter via
`--preloaded_adapter`, never 200 resident ones.

⚠ The two `--out_prefix` values must differ. See
[CAVEATS #10](CAVEATS.md#10-reportsnmerge_mucsv-is-a-half-table).

## P4 / Table E — PEFT bake-off

```bash
python test_compose_peft.py                                    # gate; also writes reports/dora_merge_probe.json
bash submit_peft_bakeoff.sh configs/peft_bakeoff_1b.json all   # smoke -> train -> compose -> eval
```

The `smoke` stage runs four 2-step micro-trains, one per method — a real pipeline gate, not a
formality: prefix-tuning and VeRA both have setup that fails silently at scale. VeRA shards must
share `projection_prng_key` with `save_projection=True`, or the frozen basis differs per shard and
the composition is meaningless.

## P5 / Table F — full-parameter task vectors

```bash
python test_sift_masks.py                                          # exactness gate
bash submit_sift_masks_tofu.sh configs/sift_masks_tofu_1b.json all
python test_clamu.py                                               # exactness gate
bash submit_clamu_tofu.sh configs/clamu_tofu_1b.json all
for K in 1 4 16 50 100 200; do
  bash submit_clamu_tofu.sh configs/clamu_tofu_1b_K${K}.json all
done
```

Full-FT in **fp32** for determinism, embed + lm_head frozen (tied on Llama-3.2-1B). The K-dial
directories symlink `tau_bar.pt` / `author_emb.npy` from the K=16 build — those are K-independent —
so a dial run only re-clusters, re-localizes and re-evals.

`measure_sift_exactness.py` re-derives a forget author's τ twice on GPU and reports whether the
unlearn is bitwise-exact or only distributional. That is the claim behind "Exact (GPU bitwise)" in
Table F, and it is worth re-running rather than taking on trust.

## Table F′ — ctv arms

```bash
python test_struct_tv.py && python test_linear_tv.py && python test_ds_support.py
bash submit_ctv.sh configs/ctv_1b_ctrl.json all
bash submit_ctv.sh configs/ctv_1b_wd.json   all
bash submit_ctv.sh configs/ctv_1b_lin.json  all
bash submit_ctv.sh configs/ctv_1b_ds.json   all
```

The `verify` stage is a real gate, not a formality: `ds_support.py locality` asserts that each
author's stored parameter indices are a subset of its derived support and that supports are
pairwise disjoint, and **exits non-zero on violation**. Without it, "disjoint support" is an
assumption rather than a checked property.

## Part II / Table H — the out-of-tree threads

Vendored under [`../external/`](../external/), each with its own driver and store. The one that is
a hard dependency of this tree is **memsinks**: `eval_tofu.py --memsinks_config` sys.paths the
project directory from the config and imports `memsinks_routed_model.py` from it.

## Cost

The report's cells span roughly a GPU-week of accumulated work. The two arms with the worst
ratio of GPU-hours to cells are the 7B k=200 pools (200 single-author trains) and the T=200
full-FT builds (200 fp32 full fine-tunes). If you only want to check the *claim* rather than
rebuild the evidence, tier 1 does that in seconds.

**Respect the GPU cap.** Every array carries a `%N` throttle and the throttles of all queued jobs
must sum to the cluster's cap (4 on the original box, 16 on CISPA per `cluster_env.sh`). Check
`squeue` before every submit.
