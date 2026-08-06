# SETUP — run the plan as if on the cluster (no models in git)

This repo carries **all the code, configs, and result data** to run the plan in `PLAN.md`. It does **not**
carry model weights or checkpoints — those live on `/storage2` behind a symlink and are far too large for
git. To actually execute jobs you must be on the cluster (GPUs + `/storage2` + SLURM). Everything else is here.

## 1. Python environment (training / eval / merges)
Target **Python 3.12** (the cluster's `test-env`):
```bash
conda create -n test-env python=3.12 -y && conda activate test-env
pip install -r tofu_sisa_lora/requirements.txt
```
`torch` is pinned to the CUDA 12.1 wheel — swap for your CUDA/CPU build if needed (see the notes at the top
of `requirements.txt`). The `plot_*.py` scripts run under **base anaconda Python 3.13 + matplotlib**, which
is a *separate* interpreter (matplotlib is deliberately absent from `test-env`).

## 2. Recreate the `checkpoints` symlink (the models)
The scripts read/write model weights through `tofu_sisa_lora/checkpoints`, which in the original tree is a
symlink to the `/storage2` weight store. It is **not** in git. Recreate it (on a cluster node where
`/storage2` exists):
```bash
cd tofu_sisa_lora
ln -s /storage2/jack/checkpoints/tofu_sisa_lora checkpoints
```
On a machine without that path, point it wherever the pools live (the k=200 per-author pools are
`Llama-2-7B-chat-hf_k200_r32_e5_lr1e4`, `..._e25_lr1e4`, `..._r8_e5_lr1e4`, and the merge artifacts
`..._nmerge_r32*`, `..._ctv_sparse`).

## 3. HuggingFace cache + token
```bash
export HF_HOME=/storage2/jack/data/huggingface        # or your own cache dir
mkdir -p "$HF_HOME"
printf '%s' "<your-hf-token>" > "$HF_HOME/token"       # the submit_*.sh read this at job start
```
The gated `meta-llama/...` repos need an HF account with Llama access. No token is stored in git — it is
read at runtime from `$HF_HOME/token`.

## 4. Config paths to repoint (only if NOT on this cluster)
Most `configs/*.json` use `meta-llama/...` hub IDs (fine as-is). If your `/storage2` layout differs, edit
these absolute-path fields:
- `hf_home` — in the clamu / sift / ctv_ds / legonet / ramole / skills / entangled configs.
- `output_dir` / `out_dir` / `shards_dir` / `retain_tr_source` — in the `nmerge_*_7b.json`, `legonet_*`,
  `ramole_*`, `entangled_facts_1b.json` configs.
- Two configs use a **local** model path (not a hub ID): `sift_masks_tofu_1b_scaf.json` (`model_name`) and
  `entangled_facts_1b.json` (`base_model`) — repoint or regenerate the scaffolded base first.
The `HF_HOME` default `/storage2/jack/data/huggingface` is also hardcoded as a fallback in `eval_tofu.py`
and `train_peft_shard.py`; setting the `HF_HOME` env var (step 3) overrides it.

## 5. Run the plan
Priority-ordered next actions, GPU-hour estimates, and exact commands are in **`STATUS.md`** (and the full
rationale in **`PLAN.md`**). The pattern for every phase:
```bash
cd tofu_sisa_lora
python test_merge_extra.py          # CPU gate FIRST (per-phase gate listed in STATUS.md)
# then the SLURM driver, e.g.:
bash submit_ctv.sh configs/sparsify_7b.json          # w5 sparse evals (Phase M)
```
Respect the **global 4-GPU cap**: `squeue -u jack` before every submit; queued array `%N` throttles must
sum to ≤ 4 (the drivers refuse over-cap submits).

## Scope note
The merge-battery / PEFT / ctv workflows in the plan import as a self-contained tree — no sibling project
dir is required. The one exception is the `--memsinks_config` eval arm, which needs
`memsinks_routed_model.py` from the out-of-tree `~/memsinks_tofu` project; it is **out of scope** for this
plan and intentionally not vendored.

---

## 6. Checking the results without a cluster

Everything above is for *running* jobs. To **verify** the published numbers you need none of it —
no GPU, no `/storage2`, no HF token, no `test-env`. [`reproduce/`](reproduce/) carries a snapshot of
the result JSONs behind the master report plus a stdlib-only harness:

```bash
python reproduce/verify_report.py        # recompute every cell, diff against the report
python reproduce/rebuild_tables.py       # regenerate the report's tables from the JSONs
python reproduce/snapshot_results.py --check    # re-hash the snapshot against its manifest
```

Start at [`reproduce/README.md`](reproduce/README.md). Rebuild recipes for the pools themselves are
in [`reproduce/PIPELINES.md`](reproduce/PIPELINES.md), which assumes the environment set up above.

## 7. Scope note superseded

§5 above (and the original scope note) said the `--memsinks_config` eval arm needs
`memsinks_routed_model.py` from an out-of-tree `~/memsinks_tofu` that was **not** vendored. It is
now vendored, at [`external/memsinks_tofu/`](external/memsinks_tofu/), along with the other five
Part II projects — see [`external/README.md`](external/README.md). Their `/storage2` weight stores
are still excluded, so the code imports but the results must be regenerated.

## 8. Two cluster environments

The `submit_*.sh` drivers in this repo have been ported to CISPA A100s and read their settings from
[`tofu_sisa_lora/cluster_env.sh`](tofu_sisa_lora/cluster_env.sh) — interpreter, partition, account,
checkpoint root, GPU cap, bad-node excludes. Steps 1–4 above describe the original sprint-cluster
box (A40s, `/storage2`, conda `test-env`); on CISPA, `cluster_env.sh` is the single source of truth
and drivers must not hardcode paths.

⚠ **Check file sizes, not existence, before trusting a resumed pool.** The vendored k=200 adapter
pools arrived on CISPA as 0-byte placeholders, which satisfy every driver's skip-if-exists test and
silently produce empty runs. This and ten other traps are in
[`reproduce/CAVEATS.md`](reproduce/CAVEATS.md).
