# SETUP — standing this tree up on a cluster

This repo carries **code, configs, and result data**. It does not carry model weights or
adapter checkpoints; those are either re-downloaded from HuggingFace or **retrained in about
6.5 GPU-hours** (see §4 — retraining is usually the right call, and it has been validated).

`CLAUDE.md` is the how-the-code-works reference. The dated experiment narrative lives in the
separate `log/` repo; clone it as a sibling so the `../../log/...` links in `reports/` resolve:

```
<parent>/
  tofu_sisa_lora/     <- this repo
  log/                <- the research ledger
  legonet_lora/       <- optional; only eval_mmlu.py needs it (see §2)
```

---

## 1. Pick a site

Cluster settings live in `cluster_env.<site>.sh`, selected by `$TOFU_SITE` (auto-detected from
the hostname, default `sprint`). `slurm_nodes.sh` is a shim over it, so all 57 `submit_*.sh`
keep working unchanged.

```bash
TOFU_SITE=cispa bash submit_expa.sh configs/nmerge_sum_expA_7b.json gates
```

**Adding a new cluster is one file.** Copy `cluster_env.sprint.sh`, set the interpreter,
`HF_HOME`, `TOFU_CKPT_ROOT`, partition/account/exclude, the array cap, and — the one that bites
— `TOFU_SUPPORTS_MEM`. On CISPA the nodes report `RealMemory=1` and the partition sets
`DefMemPerNode=UNLIMITED`, so **any `--mem` fails at submit time** with "Memory specification
can not be satisfied"; it must be dropped, not lowered. `tofu_sbatch_resources` handles that.

Validate a site's job scripts **before** you have an account on it:

```bash
TOFU_SITE=cispa STUB=1 bash submit_expa.sh configs/nmerge_sum_expA_7b.json mmlu
TOFU_SITE=cispa STUB=1 bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json norms
```

Submit-time config reads go through `LOCAL_PY`, not the target site's interpreter, so this works
from any machine. Check the emitted header has the right partition/account and the right
presence-or-absence of `--mem`.

## 2. Python environment

Target **Python 3.12**:

```bash
conda create -n tofu python=3.12 -y && conda activate tofu
pip install -r requirements.txt
```

`torch` is pinned to the CUDA 12.1 wheel — install the matching build for your CUDA. Two notes
that have cost time before:

- **matplotlib is deliberately absent.** The `plot_*.py` scripts run under a *separate*
  interpreter (`$TOFU_PLOT_PYTHON`). Everything else runs under `$TOFU_PYTHON`.
- **`eval_mmlu.py` needs `legonet_lora/`** as a sibling directory (it borrows `_mmlu_prompt` /
  `_pred_letter` verbatim so the MMLU number stays comparable with that project's runs). Set
  `LEGONET_DIR` to point elsewhere, or skip the `mmlu` stage if you don't need it.

## 3. HuggingFace cache + token

```bash
export HF_HOME=/path/to/hf_cache        # or set it in cluster_env.<site>.sh
mkdir -p "$HF_HOME"
printf '%s' '<your-hf-token>' > "$HF_HOME/token"   # read at job start; never committed
```

`meta-llama/Llama-2-7B-chat-hf` is **gated** — the token needs Llama access. Pre-warm the cache
once with network (the jobs run with `HF_HUB_OFFLINE=1`):

| repo | size | why |
|---|---|---|
| `meta-llama/Llama-2-7B-chat-hf` | 13.5 GB | the base model |
| `locuslab/TOFU` | ~125 MB | all 13 configs |
| `cais/mmlu` (`all`) | ~220 MB | the `mmlu` stage |
| `sentence-transformers/all-MiniLM-L6-v2` | 88 MB | routing arms only |

⚠ On a cluster whose `$HF_HOME` is on NFS, stage it to node-local disk. Measured on CISPA: a
cold 13.5 GB load with 12 readers in flight took **~9 minutes** against ~12 s of actual
training — ~98 % I/O overhead on every job. See `merge-tables-7b/tofu_sisa_lora/stage_hf_cache.sh`.

## 4. Checkpoints: retrain, don't copy

The adapter pools are ~105 GB and are **not** in git. Retraining is measured at **~6.5 GPU-hours
total** (40–46 s per shard; per-shard cost is dominated by process startup, not training):

| artifact | what | GPU-h |
|---|---|---|
| `retain90` oracle | authors 0–179, **legacy r8/α16/e3/lr2e-4** | 0.53 |
| `ft_lr1e4_e5_r32` | all 200 authors, one adapter | ~1.1 |
| `k200_r32_e5_lr1e4` | 200 per-author adapters, r32/α64/e5/lr1e-4 | 2.3 |
| `k200_r32_e25_lr1e4` | same at 25 epochs (Experiment B needs this) | 2.6 |

⚠ **Keep the retain90 oracle at its legacy r8 recipe.** It deliberately differs from the r32
pool; "fixing" it to r32 moves every `forget_quality` number in the repo.

Run the anchors **first** — the forget-quality KS reference derives from the oracle. A ready
driver exists at `merge-tables-7b/tofu_sisa_lora/submit_pool_7b.sh` (arms
`anchors | r32 | r8 | pilot`); start with `pilot` (2 authors) as an end-to-end gate before the
200-way fan-out. Trains self-skip on an existing `adapter_config.json`, so re-running is cheap.

**Retraining is distribution-identical, not bit-identical** — a different driver/torch changes
reduction order. That is fine for a new campaign and has been checked: pools retrained from
scratch on A100s reproduced the A40 `additive_mean` ladder to mean Δ +0.0001, max |Δ| 0.0143
(`merge-tables-7b/reproduce/CROSS_HARDWARE.md`). Copy the pools instead only if a *published*
number must reproduce bit-for-bit.

Sanity-check the disk budget first: materialized merges cost `2.013M × rank × 4 B` (fp32), i.e.
7.7 GiB per rank-1024 merge. The APA config alone writes 55 GiB.

## 5. Run something

Always gates first — they are CPU-only and catch the errors that otherwise surface as a failed
array an hour later:

```bash
bash submit_expa.sh configs/nmerge_sum_expA_7b.json gates
```

Then, respecting your site's GPU cap (queued array `%N` throttles must sum to it):

```bash
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json plan     # manifests, login node
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json merge    # CPU array, no GPU
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json norms    # CPU, independent
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json eval     # GPU array
DEP=<eval_jobid> bash submit_expa.sh configs/nmerge_sum_expA_7b.json mmlu
```

⚠ `submit_nmerge.sh collect` defaults to the shared `reports/nmerge` prefix and **will clobber
the existing ladder CSVs** — pass `OUT_PREFIX=.../reports/expA/nmerge`.

⚠ `sacct` is empty on the sprint cluster, so the drivers use file-existence idempotency
(`if [ -f "$OUT" ]; then skip`) rather than job accounting. That is portable; keep it.

⚠ Where `kill_invalid_depend` is off cluster-wide, an `afterok` chain hangs pending forever if
its parent fails — `scancel` the dependents yourself, or chain with `afterany` plus an in-task
input assert (which is what `submit_expa.sh` does).
