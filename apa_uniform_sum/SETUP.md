# SETUP

From a bare clone to a running campaign. Roughly: 20 min of setup, ~14 GB of downloads, and
~6.5 GPU-hours to retrain the adapter pools.

## 1. Environment

Python 3.12. The pins are not decorative — `transformers` 5.x / `peft` 0.19 resolve the LoRA
config differently and cannot run this code.

```bash
conda create -n tofu python=3.12 && conda activate tofu
pip install -r requirements.txt
```

`torch` is pinned to a CUDA 12.1 wheel (`2.5.1+cu121`). On different hardware install the
matching build instead (`pip install torch==2.5.1`, see pytorch.org) and leave the rest.

**Plots need a second interpreter.** `matplotlib` is deliberately absent from
`requirements.txt`; `plot_*.py` run under `${TOFU_PLOT_PYTHON}`. Either point that at an
interpreter that has matplotlib, or:

```bash
pip install -r requirements-plots.txt
```

Verify — no GPU, no network, no cluster:

```bash
python test_repo_selfcontained.py && python test_ou_equivalence.py && python test_merge_subset.py
```

## 2. Pick a site

```bash
export TOFU_SITE=local                          # or sprint / cispa
export HF_HOME=$HOME/.cache/huggingface         # must contain hub/
export TOFU_CKPT_ROOT=$HOME/tofu_checkpoints    # needs ~100 GB for both pools + merges
```

`cluster_env.local.sh` sets no absolute path; the two values above have no sensible default and
fail loudly, naming the export that fixes them. To add a cluster, copy `cluster_env.local.sh`
to `cluster_env.<yoursite>.sh` and fill in the partition/account/limits — that is the whole
port. Check it with:

```bash
python test_cluster_env.py
TOFU_SITE=<yoursite> STUB=1 bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json merge
```

## 3. Data and the gated model

| what | size | note |
|---|---|---|
| `meta-llama/Llama-2-7B-chat-hf` | 13.5 GB | **gated** — needs a token |
| `locuslab/TOFU` | small | `full`, `forget10_perturbed`, `retain_perturbed`, `holdout10` |
| `cais/mmlu` (`all`) | small | Experiment A/C general-knowledge channel |
| `tatsu-lab/alpaca`, `fancyzhx/dbpedia_14` | small | Experiment C far-OOD tiers (optional) |

Accept the Llama-2 licence on HuggingFace, then:

```bash
printf '%s' "$HF_TOKEN" > "$HF_HOME/token"     # the drivers read it from disk, never from git
huggingface-cli download meta-llama/Llama-2-7B-chat-hf
python -c "
from datasets import load_dataset
for c in ('full','forget10_perturbed','retain_perturbed','holdout10'):
    load_dataset('locuslab/TOFU', c)
load_dataset('cais/mmlu','all')"
```

Pre-warm before submitting: the jobs run with `HF_HUB_OFFLINE=1`, so a cache miss is a job
failure rather than a download.

**On a cluster with networked storage, wire in `stage_hf_cache.sh`.** Measured on CISPA: a cold
13.5 GB load off NFS took ~9 minutes with 12 readers in flight, against ~12 s of actual
training — ~98% I/O overhead on every train and every eval.

## 4. Train the pools (~6.5 GPU-hours)

```bash
bash submit_pool.sh anchors     # ft + retain90 — RUN FIRST
bash submit_pool.sh pilot       # 2 authors: the end-to-end gate, ~2 min
bash submit_pool.sh r32         # 200 adapters, 5 epochs   -> Experiments A and C
bash submit_pool.sh e25         # 200 adapters, 25 epochs  -> Experiment B
```

Order matters, and two recipes are deliberately *not* uniform:

- **`anchors` first.** The forget-quality KS reference derives from the retain90 oracle; without
  it `forget_quality` is NaN and every downstream table has a hole.
- **The retain90 oracle keeps its legacy r8 / α16 / e3 / lr2e-4 recipe.** It differs from the
  r32 pool on purpose. "Fixing" it to match would move every forget-quality number ever recorded.
- **Experiment B needs e25, not e5.** The e5 adapters have almost no ROUGE headroom — isolated
  own-author `forget_rouge` runs 0.402–0.575 against a base floor of **0.404** — so a forget gap
  would be unmeasurable. e25 experts store ~100% of their signal.

`pilot` writes into the same directory as `r32` and the trainers self-skip on an existing
`adapter_config.json`, so running it first costs nothing.

Each 200-adapter pool is ~49 GB; merges add ~258 MB × N. Check free space before the ladder —
`merges/` is reclaimable (rebuildable from the config plus `merge_meta.json`) once the evals land.

## 5. Run a campaign

```bash
# Experiments A and C — they share every merge
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json plan
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json merge      # CPU
bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json norms      # CPU, parallel with merge
DEP=<merge_job> bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json eval
DEP=<eval_job>  bash submit_expa.sh   configs/nmerge_sum_expA_7b.json mmlu
DEP=<mmlu_job>  bash submit_expa.sh   configs/nmerge_sum_expA_7b.json contrib
OUT_PREFIX=reports/expA/nmerge bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json collect

# Experiment B — `plan` prints this chain with real job ids
bash submit_expb.sh configs/expb_selectivity_7b.json plan
```

Three things that will bite:

- ⚠ **`submit_nmerge.sh collect` defaults to the shared `reports/nmerge` prefix and will
  overwrite the existing ladder CSVs.** Always pass `OUT_PREFIX=…/reports/expA/nmerge`.
- ⚠ **Never leave two GPU arrays queued at once.** Each is `%${TOFU_ARRAY_CAP}`; on `sprint`
  that cap is the *global* budget, and SLURM may start anything pending at any moment. Chain
  with `DEP=`.
- ⚠ **`kill_invalid_depend` is off on these clusters**, so a dependent array still runs after its
  parent fails. The job bodies assert their inputs for that reason; `scancel` dependents
  yourself when a stage fails.

## 6. Figures

```bash
"${TOFU_PLOT_PYTHON}" plot_expa.py --summary reports/expA/expA_summary.csv \
                                   --norms   reports/expA/expA_norms.csv
"${TOFU_PLOT_PYTHON}" plot_expb.py --prefix  reports/expb/expb
"${TOFU_PLOT_PYTHON}" plot_expc.py --summary reports/expA/expA_summary.csv \
                                   --contrib reports/expA/expA_contrib.csv
```

All three are safe on partial results — a missing arm is annotated in the panel rather than
crashing the run. `--mode dark` renders the dark-surface palette (a selected set of steps, not
an automatic flip).

## 7. Rank ceiling — measure it, do not inherit it

`TOFU_MAX_EXACT_RANK` decides where the exact ladder stops and the `svd_rank` rungs begin. The
merge is a rank-32N concatenation held in fp32 beside the bf16 base:

```
materialized bytes ≈ 2.013e6 × rank × 4
```

1024 is the measured 44.5 GiB A40 figure (rank 2064 OOM'd). A 40 GB A100 has *less* headroom on
one card and more across two with `device_map="auto"`. Measure on your card and set it; do not
carry the A40 number over silently.

## Troubleshooting

| symptom | cause |
|---|---|
| `HF_HOME is unset and cluster_env.<site>.sh did not export it` | export it, or set it in your site file |
| `out_dir still contains an unexpanded variable` | the site file does not export `TOFU_CKPT_ROOT` |
| `Memory specification can not be satisfied` at submit | that site must not emit `--mem` — set `TOFU_SUPPORTS_MEM=0` |
| `matplotlib is not installed in this interpreter` | run plots under `${TOFU_PLOT_PYTHON}` |
| `forget_quality` is NaN everywhere | the retain90 oracle / KS reference is missing — run `submit_pool.sh anchors` |
| `forget_quality` NaN for one probe author | that author is in 20–179, which has no perturbed rows. Not a bug; not measurable |
| a merge exists but `contrib` skips it | `measure_expb_contrib` refuses SVD-compressed merges — block identity is gone |
