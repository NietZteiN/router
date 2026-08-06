# SETUP

From a bare clone to a running campaign.

## 1. Environments

**Python 3.12.** The pins are not decorative — `transformers` 5.x / `peft` 0.19 resolve the LoRA
config differently and cannot run this code.

```bash
conda create -n tofu python=3.12 && conda activate tofu
pip install -r tofu_sisa_lora/requirements.txt
```

`torch` is pinned to a CUDA 12.1 wheel (`2.5.1+cu121`). On different hardware install the matching
build instead (`pip install torch==2.5.1`, see pytorch.org) and leave the rest alone.

**A second env is required for the open-unlearning eval track** (the `Agg`/`Priv` rows of
sepmlp / blocktc / memadapt). It is a different `torch` *and* `transformers` major and cannot be
merged into the first:

```bash
conda create -n unlearning python=3.11 && conda activate unlearning
pip install -r requirements-ou.txt
```

**Plots need a third interpreter.** `matplotlib` is deliberately absent from `requirements.txt`;
`plot_*.py` run under `$TOFU_PLOT_PYTHON`. Point that at an interpreter that has it, or
`pip install -r apa_uniform_sum/requirements-plots.txt`.

## 2. Pick a site

```bash
export TOFU_SITE=local                          # or sprint / cispa / your own
export HF_HOME=$HOME/.cache/huggingface         # must contain hub/
export TOFU_CKPT_ROOT=$HOME/tofu_checkpoints    # needs ~100 GB for one 200-adapter pool + merges
```

`cluster_env.local.sh` sets no absolute path; the two values above have no sensible default and
fail loudly, naming the export that fixes them. Adding a cluster is one file — see
[`PORTING.md`](PORTING.md).

## 3. Verify, before anything expensive

No GPU, no network, no cluster, no model weights:

```bash
python test_repo_selfcontained.py          # 12 checks on the layout and the site layer
python snapshot_results.py --check         # 476 result files vs their sha256 manifest
cd tofu_sisa_lora && for t in test_*.py; do python "$t" || echo "FAIL $t"; done
```

## 4. Upstream clones

```bash
bash fetch_upstream.sh
```

Clones S3T, MemSinks and open-unlearning at pinned commits and applies the `ou_integration/`
registry patch. Until you run it, `memsinks_tofu/test_memsinks.py` and three
`memadapt_tofu/tests/test_data.py` cases fail by design — they check against upstream files.

## 5. Data and the gated model

| what | size | note |
|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 2.5 GB | the 1B track |
| `meta-llama/Llama-2-7B-chat-hf` | 13.5 GB | **gated** — needs an accepted licence + token |
| `locuslab/TOFU` | small | `full`, `forget10_perturbed`, `retain_perturbed`, `holdout10` |
| `sentence-transformers/all-MiniLM-L6-v2` | 90 MB | the centroid routers |
| `facebook/bart-large-mnli` | 1.6 GB | `sea/`'s zero-shot domain router only |
| `cais/mmlu`, `tatsu-lab/alpaca`, `fancyzhx/dbpedia_14` | small | OOD channels, the scaffold, the RAMoLE retriever |

Accept the Llama-2 licence on HuggingFace, then put the token where the job prologue looks for it:

```bash
mkdir -p "$HF_HOME" && printf '%s' 'hf_xxx' > "$HF_HOME/token"    # never committed
bash stage_hf_cache.sh                                            # pre-download into $HF_HOME
```

## 6. Checkpoints

Model weights are not in git — 674 GB for `tofu_sisa_lora` alone. Each project reads its store
through `$TOFU_CKPT_ROOT` / `$TOFU_CKPT_STORE`, so nothing needs a symlink; if you prefer the
original layout:

```bash
ln -s "$TOFU_CKPT_ROOT" tofu_sisa_lora/checkpoints
```

[`STATUS.md`](STATUS.md) lists, per result, whether it is checkable from `results_snapshot/` with
no GPU or needs a retrain, and what that retrain costs.

## 7. Run something

```bash
# preview any driver without submitting
TOFU_SITE=local STUB=1 bash tofu_sisa_lora/submit_router_family.sh

# the router-leak battery (needs an adapter pool)
bash tofu_sisa_lora/submit_router_leak.sh

# the CPU-only consumer of results already in the snapshot
python tofu_sisa_lora/analyze_router_family.py --help
```

**The 4-GPU global cap is enforced** by `cluster_env.sh`, summed across every queued job, not per
job. Check `squeue -u $USER -o "%.10i %.20j %.10T %.10b %F"` before submitting, and chain with
`DEP=afterany:<jobid>` rather than over-submitting.
