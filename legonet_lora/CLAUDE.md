# CLAUDE.md — legonet_lora

Guidance for Claude Code working in this repo. Keep this current in the same change that alters
code. The dated experiment narrative lives in `~/log/legonet_lora/` (see `~/log/README.md`); this file is the *how the code works* map.
Parent rules (`~/CLAUDE.md`) still apply: GPU jobs via SLURM on sprint1-3 only, ≤4 GPUs total across ALL jobs in squeue, artifacts
under `/storage2`, `HF_HOME=/storage2/jack/data/huggingface`, seeds pinned, no destructive `rm`.

## What this is

A faithful **LegoNet** (Yu et al., AAAI 2023) port to an LLM for **exact, verifiable record-level
unlearning**: a frozen base + `n` LoRA adapters addressed by **frozen k-means keys** in MiniLM
embedding space, with semantic **top-k k-NN routing**. Deleting a record = retrain only the `k`
adapters it activated; every other adapter is provably untouched. Plan:
`~/.claude/plans/papers-legonet-merging-loras-like-partitioned-russell.md`.

Two exactness conditions: **(A)** keys frozen at setup, never recomputed on deletion (derived from a
DBpedia *test*-split reference disjoint from the deletable corpus); **(B)** per-adapter training is
seeded + deterministic. Given A+B, the post-unlearn model equals a from-scratch retrain on `D\{r}`.

## Environment

`/home/jack/anaconda3/envs/test-env/bin/python` (torch 2.5.1+cu121, transformers 4.48.3, peft 0.14.0,
trl 0.9.6, datasets 4.8.5, scikit-learn 1.9.0, **sentence-transformers 3.4.1** installed `--no-deps`).
Run via SLURM with `bash submit_legonet.sh`.

## Pipeline (one script per stage; all driven by a config JSON)

| Stage | Script | Notes |
|---|---|---|
| corpus | `build_corpus.py --config C` | DBpedia-14 balanced subsample + per-record Secret-Sharer canary; `synthetic` source for tests. Train split = deletable corpus, test split = key reference. |
| keys | `keys.py --config C [--device cuda]` | k-means(n) over MiniLM embeddings of the reference → frozen `keys_n{n}.npy`. |
| routing | `routing.py --config C` | `KNNRouter` (pure numpy) top-k; caches `assignment_n{n}_k{k}.json` ({record→k keys} + {adapter→members}). Calls `keys.build_keys` first. |
| train | `train_adapter.py --config C --adapter j [--exclude_record_id ...] [--out_dir D]` | One seeded (BASE_SEED+j) deterministic LoRA on adapter j's members. CPU fallback: adamw_torch+fp32 when no CUDA (for tests). Empty member set → zero-delta "disabled" adapter. |
| infer | `combine.py` (`LegoNetModel`) | Loads base + adapters; `activated(idxs)` = `add_weighted_adapter(linear, w=1/k)` over the k. Eval groups records by adapter-set, merges once per group. |
| unlearn | `unlearn.py --config C --forget_record_id ... --tag T` | Retrain affected adapters (union of forget records' activated keys) on members minus forget set → `runs/{name}/unlearn/{T}/a{j}`. Untouched originals preserved. Whole-adapter-forgotten → O(1) disable. |
| eval | `eval_memorization.py --config C --which {base,legonet}` | EM/ES (OU formulas), VerbMem (LCS ROUGE-L recall), canary_hit (exact code in greedy gen). |
| verify | `verify_exactness.py --config C --mode {reproducibility,deletion}` | reproducibility: train j twice → param distance. deletion: oracle (from-scratch on D\F) vs unlearn (affected) / original (untouched sample). bitwise on CPU, distributional bound on GPU. |
| headline | `run_exactness_sample.py --config C --n_del N` | sample deletions → verify + forget-efficacy table → `results/exactness.json`, `reports/exactness_table.md`. |

Orchestration: `bash submit_legonet.sh CONFIG [setup|train|eval|exact|all]`. `STUB=1` prints sbatch
scripts without submitting. Env knobs: `LEGO_MEM` (64G 7B / 24G TinyLlama), `LEGO_ARRAY_CAP` (≤4, the global cap),
`N_DEL`, `N_EVAL`. `slurm_nodes.sh` holds the node policy.

## Configs

`configs/legonet_7b.json` (Llama-2-7B, DBpedia n=4000, n=32, k=3) — primary. `configs/legonet_smoke.json`
(TinyLlama, synthetic, n=4, k=2) — Phase-0 GPU smoke. All hyperparameters live in the config (CLAUDE.md
§5): base_model, n, k, lora{rank,alpha,dropout,target_modules}, train{epochs,lr,batch_size,grad_accum,
max_length}, base_seed, kmeans_seed, corpus{...}.

## Layout (under `/storage2/jack/checkpoints/legonet_lora`, symlinked as `./checkpoints`)

```
corpus/{corpus_name}/   records.jsonl  reference.jsonl  manifest.json
keys/{corpus_name}/     keys_n{n}.npy  keys_n{n}.json
runs/{name}/            assignment_n{n}_k{k}.json
            adapters/a{j}/      (LoRA + meta.json: seed, member_ids, config_hash)
            unlearn/{tag}/a{j}/ oracle/{tag}/a{j}/ verify/  results/  reports/  logs/
```
`corpus` + `keys` are shared across an n/k sweep (Condition A: keys never move).

## Tests (CPU, run before any SLURM job — CLAUDE.md §4)

`python tests/test_routing.py` (k-NN, frozen-key invariance, determinism), `test_metrics.py`
(EM/ES/ROUGE vs OU formulas), `test_exactness.py` (tiny random Llama: reproducibility bitwise,
untouched invariance, affected change, disabled zero-delta), `test_pipeline.py` (full
train→eval→unlearn→verify integration; deletion exactness bitwise on CPU).

## Key design invariants

- **Routing text = `content` only** (canary + title excluded) so clusters stay semantic; train text =
  `"{title}: {content} {canary}"`; eval prompt = `"{title}:"`, completion = `" {content} {canary}"`.
- **Frozen keys / cascade-free**: a record's k-NN depends only on that record + the frozen keys, so
  removing one record never changes another's assignment. `test_routing.test_frozen_key_invariance`.
- **Per-adapter seed = base_seed + j**; deterministic kernels via `set_determinism` (`warn_only=True`
  so kernels without a deterministic impl fall back → exactness becomes *distributional* there, which
  `verify_exactness` measures rather than asserts). Bitwise holds on CPU.
- **Exactness ≠ combine rule.** Delta-average is fixed for inference; exactness is a property of the
  per-adapter *training*. Don't "improve" the combine rule expecting it to affect exactness.
- **We merge only k adapters per inference, never all n** → avoids the PEFT fp32-cast memory wall that
  bit the tofu high-k path. Loading all n into the pool is fine at n≤~64; revisit for large n.
- **Canary is the clean forget signal**: a never-trained model can't reproduce a random 12-char code,
  so `canary_hit` ≈ 0 at baseline, high when trained, ≈ 0 after unlearning — robust to the 7B having
  seen Wikipedia. EM/ES/VerbMem over the whole completion are secondary (contamination-aware).
- **Load imbalance is a known failure mode** (DBpedia n=32: adapter sizes 83–664, ~8×). Mitigation if
  it bites: balanced k-means / size cap / larger n. Mean cluster purity ≈0.63 vs 14 classes confirms
  routing is genuinely semantic (the TOFU collapse this design avoids).
- Heredoc footgun (inherited from tofu): keep each python command in `submit_legonet.sh` on ONE line;
  backslash continuations inside `$(sbatch <<EOF)` get double-escaped into literal-space args.
