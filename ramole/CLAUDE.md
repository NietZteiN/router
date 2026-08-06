# CLAUDE.md — ramole

Guidance for Claude Code working in this repo. Keep it current in the same change that alters
code. The dated experiment narrative lives in `~/log/ramole/` (see `~/log/README.md`); this file is
the *how the code works* map. Parent rules (`~/CLAUDE.md`) still apply: GPU jobs via SLURM on
sprint1-3 only, ≤4 GPUs total across ALL jobs in squeue, artifacts under `/storage2`, `HF_HOME=/storage2/jack/data/huggingface`,
seeds pinned, no destructive `rm`.

## What this is

A faithful **RAMoLE** (Retrieval-Augmented Mixture of LoRA Experts) implementation built **on top of
an existing LegoNet expert pool** so the learned composition is directly comparable to LegoNet's
uniform `1/k` delta-average. RAMoLE keeps the retrieve-then-compose shape but replaces both weak
links: a learned **LoraRetriever** (Stage 1) instead of frozen k-means keys, and a learned per-layer
**RouterLoRA cross-attention** (Stage 2) instead of `1/k`. Both are decoupled from individual
experts, so newly added experts route **zero-shot**.

We do **not** retrain experts: a config names a `source_run` (a `legonet_lora` run) and borrows its
trained adapters, corpus, frozen keys, and routing assignment. The cluster a record routes to is its
*task label* (same-cluster = contrastive positive; own-cluster = ideal expert; IID/OOD = whether the
own cluster is retrievable). Primary config: `configs/ramole_l32_3b.json` (base
meta-llama/Llama-3.2-3B-Instruct, source_run `legonet_l32_3b_n32_k3`, n=32, k=3).

**Unlearning tie-in.** Because the router is task-agnostic and trained on a held-out split (default
`router_train_split="reference"`, the disjoint reference split that built the frozen keys — LegoNet
Condition A), deleting a record retrains only its k affected experts and the router needs **no**
retraining. The O(1)/cascade-free deletion of the LegoNet pool is preserved; only the composition
rule is upgraded.

## Environment

`/home/jack/anaconda3/envs/test-env/bin/python` (torch 2.5.1+cu121, transformers 4.48.3, peft 0.14.0,
sentence-transformers 3.4.1, safetensors). Reuses `legonet_lora/` modules directly (added to
`sys.path` by `ramole_common.py`). Run via SLURM with `bash submit_ramole.sh`.

## Pipeline (one script per stage; all driven by a config JSON)

| Stage | Script | Notes |
|---|---|---|
| (experts) | — | reused from `source_run`; never retrained here |
| retriever | `retriever.py --config C --device cuda --stage {all,train,index,eval}` | instruction-prefixed encoder; contrastive InfoNCE FT on 40% clusters (`MultipleNegativesRankingLoss`); LoRA index = mean of m member embeddings/cluster; writes `retriever/`, `lora_index_n{n}.npy`, `results/retrieval_accuracy.json` (top-k IID/OOD, off-the-shelf vs FT). |
| router | `train_router.py --config C --device cuda` | deterministic AdamW on `{A_r,B_r}` only; base+experts frozen; **Random LoRA Dropout p=0.5** per step; writes `router.safetensors` + `router_meta.json`. |
| eval | `eval_ramole.py --config C --method {router,mean,perfect} --route {keys,retriever} --condition {iid,ood}` | one method/route/condition per call → `results/{label}.json`. Reuses `legonet_lora/eval_memorization.metrics_for_records` (EM/ES/VerbMem/ppl/canary). |
| alpha diag (E2) | `analyze_router.py --config C --route {keys,retriever} [--router_ckpt P] --n_eval 200 --device cuda --out J`; report: `--report "glob" --out MD` | captures per-layer alpha (`RouterController.capture_alpha`, opt-in, zero cost when off) over ONE teacher-forced b=1 forward per record — **never `generate`** (KV-cache decode l=1 breaks position pooling). Normalized entropy H vs uniform 1.0, max-share vs 1/m, ideal-expert mass, per-layer + completion-decile profiles, sharpness↔EM/−ln ppl Spearman (joins `results/router_{route}_iid.json` by id). Shared lib for `tofu_sisa_lora/analyze_router_tofu.py`; `--router_ckpt` serves an ablation router (e.g. d0) through this config's pool. |
| routing audit (E3) | `routing_audit.py --config C --tags d0 d1 d2 d_batch15 --n_retain 200 --device cuda --out J` | routing-only (no LLM): per deletion tag, builds the rebuilt retain-only index via `build_lora_embeddings(exclude_ids=manifest forget_ids)` and saves it to `{run_dir}/lora_index_n{n}_ex{tag}.npy` — the EXACT file `eval_ramole.py --index_policy rebuilt` loads (stale `lora_index_n{n}.npy` is sha256-asserted untouched; stale READ honors `retriever_run`). Routes forget + retain records stale-vs-rebuilt (rank-preserving argsort, k=cfg k): per-record orphan rows (rates only pooled across tags — d0-d2 are n=1 each), retain selection-shift, per-cluster cos displacement. Untouched-by-top-1 rows are bitwise-asserted only for clusters BEFORE the first affected top-1 cluster (`build_lora_embeddings` shares one RandomState across clusters, so later clusters re-sample members); exclude logic is checked exactly for ALL clusters via `_members_by_cluster` member sets. Tags without a manifest (e.g. `d_batch15` before its deletion runs) are skipped with a warning. **Router-leak Phase-3 additions (2026-07-20):** `--policies {stale,rebuilt,dropped,abstain}` (default `stale rebuilt` = byte-identical historical output, regression-locked in `tests/test_routing_audit.py`) — `dropped` = §9-D drop-an-expert (affected clusters masked −inf before ranking; per-tag + pooled `dropped_extras` with TOFU-verbatim keys incl. masked/unmasked top-1 sim ratio; `selection_shift_stale_vs_dropped`; hard-raises if surviving clusters < k — NO per-query fallback, unlike the tofu arm); `abstain` = the C1 fix arm ported (τ on RETAIN top-1 sims, percentiles {1,5,10} + 90/99% orphan-catch operating points, per tag AND pooled — forget data never touches τ); `--dump_sims` = `<out>.sims.npz` sidecar (stale sims forget+retain, affected mask, tag indices; never mutates the JSON). Consumed by `tofu_sisa_lora/analyze_router_family.py` (H-DATASET). |

Orchestration: `bash submit_ramole.sh CONFIG [setup|retriever|router|eval|all]`. `setup` runs on the
**login node** (encoder download only). retriever ∥ router run in parallel; eval is a SLURM array
(`%RAMOLE_CAP`, one task per method/route/condition) depending on both. `STUB=1` prints sbatch scripts
without submitting. Knobs in `slurm_nodes.sh` (`RAMOLE_MEM`, `RAMOLE_CAP`, times); `N_EVAL` (default
200). Heredoc footgun (inherited): keep each python command on ONE line.

Follow-up battery (E1–E6): `bash submit_followup.sh [all|wave1|wave2]` — wave 1: E1 seed-variance
router retrains (DBpedia configs `_s43/_s44`; TOFU `--seed/--router_out`) ∥ E6 `d_batch15` 15-record
deletion (seeded sample, excludes rec_000000-2) ∥ E3 audits (`routing_audit[_tofu].py`; the DBpedia
audit depends on E6) ∥ E2 alpha jobs ∥ E5 `benchmark_serving.py`; wave 2 (afterok): E1/E3/E4/E6 evals
(`%4`). E4 k-sweep configs `_k5/_k8` keep the run NAME unchanged (same trained router; serve-time k)
so every k-sweep eval passes `--label_suffix k{5,8}` to avoid result-file collisions.
`collect_followup.py` aggregates everything → `/storage2/jack/checkpoints/ramole/FOLLOWUP_REPORT.md`
(robust to missing cells).

Overnight campaign: `submit_overnight.sh` adds the ablation arms + unlearning demo on top of a
running Arm A — Stage R (ablation routers d0/corpus/r6, array `%RAMOLE_CAP`, `AFTER_JOB=` chains it
behind Arm A's eval to hold peak ≈4 GPUs) → Stage E (one dispatch eval array: ablation comparisons +
the 12 unlearning-demo evals, `kind|config|a|b|c` per task). `collect_overnight.py` globs every arm's
`results/*.json` into `runs/.../OVERNIGHT_REPORT.md` (robust to missing files). Ablation configs
`configs/ramole_l32_3b_{d0,corpus,r6}.json` set one field each + `retriever_run` (share Arm A's
fine-tuned encoder/index — see `LoraRetriever.load`). Unlearning demo: `eval_ramole.py --unlearn_tag
{d0,d1,d2} --unlearn_state {before,after}` serves the post-deletion pool (legonet
`post_unlearn_adapter_dir_fn` via `adapter_dir_fn`) through the UNCHANGED router.

## Configs

`configs/ramole_l32_3b.json` (primary). `configs/ramole_smoke.json` (TinyLlama source, MiniLM
encoder, n=4 k=2 — for a tiny GPU smoke). All hyperparameters live in the config: `source_run`,
`source_root`, `base_model`, `encoder_model`, `instruction`, `n`, `k`, `lora{rank,alpha}`,
`router{rank}`, `train_cluster_frac`(0.4), `dropout_p`(0.5), `router_train_split`(reference|corpus),
`train{...}` (router loop), `retriever_train{epochs,lr,batch_size,m_samples}`, `base_seed`, `corpus`.

## Layout (RAMoLE artifacts under `/storage2/jack/checkpoints/ramole/runs/{name}/`)

```
retriever/                 fine-tuned SentenceTransformer
lora_index_n{n}.npy/.json  (n, D) normalized LoRA embeddings + meta
router.safetensors         router-only params (A_r,B_r per target layer; 2·112 tensors at n=32 3B)
router_meta.json           train_clusters, scaling, rank, split, config_hash, final_loss
results/{label}.json       per-method eval; retrieval_accuracy.json
```
Experts/corpus/keys/assignment are read from `{source_root}/.../{source_run}` (legonet layout) and
never written.

## Tests (CPU, run before any SLURM job — CLAUDE.md §4)

`python tests/test_router_lora.py` — the core math: extraction (112 paths, GQA d_out split,
scaling=α/r), **single active expert ≡ that LoRA via PEFT** (load-bearing identity), m-identical ≡
one, per-sample mask routing, alpha sums to 1, no-NaN over dropout, router-only gradients, bitwise
save/load. `python tests/test_pipeline.py` — full fixture → retriever → router → eval(all methods) +
batched-routing ≡ per-sample. `python tests/test_unlearn_shared.py` — campaign additions:
adapter_dir_fn override, the real legonet `post_unlearn_adapter_dir_fn` post-deletion pool, and the
`retriever_run` shared-index key. `python tests/test_alpha_capture.py` — the E2 alpha-capture gate:
capture off by default, every installed path captured exactly once per b=1 forward with alpha (m,1,l)
summing to 1, captured alpha ≡ an external hook reproducing the q/k/s/alpha math (1e-6), near-uniform
at init (H_norm > 0.95, B_r std 0.01), `alpha_stats` unit math (known entropy, ideal present/absent,
m==1 guard, deciles, multi-entry raise), and the `capture_for_records` clear/restore contract.
`python tests/test_routing_audit.py` — the E3 routing-audit gate: fixture run + synthetic 1-record
manifest → rebuilt index lands in a NEW `_ex{tag}` file with the stale index bytes unchanged,
untouched cluster rows bit-identical (forget record picked with top-1 = last cluster for full
coverage), per-record orphan rows, missing-manifest tags skipped with a warning. `tests/_fixture.py` builds a tiny random Llama + random PEFT experts
+ synthetic corpus/keys/assignment (GQA via heads>kv_heads), so everything runs in seconds on CPU.

## Key design invariants

- **RouterLoRA must keep experts separate.** Each active expert's `v_i = (α/r)·B_i A_i x` is kept
  distinct and weighted by `softmax_i(⟨A_r x, B_r^T v_i⟩/√r)`. PEFT's `add_weighted_adapter`
  (LegoNet's combine) fuses experts into one delta and **cannot** express this — RAMoLE manages LoRA
  weights manually (not via PEFT) at router-train/serve time. `combine.py` is only the baseline.
- **scaling = α/r = 2.0** for these adapters (`use_rslora=False`) — NOT the √r convention of the tofu
  rslora shards. Read α,r from each `adapter_config.json`; assert the pool is homogeneous.
- **Per-layer A_r/B_r.** GQA ⇒ q/o have d_out=3072, k/v d_out=1024; a single global router pair is
  shape-incompatible. `install_router` sizes each from `base_linear.{in,out}_features`.
- **Router math in fp32** even when base+experts are bf16 (stable softmax/AdamW moments). A_r/B_r are
  fp32 Parameters; expert weights are non-persistent buffers (excluded from the router checkpoint).
  `freeze_to_router` enables grads on `.A_r`/`.B_r` only — assert nothing else gets `.grad`.
- **Random LoRA Dropout (p=0.5)** is what buys the OOD/zero-shot gains; keep it on. `sample_dropout`
  forces ≥2 survivors (no all-masked softmax) and draws from a seeded generator for reproducibility.
- **Determinism:** `set_determinism(base_seed)` first; bf16 on GPU is only *distributional* (as in
  legonet); bitwise on CPU. Router is saved/loaded by exact key-set match — `load_router` raises on
  mismatch.
- **Encoder is config-driven.** Default `hkunlp/instructor-xl` (paper). It is NOT pre-cached and may
  not load cleanly under sentence-transformers 3.4.1 — `submit_ramole.sh setup` verifies it on the
  login node; if it fails, fall back by editing `encoder_model` to `hkunlp/instructor-large` or
  `sentence-transformers/all-MiniLM-L6-v2` (one line). The instruction is applied as a TEXT PREFIX
  uniformly (FT and inference), so any SentenceTransformer works.
- **Eval comparability:** `--route keys` feeds RAMoLE (`router`) and LegoNet (`mean`) the SAME
  frozen-key top-k so only the composition rule differs; `--route retriever` is full RAMoLE.
  `perfect` = the single top-1 expert (upper bound). OOD masks the record's own cluster.
- **Memory:** per-layer `v` is `(m,b,l,d_out)`; dropout halves m, batch_size=1/grad_accum=8 bound the
  peak; experts in bf16, score path fp32. Serving uses the deduped top-k union (m≪n).
