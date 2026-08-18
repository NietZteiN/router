# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintaining This File

Keep this doc current **in the same change** that alters the code — don't defer it. It is a
map of intent, not an exhaustive listing; update it when behavior or structure shifts, not for
cosmetic edits. Concretely:

- **New script** → add a one-line row to *Core Workflow*, *Comparison Tracks*, or *Eval Script Variants*, whichever fits.
- **New label prefix / merge method / routing strategy** → update the matching table in *Architecture* (`activate_label`, `MERGE_METHODS`, router strategies).
- **New checkpoint suffix or model** → update *Directory suffix conventions* and the models-in-rotation list.
- **New non-obvious rule, gotcha, or footgun you just hit** → add a bullet to *Key Design Invariants* (e.g. the `--k 10` rule came from a real ZeroDivisionError).
- **Changed a CLI flag** → fix every command block that uses it; commands here must stay copy-paste runnable (verify against the script's argparse).
- **Removed/renamed something** → delete or rename its mention; stale instructions are worse than none.

Scope: this file is the *how the code works* reference. The dated experiment narrative
(goals, results, observations) stays in `~/log/sisa_lora/` (see `~/log/README.md`) per the repo protocol — don't duplicate it here.

## Environment

Python environment: `/home/jack/anaconda3/envs/test-env`  
HuggingFace cache: `/storage2/jack/data/huggingface` (symlinked as `~/data`)  
SLURM cluster nodes: `sprint1`, `sprint2`, `sprint3` (never `sprint4`)

**Site abstraction (2026-07-28).** Those values are no longer hardcoded in the drivers. They live
in `cluster_env.<site>.sh`, selected by `$TOFU_SITE` (default: auto-detected from the hostname,
falling back to `sprint`), and `slurm_nodes.sh` is now a **shim** that sources `cluster_env.sh`
and re-exports the same six legacy variables so all 57 `submit_*.sh` that source it are
unaffected — pinned by `test_cluster_env.py`. ⚠ "Unaffected" cuts both ways: `TOFU_PARTITION` /
`TOFU_ACCOUNT` / `TOFU_SUPPORTS_MEM` are NOT among the six, so the 54 drivers that still write
`#SBATCH --partition=all` and `#SBATCH --mem=` by hand **cannot submit on cispa at all** — the
partition does not exist there and any `--mem` is rejected. Converting one to
`tofu_sbatch_resources` is the fix (`submit_k200_routed.sh` is the worked example, verified at
both sites); see STATUS.md for the full list. Sites: `sprint` (A40, partition `all`, 4-GPU
global cap, `--mem` honoured) and `cispa` (A100-40GB, partition `xe8545`, account `testing`,
cap 6, **`--mem` must NOT be emitted** — nodes report `RealMemory=1` and any `--mem` fails at
submit). New drivers should emit resources with `tofu_sbatch_resources <gpus> <cpus> <mem>` and
the in-job env with `tofu_job_prologue`, rather than writing `#SBATCH` lines by hand; that is
what makes a memory policy a site fact instead of a per-driver edit. `gpus=0` emits no `--gres`
(CPU-only stages stay off the GPU cap).

Configs may use `${TOFU_CKPT_ROOT}` etc.; `merge_subset.load_config` expands them (recursively,
leaving absolute paths untouched) and **hard-errors on an unresolved variable** — a literal
`${TOFU_CKPT_ROOT}/` directory got created on disk once before that guard existed. `tofu_env.py`
sources `slurm_nodes.sh` once so a script run by hand sees the same values a job would.

`TOFU_SITE=cispa STUB=1 bash submit_expa.sh CONFIG mmlu` previews another cluster's job scripts
**from here** — submit-time config reads go through `LOCAL_PY`, not the target site's
interpreter. That preview is how the "no `--mem` on CISPA" class of error is caught before
anything is submitted. Porting reference: `merge-tables-7b/tofu_sisa_lora/cluster_env.sh`
(the exercised CISPA A100 port) and its `reproduce/CROSS_HARDWARE.md`.

Run scripts with:
```bash
/home/jack/anaconda3/envs/test-env/bin/python <script.py>
```

> ⚠️ Shared cluster: other users sometimes run GPU processes without a gres allocation, so
> SLURM can place our 1-GPU job on an already-occupied card. Before debugging an OOM as a
> memory-math problem, check the foreign PIDs listed in the CUDA OOM message (seen 2026-06-12:
> three ~7 GiB processes on sprint1 killed 6 eval tasks). Re-submit with `--exclude=<node>`.

## Core Workflow

The pipeline has four stages, each with a separate script:

**1. Train shards** — one LoRA adapter per data shard, runs per-shard on SLURM:
```bash
python train_lora_shard.py --shard_id 9 --k 10 --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir ./checkpoints
```
Flag-free defaults = the **frozen shard recipe** (rank 32, α 64, 5 epochs, lr 1e-4 — winner of
the 2026-06-11 grid, `reports/SHARD_GRID_REPORT_2026-06-11.md`; best at k=1/4/10). Checkpoints
trained before 2026-06-11 used r8/α16/e3/lr2e-4 (now "legacy recipe").

**Negative-anchored isolation (merge_mechanism §6.3, 2026-07-15):** `--anchor_lambda λ`
(+ `--anchor_n 2000 --anchor_seed 42 --anchor_batch_size 4 --anchor_source alpaca`) adds
`λ · mean_{LoRA modules, anchor tokens} ‖scaling·B(A(h))‖²` on public Alpaca text to the SFT
loss (`AnchoredSFTTrainer` + `anchor_penalty`, hooks on `lora_B`, fp32, pad-masked) — the
off-author negatives isolated training lacks; anchor set is public + seeded so exact deletion
survives. **Defaults OFF — flag-free behavior is bit-identical to the frozen recipe.** Recorded
in `shard_meta.json` (`anchor_*` fields). CPU gate: `python test_train_anchor.py`. Design:
`log/merge_mechanism/2026-07-15_negative-anchor-design.md`.

**2. Prepare eval** — caches the retain90 oracle's forget-set truth-ratio reference
(`retain_tr_scores.npy`, used by the `forget_quality` KS test) and writes the eval label manifest.
Requires the `retain90/` adapter first — oracles keep the **legacy recipe explicitly** (do not
rely on the new defaults; existing KS references were built at r8):
`python train_lora_shard.py --retain90 --output_dir ... --model_name ... --k 10 --rank 8 --alpha 16 --epochs 3 --lr 2e-4`
```bash
python prepare_eval.py --smoke --output_dir ./checkpoints/TinyLlama-1.1B-Chat-v1.0 --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --k 10
```

**3. Eval one adapter** — runs a single label through all TOFU metrics:
```bash
python eval_tofu.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir ./checkpoints/TinyLlama-1.1B-Chat-v1.0 --label merged_dare_ties --k 10 --forget_shard_id 9 --out result.json --smoke
```

**4. Collect results** — walks every per-model subdir under `--root` and merges their `results/{smoke,extended}/*.json` into one CSV (`<root>/all_metrics[_smoke|_extended].csv`):
```bash
python collect_results.py --root ./checkpoints --smoke
```

SLURM batch submission (runs all labels in parallel):
```bash
bash submit_eval_smoke.sh ./checkpoints/TinyLlama-1.1B-Chat-v1.0 TinyLlama/TinyLlama-1.1B-Chat-v1.0 10
```

**k-scaling sweep** — `submit_scale_grid.sh` submits the whole k ∈ {50, 100, 200} chain
(stage-1 train+prep array → gates → eval arrays → collect) capped at 4 concurrent GPUs (the global cap);
`STUB=1` prints every sbatch script without submitting; `backup_r8` arg submits only the
k=200 r8 fallback chain (auto-invoked by a failing r32 memory gate). `gate_scale_load.py`
is its go/no-go gate: loads ALL k shard adapters + one k-way merge on 1 GPU, asserts the
adapter count (the loader skips missing dirs silently), prints peak CUDA memory; nonzero
exit blocks the dependent eval array.

## Comparison Tracks (baselines & references)

The SISA-LoRA pipeline above is the novel method. Two other tracks exist purely as
reference points for it — both reuse the same eval/metrics code.

**Full-data LoRA fine-tune (`{slug}_ft/`)** — k=1, one adapter trained on *all* TOFU
authors (rank=8, 3 epochs). This is the "retain everything" upper bound and the starting
point for the gradient baselines. Built by `submit_ft_baseline.sh`.

**Gradient-based unlearning baselines (`{slug}_ft_unlearn_{method}/`)** — the TOFU paper's
own methods, for utility comparison against SISA-LoRA. `train_tofu_unlearn.py` loads a
`{slug}_ft` adapter and fine-tunes it with one of four objectives:

| `--method` | Objective |
|---|---|
| `ga`  | Gradient Ascent — maximize loss on forget set |
| `gd`  | Gradient Difference — GA on forget + CE on retain |
| `kl`  | GA on forget + KL(orig‖cur) on retain (ref = CPU snapshot of initial LoRA, swapped in per step) |
| `idk` | Preference opt — CE on forget-with-IDK-answers + CE on retain |

```bash
bash submit_tofu_unlearn.sh            # 6 models × 4 methods
bash submit_tofu_unlearn.sh ga phi-2   # one method / model; prereq: submit_ft_baseline.sh
```
Hyperparams match the TOFU paper (lr=1e-5, effective batch=32, 5 epochs). Each writes only
`shard_0/` plus `shard_meta.json`. The `forget10_idk` split is absent from the HF dataset,
so `idk` falls back to 20 hardcoded IDK strings.

> ⚠️ Unlearn/ft checkpoints have **only `shard_0`**, but must still be evaluated with
> `--k 10 --forget_shard_id 9` so the forget10/retain90 split matches the SISA-LoRA runs.
> `load_all_shard_adapters` skips the missing shard dirs. Using `--k 1` makes the whole
> dataset the forget set → empty retain → ZeroDivisionError. Re-run `prepare_eval.py` to refresh
> the cached `retain_tr_scores.npy` if you change `k`.

**LegoNet arm (`{slug}_legonet_n{n}_k{k}/`)** — a LegoNet-inspired comparison method
on the same TOFU metrics. Frozen base + `n` LoRA adapters addressed by **frozen k-means
keys** over the 200 authors' answer-mean MiniLM embeddings, with **author-level top-k
k-NN routing** and a **1/k delta-average** combine (`add_weighted_adapter(linear)`).
The forget unit is the author; deleting authors retrains only the affected adapters
(union of their top-k keys), leaving the rest byte-identical (frozen keys ⇒ cascade-free).
Built/served entirely inside this repo so it reuses `eval_tofu.py`'s OU-faithful
`model_utility`/`forget_quality`. Recipe lives in the config (`configs/legonet_tofu.json`,
default rank16/α32/[q,k,v,o]/6ep/lr2e-4, **`use_rslora=False`** so the 1/k average is a
true mean — do not compare its scale to the rslora SISA merges without noting this).

Pipeline (one config drives it; `bash submit_legonet_tofu.sh CONFIG [all|setup|train|unlearn|eval|collect]`, `%4` GPU cap, `STUB=1` previews):
| Stage | Script | Role |
|---|---|---|
| setup | `prepare_legonet.py --config C --device cuda` | frozen `author_emb.npy` + `keys_n{n}.npy` + `assignment_n{n}_k{k}.json`; prints authors/adapter, empty-adapter count, forget-set affected adapters. Symlink the SISA `retain90/` in + `prepare_eval.py` for the KS reference (reused, method-independent). |
| train | `train_legonet_adapter.py --config C --adapter j` | one LoRA on adapter j's member authors (seed `base_seed+j`); reuses `train_lora_shard.load_shard_dataset`/`format_prompt`. |
| unlearn | `unlearn_legonet.py --config C --tag forget10 [--plan\|--only_adapter j]` | retrain affected adapters minus forget authors → `legonet/unlearn/{tag}/a{j}` + `manifest.json` (affected/untouched). |
| eval | `eval_tofu.py --legonet_config C [--legonet_unlearn_tag forget10] --label legonet_full\|legonet_unlearn --k 10 --forget_shard_id 9` | per-query top-k 1/k serve via `legonet_model.LegoNetRoutedModel` (bypasses `load_all_shard_adapters`/`activate_label`, like `--preloaded_adapter`). |

Configs: `configs/legonet_tofu.json` (Llama-2-7B), `configs/legonet_tofu_llama3p2_1b.json`
(Llama-3.2-1B-Instruct = the open-unlearning canonical 1B). Set `"balanced": true` (+
`"capacity_slack"`, default 1.5) for the anti-hub capacity-capped assignment (cap = ceil(slack·k·N/n));
vanilla top-k lets one near-global centroid become a hub on TOFU's partially-collapsed answer
embeddings (n=32: one adapter saw 135/200 authors). `{slug}_legonet_..._bal` = the balanced variant dir.

Helpers: `legonet_tofu.py` (config loader, paths, `KNNRouter`, balanced `_balanced_topk`, keys/assignment/affected/q2author),
`legonet_model.py` (loaders + the routed wrapper). CPU regression: `python test_legonet_tofu.py`
(run before any SLURM job). On-disk under `{slug}_legonet_.../legonet/`:
`author_emb.npy`, `keys_n{n}.npy`, `assignment_n{n}_k{k}.json`, `adapters/a{j}/`, `unlearn/{tag}/a{j}/`.
Routing resolves TOFU-author queries by the frozen author assignment (`q2author`) and OOD
queries (real_authors/world_facts) by nearest-cluster of the question embedding.

**RAMoLE arm (on the LegoNet-TOFU pool)** — the full RAMoLE method (embedding retrieval / RAG over
LoRAs + a learned RouterLoRA cross-attention) layered on the *existing* `{slug}_legonet_n32_k3` pool,
replacing the 1/k delta-average. Imports `ramole/router_lora.py` (RouterLoraLinear/extract/build/
save/load) and `ramole/ramole_common.py` (instruction encoder); reuses `legonet_tofu.py`
paths/assignment/q2author/KNN and `train_lora_shard` data loaders. No expert retraining.
| Stage | Script | Role |
|---|---|---|
| retriever (opt) | `train_retriever_tofu.py --config C --device cuda` | contrastive (InfoNCE, same-author pairs) fine-tune of the instructor-xl retriever on retain authors → `legonet/ramole/retriever/`; `_encoder_source` then auto-uses it. Closes the embed↔key routing gap. |
| index | `ramole_tofu.py --config C --build_index --device cuda` | per-expert embedding (mean of member authors' question embeddings, fine-tuned encoder if present) → `legonet/ramole/expert_index_n{n}.npy` (RAG index). |
| router | `train_router_tofu.py --config C --device cuda` | train `{A_r,B_r}` on **retain authors 0–179** only (experts frozen, Random LoRA Dropout p=0.5) → `legonet/ramole/router.safetensors`. |
| eval | `eval_tofu.py --legonet_config C --ramole_router PATH --ramole_route {embed,key} [--legonet_unlearn_tag forget10] --label ramole_full\|ramole_unlearn\|routerkey_full\|routerkey_unlearn` | builds `ramole_tofu.RamoleTofuModel` (per-query route → `controller.set_active` → RouterLoRA compose). `embed`=RAG retrieval, `key`=author lookup. |
Driver: `bash submit_ramole_tofu.sh configs/ramole_tofu_1b.json [all\|index\|router\|eval\|retriever]`
(≤4 GPU: `all` = index ∥ router → smoke %4 → extended %4; `retriever` = FT encoder → rebuild index →
re-eval the embed arm as `ramoleft_{full,unlearn}`). Labels: `ramole_*` = off-the-shelf embed,
`ramoleft_*` = fine-tuned embed, `routerkey_*` = key route (all + router). Report:
`ramole_tofu_report.py` (RAMoLE vs the on-disk `legonet_*` 1/k baseline). CPU regression: `python test_ramole_tofu.py` (before any SLURM job).
Config `configs/ramole_tofu_1b.json` = the 1B legonet config + `router{rank}` + `ramole_train{...}` +
`retriever_encoder`/`instruction`. The router is loaded UNCHANGED for full and unlearn (only the served
experts differ via `adapter_dir_fn`) — deletion needs no router retrain.

**SIFT-Masks arm (`{slug}_sift_masks/`)** — a faithful **full-FT** (not LoRA) build of
SIFT-Masks (Kuo et al. 2025, arXiv:2504.04626 "Exact Unlearning … via Model Merging at
Scale"), the only non-LoRA track. One task per author (T=200), so it shows the
merge-at-scale collapse that FT+Merge suffers and SIFT fixes. A global ±1 sign vector `v`
(seed-fixed, drawn before training, shared across tasks) constrains each task's full-FT so
`τ_c⊙v ≥ 0` (project after each Adam step); the free mask `m_c = 1{τ_c≠0}` depends only on
local data + `v`. Merge = streaming **sum** `τ̄ = Σ τ_c` (only `τ̄` + per-task bit masks kept,
≈model/32 each). Serve task t: `θ0 + (τ̄⊙m_t)/T`. **Exact unlearn** = deterministically
re-derive `τ_u` (same per-task seed) and subtract `τ̄ ← τ̄ − τ_u`, drop `m_u`, `T ← T−1` —
O(1), no retrain over the retain set. Trains in fp32 for determinism; embed+lm_head frozen
(tied on Llama-3.2-1B).

| Stage | Script | Role |
|---|---|---|
| build | `train_sift_masks.py build --config C` | SIFT-train all 200 author-tasks, stream `sift/tau_bar.pt` + `sift/masks/m_{a}.pt` + `sift/sign_v.pt` + `meta.json`. Reuses `sift_masks_data.load_tofu_full`/`build_task_batch` (answer-span loss). |
| unlearn | `train_sift_masks.py unlearn --config C --tag forget10` | re-derive forget authors' `τ_u`, subtract → `sift/tau_bar_forget10.pt` + `unlearn_forget10.json`. |
| eval | `eval_tofu.py --sift_masks_config C [--sift_unlearn_tag forget10] --label sift_full\|sift_unlearn\|merge_full\|merge_unlearn --k 10 --forget_shard_id 9` | builds `sift_masks_model.SiftMasksModel` (per-query oracle route via `legonet_tofu.build_q2author` → apply `θ0+(τ̄⊙m_a)/T` in place, cached by author; OOD → base `θ0`; **forgotten author → maskless merged `θ0+τ̄_tag/T′`**, the paper's Fig-8 held-out rule — H8 fix 2026-07-02, pre-H8 result JSONs served base θ0 for forgotten and are kept as `*.pre_h8.json`). `sift_*` = masked; `merge_*` = FT+Merge no-mask baseline (`θ0+τ̄/T`). Reuses the SISA retain90 KS reference. |
| secondary | `eval_sift_masks.py --config C --mode full\|unlearn` | the paper's own metric (answer probability, held-in/held-out) — a cross-check, not the OU headline. |

Driver: `bash submit_sift_masks_tofu.sh configs/sift_masks_tofu_1b.json [all|build|unlearn|eval|collect]`
(`STUB=1` previews; `build`→`unlearn`→`eval %2`→`collect`; `SIFT_LABELS`/`SIFT_TAG`/`SIFT_EVAL_ARGS`
env overrides). Follow-ups driver `submit_sift_followups.sh CONFIG [all|extended|ansprob|exact]` reuses
the build artifacts: `extended` = build the extended retain90 KS ref (`prepare_eval.py --extended`) +
extended-cap `eval_tofu`; `ansprob` = `eval_sift_masks.py`; `exact` = `measure_sift_exactness.py`
(re-derive a forget author's τ_u twice on GPU → bitwise-vs-distributional floor). Config
`configs/sift_masks_tofu_1b.json` (Llama-3.2-1B; `frozen_substr`, `steps`, `lr`, `sign_seed`,
`unlearn_tags`). CPU regression (the exactness gate, run before any SLURM job):
`python test_sift_masks.py`. Core lib `sift_masks.py` (sign vector, `sift_one_task`,
merge/serve/unlearn, mask pack/unpack) is model-agnostic via `frozen_substr`.

**T=200 result (2026-07-02, `log/sift_masks/2026-07-02_t200-results.md` + `_extended-caps.md`):**
SIFT-Masks reproduces the paper on OU metrics — `sift_full` mu **0.737** (matches joint-FT / k=1
ceiling) vs the FT+Merge collapse `merge_full` mu **0.407**; `sift_unlearn` deletes forget10
(subtract 20 re-derived task vectors) raising forget_quality **0.135→0.393** with mu preserved
(**0.738**). Extended caps confirm utility (0.7364/0.7370 vs 0.4051/0.4055); GPU unlearn is
**bitwise-exact** (`measure_sift_exactness.py`). ⚠️ Extended-cap fq for `*_unlearn` is LOW by
construction, not leakage (H8, resolved 2026-07-06): with the paper-faithful Fig-8 serving
(forgotten → maskless merged, the current `SiftMasksModel._apply`) extended fq = **0.0505** (was
0.0045 under the pre-H8 base-θ0 serving; mu bit-unchanged). The n=120 KS measures style-match to
the retain-FINETUNED oracle, and the T′=180 maskless merge is collapsed ≈ base — so extended fq
across tracks is only comparable within the same forget-serving style (legonet serves concentrated
retain-trained adapters → 0.89). Leakage is separately ruled out (bitwise audit + answer-prob
0.122 ≈ zero-shot). Pre-H8 JSONs kept as `*.pre_h8.json`.

**ClAMU arm (`{slug}_clamu/`)** — ClAMU (Kuo et al., *Exact Unlearning of Finetuning Data
via Model Merging at Scale*, ICLR-2025; `papers/ClAMU.pdf`), the **sibling of SIFT-Masks**:
same full-FT, T=200, deterministic per-task `τ_t`, streaming sum `τ̄`, exact unlearn by
re-derive-and-subtract. ClAMU differs in two ways — training is **not** sign-constrained
(`sift_masks.sift_one_task(..., use_sign_constraint=False)`, the only change to the SIFT
lib), and masks are **per-cluster** and **directly optimized** by a score-vector + STE
(`clamu.optimize_mask_ste`) rather than sign-derived. Authors are clustered (feature
k-means over MiniLM answer embeddings, K clusters; `cluster_affinity:"random"` for the
ablation) **before** finetuning, so clustering is frozen/cascade-free. Served:
`θ0 + (m_c⊙τ̄)/T` per query (oracle author→cluster route, reusing `legonet_tofu.build_q2author`);
the optimized mask is trained against that same `/T` form, so ClAMU is directly comparable
to the `merge_*`/`sift_*` labels. The point: re-evaluate ClAMU under the OU `model_utility`
(the paper reports only answer-probability) and the Global/EMR/TALL/ClAMU ladder.

| Stage | Script | Role |
|---|---|---|
| setup | `train_clamu.py setup --config C` | MiniLM author embeddings + k-means → `clamu/assignment_K{K}.json` (frozen). |
| build | `train_clamu.py build --config C` | full-FT all authors (no sign constraint), stream `clamu/tau_bar.pt` + per-cluster sums `clamu/cluster_sums/tau_c{c}.pt` + `meta.json`. |
| localize | `train_clamu.py localize --config C [--tag T] [--cluster J]` | optimize each cluster mask (STE) + derive EMR/TALL baselines from `τ_c` (full only) → `clamu/masks[_{tag}]/{clamu,emr,tall}_{c}.pt`. |
| unlearn | `train_clamu.py unlearn --config C --tag forget10` | re-derive forget `τ_u`, subtract → `clamu/tau_bar_{tag}.pt`; re-cluster the **retain** authors → `assignment_{tag}.json` + `unlearn_{tag}.json`. Then `localize --tag` rebuilds masks on retain data. |
| eval | `eval_tofu.py --clamu_config C [--clamu_unlearn_tag T] --label LABEL --k 10 --forget_shard_id 9` | per-query oracle route via `clamu_model.ClamuModel`. Label families: `clamu_*` (optimized), `emr_*`/`tall_*` (heuristic baselines), `merge_*` (no-mask Global merge) × `_full`/`_unlearn`. |

Driver: `bash submit_clamu_tofu.sh configs/clamu_tofu_1b.json [all|setup|build|localize|unlearn|localize_tag|eval|collect]`
(`STUB=1` previews; linear chain setup→build→localize→unlearn→localize_tag→eval %2→collect;
reuses the SISA retain90 KS reference). Config `configs/clamu_tofu_1b.json` (Llama-3.2-1B,
K=16; `num_clusters`, `cluster_affinity`, `mask_steps`, `mask_lr`, `mask_opt_seed`,
`mask_batch_rows` (cap the STE forward's batch rows — the 1B OOM fix; gradient checkpointing
is INCOMPATIBLE with `torch.func.functional_call`, so cut the batch instead), `tall_lambda`,
`unlearn_tags`); `configs/clamu_tofu_smoke.json` (TinyLlama micro). Optional keys:
`mask_epochs` (localize steps = ceil(epochs×members), overrides `mask_steps` — equalizes mask
training across K for the K-dial), `heuristic_masks: false` (skip EMR/TALL and drop the
cluster-sum requirement from build's resume guard), `forgotten_serve: "merged"` (Fig-8 /
H8-align serving: forgotten authors get the maskless retain merge `θ0+τ̄_<tag>/T` instead of
raw θ0 — still exact, makes extended-cap `forget_quality` oracle-comparable;
`configs/clamu_tofu_1b_fig8.json`). **K-dial dirs** (`configs/clamu_tofu_1b_K{1,4,16,50,100,200}.json`
→ `{slug}_clamu_K{K}`): `tau_bar.pt` / `tau_bar_forget10.pt` / `author_emb.npy` are
K-independent and **symlinked** from the K=16 build (absolute /storage2 targets); build and
the unlearn subtract skip via resume guards, so a dial chain only re-clusters + localizes +
evals. `cluster_authors` caps K at the author count (K=200 retain re-cluster → 180
singletons, not a sklearn error). CPU regression (the exactness gate, run before any SLURM
job): `python test_clamu.py`. Core libs `clamu.py` (STE mask opt, EMR/TALL masks, clustering,
`localize_steps`) + `clamu_model.py` (`ClamuModel` serve).

**composable_tv (ctv) Wave-0 arms (`{slug}_ctv_*`)** — four per-author task-vector arms
(ctrl / wd write-disjoint / lin linearized / ds disjoint-support full-FT) on one shared
config schema (pool = `merge_subset.subset_authors(pool_seed, pool_size)`, labels
`ctv_<arm>[_<variant>]_<sum|mean>_N<n>_s<seed>` + `iso_a<a>[_<variant>]`; thread
`log/composable_tv/`). One-line tool contracts (verified against each argparse):

- `train_struct_tv.py --config C --author A --arm {control,orthblock,rowslice}` (lib `struct_bases.py`) — ctrl AND wd trainer; fp32 adapters at `<out_dir>/<arm>/shard_<a>/`. Verify: `verify_struct.py --config C --arm ARM [--authors ids --drop_authors ids --no_placebos --out P]`. CPU gate: `python test_struct_tv.py`.
- `train_linear_tv.py --config C --author A [--rank_override R --lr_override LR]` (lib `linear_tv.py`, importable only — no CLI) — tangent-space (jvp-served) adapter at `<out_dir>/shard_<a>/` (PEFT-interoperable + `b_only.pt`). CPU gate: `python test_linear_tv.py`.
- `train_ds_support.py --config C --author A [--density d] [--overwrite] [--no_support]` (lib `ds_support.py`) — disjoint-support full-FT sparse tau at `<out_dir>/ds/tau_a<a>[_d<density>]/{tau_sparse.pt,meta.json}`. `--no_support` = the H-ds-1 unconstrained comparator (same recipe, NO projection): bakes θ0+τ in-job to `<out_dir>/ds_unconstrained/a<A>_model/` (no tau_sparse.pt; `ds_support.bake_dense_model`), served via the driver's `iso_dsunc_a<p>` `model:` rows (5-task `train_unc` stage). `python ds_support.py locality --config C [--out P] [--densities d1,d2]` = the verify-stage gate (stored idx ⊆ derived S_a, cross-author disjointness, empty-tau telemetry; writes `reports/ds_locality[_d*].json` under out_dir; NONZERO exit on violation); `python ds_support.py bake --config C (--n N | --authors ids) [--subtract ids] --out DIR` = headline dense bake (`bake(all,subtract=[a])` ≡ `bake(all∖a)` bitwise). CPU gate: `python test_ds_support.py`.
- Driver: `bash submit_ctv.sh CONFIG [gate|prep|train|train_unc|verify|merge|eval|w5_build|collect|all]` (`STUB=1` previews; `train_unc` = arm-ds-only H-ds-1 comparator array; generated sbatch bodies run `set -eo pipefail`; manifests are CONFIG-basename-keyed — `eval_manifest_<cfg>.txt` — so configs sharing an out_dir never collide; merge stage writes derived per-variant `merge_cfg_<v>.json` configs and runs `merge_subset.py merge --method additive_<sum|mean>` into `mtmp_<v>/merges/`, symlinking `merges/<ctv label>` to the ABSOLUTE target; lin[linear serve] + ds rows serve in-place via `lin:`/`ds:` serve-specs → `eval_tofu --linear_tv_*`/`--ds_*`, never `--preloaded_adapter`). Analyzer: `analyze_ctv.py --config C... [--out_prefix P --floor_prob F --floor_rouge F --tail_threshold t]`. CPU gate: `python test_analyze_ctv.py`.
- Configs: `configs/ctv_1b_{ctrl,wd,lin,ds}.json` + `configs/ctv_1b_lin_nlserve.json` (H-lin-2b: the SAME lin shards under standard nonlinear PEFT serving; `serve_mode:"standard"`, variant token `nl` → labels `ctv_lin_nl_sum_N<n>_s42`) + `configs/sparsify_7b.json` (w5).

## Eval Script Variants

| Script | Use for | Notes |
|---|---|---|
| `eval_tofu.py` | The SISA-LoRA pipeline | Only eval that understands merge/remerge/tree/routing labels via `activate_label`. Canonical metrics. `--preloaded_adapter DIR` loads base + one pre-materialized adapter (skips `load_all_shard_adapters`/`activate_label`) for the JD mode-B high-k path — full forget+utility metrics, no memory wall. `--prefix_pool_dir DIR [--prefix_exclude_shard i]` serves the peft_compose prefix arm (KV-concat of per-shard prefixes via `prefix_concat.PrefixConcatModel`; labels `prefixcat_full`/`prefixcat_unlearn`), same bypass. `--memsinks_config CFG [--memsinks_unlearn_tag forget10]` serves the MemSinks/SeqTD arm (`~/memsinks_tofu/`, thread `log/memsinks/`) with per-query ROUTED author masks — gen + own sink slice per TOFU author, deleted authors + OOD → gen-only — via `memsinks_routed_model.MemSinksRoutedModel` (project dir sys.path'd from the config path; labels `memsinks_routed_full`/`memsinks_routed_unlearn`); same flags mirrored in `attack_mia.py`. Baked (non-routed) memsinks conditions use plain `--preloaded_adapter <run>/baked/<mode>`. `--linear_tv_config CFG (--linear_tv_authors ids \| --linear_tv_n N) [--linear_tv_subtract ids] [--linear_tv_serve linear\|nonlinear-debug]` serves the ctv [lin] linearized composition (`linear_tv.load_linear_tv_eval_model`; labels `ctv_lin_*`/`iso_a*`). `--ds_config CFG (--ds_authors ids \| --ds_n N) [--ds_subtract ids]` serves the ctv [ds] merged disjoint-support full model (`ds_support.load_ds_eval_model`; subtract ≡ compose-without, bitwise). ⚠ exactly ONE arm selector allowed — `build_served_model` raises ValueError on conflicts (its elif chain otherwise silently preferred `--preloaded_adapter`). `--lazy_adapter_cache N` (routed_* labels ONLY, raises otherwise): keep ≤ N shard adapters resident — `lazify_shard_adapters` patches `set_adapter` to load-on-demand + LRU-evict via `delete_adapter` (never the active adapter; missing shard dirs RAISE instead of the eager path's silent skip); numerics identical to eager (same fp32 cast). This is what makes k=200 × r32 routed evals possible on an A40 (see the memory-law invariant). **`--forget_author_ids '180-199'` (selector_audit, 2026-08-07):** score forget_* on an EXPLICIT author set instead of a shard's. `split_eval_indices` otherwise derives the forget set as `shards[measure_id]`, so at k=10 shard 9 IS forget10 (400 questions) but at k=200 the same flag measures ONE author's 20 — not comparable to any published number. Parsed by `shard_utils.parse_author_ids` (inclusive ranges, commas, range-checked); mutually exclusive with `--eval_shard_id`; None = byte-identical legacy behavior. CPU gate: `test_eval_rows.py`. |
| `eval_ft_minimal.py` | Full-FT models (`locuslab/tofu_ft_*`) & single ft/unlearn adapters | Standalone. Was written to fix 4 metric bugs (plain prompt not chat template; `answer` not `paraphrased_answer` as truth-ratio ref; ROUGE-L **recall** not F1; **geometric** mean over truth-ratio samples). Those fixes are now folded back into `eval_tofu.py`. |
| `eval_baseline.py` | Base model, no adapter | Extended TOFU metrics for the untouched model. |
| `verify_eval_tofu.py` | Regression check (7B) | Runs `eval_tofu.evaluate_model` on `locuslab/tofu_ft_llama2-7b`; after the open-unlearning port expect `model_utility ≈ 0.62` (was ~0.70 under the old diverged metrics). GPU job: `sbatch verify_llama2_full.sh`. |
| `test_ou_equivalence.py` | Metric-math regression | CPU/micro proof that truth-ratio, `probability_w_options`, the TR aggregators and `hmean` reproduce open-unlearning's formulas exactly. Run after touching metric code: `python test_ou_equivalence.py`. |
| `test_merge_extra.py` | Merge-method regression | CPU micro-tests (tiny random Llama, k=3 shards) for every `merge_extra.py` method + dispatch: closed-form identities on *effective* deltas, determinism, subspace-leak checks. Run after touching merge code: `python test_merge_extra.py`. |
| `test_ensemble.py` | Ensemble regression | CPU micro-tests for `ensemble.py`: single-constituent ≡ direct adapter, k-identical ≡ single, exclusion, logsumexp reference, HF-loss replication, batched ≡ sequential path, generate semantics, determinism. Run after touching ensemble code: `python test_ensemble.py`. |
| `test_s3t.py` | S3T regression | CPU micro-tests for the S3T mapping (`shard_utils`) + `train_s3t_shard.py`: slice/ordering invariants, masking bit-identity incl. the substring-trap case, truncation exactness. Run after touching S3T code: `python test_s3t.py`. |
| `test_sift_masks.py` | SIFT-Masks regression | CPU micro-tests (tiny random GPT2) for `sift_masks.py`: ±1 sign vector & determinism; projection invariant (`τ⊙v≥0`, `mask==τ≠0`); byte-identical `τ_u` re-derivation; exact unlearning (`(Στ)−τ_u ≈ Σ_retain`, allclose — fp non-associativity, not bit-equal); serve identity; mask pack/unpack. Run before any SIFT SLURM job: `python test_sift_masks.py`. |
| `measure_sift_exactness.py` | SIFT GPU exactness | Re-derives a forget author's `τ_u` twice on GPU (deterministic math/eager attention) and reports `bitwise_identical` + `rel_l2_floor = ‖τ_a−τ_b‖/‖τ_a‖`. Quantifies whether the GPU unlearn is bitwise-exact or only distributional (cf. legonet floor ≈4–6e-2). `python measure_sift_exactness.py --config C --author 199 --out ex.json` (GPU; via `submit_sift_followups.sh CONFIG exact`). |
| `attack_mia.py` | Deletion-audit A4 — composed-model MIA (GPU) | Membership inference on the SERVED post-deletion composition (mirrors `eval_tofu.py`'s arm flags → `eval_tofu.build_served_model`, so it attacks the identical artifact eval scores). Member = TOFU `forget10`, non-member = `holdout10` (OU's TOFU_MIA pairing); prompt = `_build_qa_prompt` + answer-only labels (NOT a chat template — the biggest silent-failure risk); cheap battery `loss,min_k,min_k++,zlib` at `--batch_size 1`; AUC via the OU-faithful port `mia_attacks.py`. `python attack_mia.py --model_name M --output_dir . --label L [arm flags] --out .../results/mia/L.json`. **Embed-routed rider (2026-07-23):** `--shards_dir D --embed_route {sibling,tombstone} [--delete_shard j] [--router_encoder E]` serves the `EmbedRoutedModel` scaffold arm — the LEAKY surface the router_leak thread measures. It lives in `eval_routed_scaffold`, not `build_served_model`, so it is constructed directly (the core one-arm selector chain is untouched). Tests whether the router leak is visible to MIA at all, or blind like `fq`. Driver `submit_deletion_audit.sh [smoke\|phase1]` (config `configs/deletion_audit.json`); CPU gate `test_deletion_audit.py`. **E5 privacy column (2026-08-07):** `--delete_shards` / `--reroute_to j` build the ORACLE-routed reroute arm via `OODAwareRoutedModel`, exactly as `eval_routed_scaffold` builds it, so the MIA AUC and that arm's forget_quality describe the same served model. `--lazy_adapter_cache N` for k=200. |
| `attack_diff.py` | Deletion-audit two-snapshot diff MIA (CPU) | AUC of the per-example score CHANGE (post − pre) separating forget10 from holdout10 — the Unlearned-but-Not-Forgotten channel; consumes two `attack_mia.py` JSONs written with `--dump_scores` (per-example arrays in each per_attack block) for the same served config. `python attack_diff.py --pre PRE.json --post POST.json [--attacks loss,min_k] --out P`. Severity report only. |
| `verify_subtraction.py` | ctv shared exactness gate (CPU) | Verifies the declared subtraction-exactness class of a deletion: `python verify_subtraction.py --merged DIR --remerged DIR --tau DIR [--tau_weight w] [--declared_class algebraic\|first_order\|...] [--report P]` — per-module bitwise + rel_l2 table, exit 1 when the declared class fails (gates the ctv merge stage). NB takes adapter DIRS, not --config. CPU gate: `python test_verify_subtraction.py`. |
| `sparsify_pool.py` | ctv [w5] post-hoc sparsification grid (CPU) | `python sparsify_pool.py --config configs/sparsify_7b.json [--ops dare0p9 topk0p25 hash dare0p9sum] [--n_ladder ...] [--dx1 --dx2] [--limit_authors n --dry_run --force --report_dir D]` — zero-training dare/topk/hash sparsify+compose grid on the 7B `_k200_r32` pool, labels `sparse_<op>_N<n>_s42`; `--dx1/--dx2` = cancellation/owned-energy diagnostics. **`<op>sum` variants (2026-07-20, e.g. `dare0p9sum`/`dare0p99sum`): identical masks to the non-sum twin (shared adapter dirs via the canonical op name) composed at weight 1.0 each instead of 1/N — the doc-1 DARE+naive-sum cell; exact drop-a-term deletion w.r.t. the DARE'd deltas.** ⚠ the eval manifest is REWRITTEN from the ops passed — rerun with the FULL config ops list (merges self-skip) or the manifest loses prior rows. One CPU SLURM job via `submit_ctv.sh CONFIG w5_build`. CPU gate: `python test_sparsify_pool.py`. |
| `mia_attacks.py` | MIA scorer library (CPU import) | Self-contained port of open-unlearning's `evals/metrics/mia/` (the OU package can't be imported — `evals/__init__.py` needs omegaconf, absent in test-env). `evaluate_probability`/`tokenwise_logprobs`/`tokenwise_vocab_logprobs` (per-example from `output.logits`, never `output.loss` → works on every composed wrapper) + `score_batch` (loss/min_k/min_k++/zlib) + `mia_auc` (OU label convention: forget=0/holdout=1, AUC→1 when members more likely). Port proven equal to a direct computation in `test_deletion_audit.py`. |
| `test_deletion_audit.py` | MIA harness regression | CPU micro-test: planted leaky-vs-clean toy model (all attacks separate AUC>0.75 vs ≈0.5 — the harness go/no-go), determinism, loss/min_k port-equivalence vs a direct hand-computation (Δ<1e-5), collator answer-mask + index contract. Run before any deletion-audit SLURM job: `python test_deletion_audit.py`. |
| `entangle_data.py` | Entangled-facts (Mode-B) plant | Builds the plant manifest (single source of truth for WHAT is planted WHERE; deterministic from `seed`; refuses overwrite) + `load_planted_shard_dataset` (a normal shard gets its host rows; `shard_id=None` = the retain oracle gets all planted rows). Donors partition forget10 across R∈{1,2,4,8}; hosts are retain authors 40–179 placed in shards 2–8 (never 0/1/9); paraphrase-mode plants TOFU's `paraphrased_*` and probes the original question. `python entangle_data.py --config configs/entangled_facts_1b.json --out .../plant_manifest.json`. |
| `eval_entangled_probe.py` | Residual-fact-recall (RFR) probe (GPU) | Per fact × surface (orig/para) × channel: `expert_max` = max answer-prob over surviving single experts ("is the fact still in the weights"); `served_key` = the real OOD-aware routed+scaffold composition (`--drop_shard 9`, hard author-key route → base+scaffold for the deleted authors); `served_embedsim` = a training-free per-shard-centroid **embedding** router (MiniLM over member questions). `--embed_policy sibling` (default, prior behavior) excludes the dropped centroid — probes fall to the nearest surviving host; `--embed_policy tombstone` keeps it as an identity sentinel — top-1-on-dropped probes serve base+scaffold (route −1; the H4 seal arm); `--embed_policy tombstone_author` uses PER-AUTHOR sentinel centroids of the dropped shard instead (the Phase-1 winning rung, c_probe≈0.97 vs shard 0.48–0.76). **`--embed_strategy <router.py strategy>`** (router_leak Wave-3, 2026-07-23) routes `served_embedsim` via a real `router.py` family (centroid_lm/ppl/activation_norm/logit_div/…) built through `merge_lora._build_routed_model` instead of the MiniLM centroid — the per-router-family Mode-B leak arm; requires `--embed_policy sibling` (no seal). **`--embed_abstain_tau <float>`** (2026-07-23) = the router-NATIVE seal: abstain to base+scaffold (route −1) when the router's own best-vs-runner-up margin < tau ("nothing fits well ⇒ orphan"), needing NO stored identity sentinel unlike the tombstone; requires a router exposing `score_candidates()` (`PplRouter`), records `served_embedsim_margin_*` per fact, and tau must be calibrated on RETAIN margins only. `--rouge` now also returns generations; `--dump_generations` stores them in per_fact (`*_gen_*` keys, excluded from aggregates). answer-prob + optional ROUGE-L. Run on planted (no-drop=ceiling, drop=post) and clean (`clean_experts_dir`=oracle-B floor) → ρ via `aggregate_rho.py`. Reuses `_answer_avg_loss`/`OODAwareRoutedModel`/`build_shard_centroids`. |
| `detect_entanglement.py` | SEUF §9-A entanglement detector (GPU) | Per forget-fact NLL-affinity `Δ_j = NLL_scaffold − NLL_j` over experts → softmax → spread (mass off the donor's expert). ROC-AUC of spread separating planted (R≥2) vs control (R=1), precision/recall at `detector.threshold`, host-shard identification vs the manifest. The actionable delete-propagation trigger. |
| `test_entangled_facts.py` | Mode-B plant regression | CPU gate: manifest determinism, condition counts (200 facts, 25 verbatim+25 paraphrase per R), host constraints (R−1 distinct host shards in 2–8, retain 40–179, none in 0/1/9), planted-loader row math (shard vs retain-oracle-all), probe-set partition, ρ/detector math. Run before any entangled-facts SLURM job: `python test_entangled_facts.py`. Driver `submit_entangled_facts.sh CONFIG [manifest\|link\|train\|prep\|probe\|detect]`; artifacts dir `{slug}_entangled_k10` (shards 0,1,9 symlinked from `_experts_scaf_k10`). |
| `test_s3t_sequences.py` | S3T paper-repro regression | CPU tests for `s3t_sequences.py` (Alg 1 cyclic rotation, Alg 2 BMS) + `s3t_deletion.py` (Alg 4 best-surviving, deletion-rate simulation): diversity, δ matches the Lemma-1 closed form within CI, δ grows with B up to L. Run after touching repro code: `python test_s3t_sequences.py`. |
| `s3t_sequences.py` / `s3t_deletion.py` | S3T paper repro (CPU) | Slice-sequence selection (Alg 1 cyclic rotation, Alg 2 BMS, Eq 24 score) and the deletion-stream simulator (Alg 4 best-surviving, coupon-collector deletion rate δ, performance-vs-#deletions, Lemma-2 retention closed forms Eq 18/20, `expected_retrains` for Fig 9). Pure CPU; the deletion-rate headline needs no GPU/training. NB: Eq-18 uses the self-consistent `(1-k/L)^r` (the printed `1-(k/L)^r` is inconsistent with Eq-21 / S3T(B=1)=SISA). |
| `s3t_rq3.py` | S3T RQ3 / Fig 8 (CPU) | Sequence-selection diversity: avg pairwise edit distance of cyclic rotation vs random (uniform prior) and BMS vs random vs sorted-cyclic (non-uniform), + Eq-24 score. Writes `rq3_diversity.json`. |
| `s3t_measure_F.py` | S3T `F(d)` driver | `build` = create `_depth{d}` symlink eval dirs from existing armA stage snapshots; `collect` = assemble `F_curve.json` (F[0..L]) from the per-depth result JSONs. F(d) is the shared performance curve both SISA and S3T compose with surviving-depth. |
| `s3t_deletion_time.py` | S3T deletion-time (Fig 9) | Times the S3T mask op (zero affected blocks' lora_B, ~ms) vs a real SISA shard retrain-from-checkpoint (GPU s). |
| `s3t_experiments.py` | S3T repro figures + report | Combines the simulator + measured F(d) + timings → Fig 6/7/9 analogues, Lemma-2 overlay, storage Table 3, RQ3 fold-in, and `reports/S3T_PAPER_REPRO_*.md`. `--src2` adds a second recipe's F(d) (armB) as the perf-vs-deletions contrast. Orchestrated by `submit_s3t_repro.sh` (armA: build depth dirs → F-eval %4 → deltime → finalize; `TRAIN_B=1` opt-in adds budget-B>1 training) and `submit_s3t_faithful.sh` (armB F(d) contrast + report regen with the RQ3/Lemma-2/Fig-9/storage sections). |
| `test_lazy_adapters.py` | Lazy adapter-cache regression | CPU micro-test (tiny random Llama + 3 saved shard dirs): lazy(cache_cap=2) forward logits BIT-EQUAL to an eager reference incl. a forced evict→reload cycle; resident count ≤ cap; active adapter never evicted; missing shard raises. Run before any `--lazy_adapter_cache` SLURM job: `python test_lazy_adapters.py`. |
| `submit_k200_routed.sh` | k=200 per-author task vectors + oracle routing driver (SLURM) | `bash submit_k200_routed.sh [train\|eval\|all]` — train = 200-task GPU array %4 completing the e25 per-author pool (`_k200_r32_e25_lr1e4`, self-skips existing shards; frozen recipe + `--epochs 25 --k 200`); eval = 8-task GPU array %4 (smoke): oracle-routed q2author OOD-aware full/del199 (`eval_routed_scaffold.py` on the PLAIN base, out `routed_oracle_{full,del199}.json`) + lexical `routed_key_exact[_no199]` (June-ladder comparable), each on the e5 AND e25 pools, all with `--lazy_adapter_cache 8`; `all` chains eval `--dependency=afterany` (afterok hangs forever on failure — kill_invalid_depend is off) with an in-task pool-completeness assert. `STUB=1` previews; `DEP=<jobid>` chains a solo eval. Copies the e5 smoke KS ref into the e25 results dir first. |
| `submit_e5_reroute.sh` | E5 — the trivial reroute-only "unlearning method" (SLURM) | `bash submit_e5_reroute.sh [eval\|mia\|all]` (`STUB=1` previews, `PACK=n` packs arms per job). Four eval arms on the k=200 e25 pool, all scoring the SAME 400-question forget10 via `--forget_author_ids 180-199`: `full` (pre-deletion reference), `delete` (experts dropped, orphans serve base — the D1-style control), `reroute0`/`reroute42` (nothing deleted; orphans forced to one fixed survivor). If the reroute arms score like `delete`, the forget metric cannot see the substitution — that is the §4.10 result. `mia` runs `attack_mia.py` over the same compositions for the privacy column. Refuses to start without the KS reference rather than reporting NaN fq. Thread: `log/selector_audit/`. |
| `submit_finalize_selector.sh` | Reads what the GPU arms produced (SLURM, CPU) | `DEP=<jobids> bash submit_finalize_selector.sh` (`STUB=1` previews). The arms already survive a logoff; what did not exist was anything to READ them, so a campaign ended with npz on disk and no answers. Chained `afterany`, this runs `analyze_router_probe` (H18) and `analyze_sequential_deletion` on every behavioral matrix that landed — gold-form and name-stripped, all three k=200 pools — then `consolidate.py`. Deliberately **not** `set -e`: one missing input must not abort the remaining analyses. Each step's exit status is recorded and printed as `STEPS THAT FAILED`, and the job exits 0 so the report is always written with the failures listed inside it. |
| `submit_overnight_selector.sh` | The unattended selector_audit campaign (SLURM) | `WAIT=<running ids> bash submit_overnight_selector.sh` (`STUB=1` previews). Everything is sbatch, so the campaign survives the login node going away — scoring and consolidation are queued as jobs too, not left as interactive steps. Submits the three full-length (`QPA 20`) CSAR arms — name_stripped, indirect, and the `random` control — then a CPU consolidation chained `afterany` on those AND on `WAIT`. **afterany, never afterok**: kill_invalid_depend is off cluster-wide, so an afterok chain hangs PENDING forever on the first failure, whereas afterany means the report still runs and names what is missing. Every stage self-skips existing outputs, so re-running after a partial night is safe. Writes `reports/SELECTOR_AUDIT_OVERNIGHT.{md,json}`. |
| `submit_ood_gate.sh` | What the OOD oracle buys (SLURM) | `bash submit_ood_gate.sh` (`STUB=1` previews). Two arms on the k=200 e25 pool with forget10 deleted, identical but for `--ood_gate`: `oracle` (a `q2author` miss serves base+scaffold) vs `route` (the miss goes to the nearest surviving centroid). `model_utility` is a harmonic mean including real_authors and world_facts, which is where the damage lands. At k=10/1B the same delta is already on record as mu 0.556 vs 0.474. |
| `analyze_deletion_size.py` | Routing metrics vs DELETION SIZE (CPU) | Sweeps how many sources were deleted on the score matrices `analyze_router_shift.py --dump_npz` already wrote, so the whole ladder is free. Per rung: routing accuracy on RETAINED rows, orphan detection AUC (via `analyze_router_probe.probe_arrays`), **RDR**, attacker capture and orphan destination `n_eff`. Two traps it exists to avoid: (1) `is_forget` is recomputed at EVERY rung — a row is an orphan only if its own author was deleted, and holding forget10's labels fixed while the drop set shrinks is exactly what produced the bad published k=50 cells; (2) rungs past 20 are flagged `⚠` because the 800-row set covers authors 0–19 and 180–199 only, so extra deleted authors contribute no orphan rows and the orphan-side columns stop being about deletion size (RDR and routing accuracy stay meaningful). Routing is post-deletion throughout (argmax over survivors); d=0 is the RDR reference. `--npz_dir --sizes --delete_order --strategies --conditions`. **Result: on gold-form queries RDR is 0.0000 all the way to d=20 and retained routing is flat at ~0.97; name-stripped, RDR grows 0.0000 → 0.0286 → 0.0383 → 0.0925 and retained routing erodes 0.343 → 0.198. Deletion locality does not merely fail without the name — it degrades with how much you delete.** |
| `submit_plain_ft_baseline.sh` | Plain fine-tuned baseline for findings 4 and 5 (SLURM) | `bash submit_plain_ft_baseline.sh [smoke\|all]` (`STUB=1` previews, `SHARDS=n`, `CONDS=` overrides). Two arms sharded over `SHARDS` GPUs each: **ft** = `locuslab/tofu_ft_llama2-7b` (the official TOFU full fine-tune — no adapters, no router) and **base** = `meta-llama/Llama-2-7B-chat-hf`, which is REQUIRED not optional because `csar.classify` subtracts the base's answer to the same question. Runs `selector_audit/eval_plain_ft.py` over `original,name_stripped,para_stripped,name_injected,name_swapped`. Answers "how much of finding 4's collapse is the ROUTER needing the name vs the MODEL needing it" — without this the audit cannot say whether the failures belong to routing or to any TOFU-trained model. Site default `TOFU_ARRAY_CAP=6` is a pacing choice; the association allows **gres/gpu=16 per user**, so `SHARDS=8` × 2 arms saturates it. |
| `submit_routed_shift.sh` | The ROUTED half of the same comparison (SLURM) | `bash submit_routed_shift.sh [qa\|qb\|all]` (`STUB=1`, `SHARDS=n`, `STRATS=`, `ATTACKER=`). One array per query transform over the k=200 e25 pool with authors 180-199 deleted, via `dump_generations_routed.py --serve_rows shift800 --row_shard`. `qa` = `none,name_stripped,para_stripped`; `qb` = `name_injected,name_swapped`. Exists because every prior routed arm served ONLY the 400 orphans, so there was no retain-side cell to compare a routerless model against. Feature-space routers only, matching the CSAR arms. |
| `submit_csar_audit.sh` | CSAR pilot — what the routed system SAYS (SLURM) | `bash submit_csar_audit.sh [gen\|score\|all]` (`STUB=1` previews, `QPA=n` questions per deleted author, default 5; `STRATS=` overrides the router set). gen = `dump_generations_routed.py --strategies` on the k=200 pool with authors 180-199 deleted; score = `selector_audit/csar.py` plus a 300-record hand-labelling sample. `QPA` samples every deleted author — `--max_questions` would head-slice the first two, which at per-author granularity would measure two people. Feature-space routers only: the behavioral family runs every expert on every query and is impractical past ~50 sources. **`QT=none|name_stripped|indirect`** selects the served-query transform and suffixes the output, so the gold-form and name-free arms coexist. |
| `submit_selector_wave.sh` | selector_audit GPU wave — behavioral at k=200 + recipe ablation (SLURM) | `bash submit_selector_wave.sh [beh\|feat\|all]` (`STUB=1` previews, `PACK=n` arms per job — one per allocated GPU, which converts the MaxJobs=6 limit into the gres/gpu=16 one; `QUERIES=all` overrides the 400-forget + 400-retain sample). **beh** = `ppl activation_norm attn_norm` at k=200 on all three k=200 pools, with `--lazy_adapter_cache 8` and `--self_check 3` (N×k activations — 50 would be 10,000 lazy loads). Never run at this granularity before: at k=10 this family was the leakiest (AUC 0.41–0.63), and whether per-author units rescue it is the open half of E1's granularity finding. **feat** = the feature-space battery on the two k=200 pools that were never audited (r32 e5, r8 e5) — same granularity, different recipe and rank, which is what separates "granularity causes detectability" from "the e25 recipe does". |
| `test_routed_scaffold_merged.py` | Routed-scaffold merged-arm regression | CPU gate (stub model/tokenizer, no HF hub) for `eval_routed_scaffold.py --merged_label`: merged adapter serves every TOFU-author query incl. forget-shard authors (Fig-8 serving), OOD stays scaffold-only, legacy shard/delete routing byte-unchanged, merged+delete_shard raises, batched forward shape/loss. Run before any `--merged_label` SLURM job: `python test_routed_scaffold_merged.py`. |
| `spotcheck_eval_port.sh` | Shard spot-check | SLURM job: train retain90 + prepare_eval + eval merged/remerge/shard_9_only for Llama-3.2-1B; sanity-checks metric ranges. |
| `sample_generations.py` | Qualitative report examples | Greedy Q/A generations for one label per invocation. |
| `routing_audit_tofu.py` | E3 post-deletion routing audit (RAMoLE arm) | Routing-only (no LLM loaded): routes all 4000 TOFU questions under `stale`/`rebuilt`/`dropped`/`abstain`/`key` policies, reports orphan (forget-author) routing, retain selection-shift, and per-expert index displacement vs the `unlearn/{tag}` manifest. `rebuilt` caches the retain-only index to the distinct `_ex...` file (stale index hash-asserted unchanged). `dropped` (§9-D drop-an-expert condition) masks the manifest's affected experts to −inf before ranking; K(a) ⊆ affected makes the orig/affected orphan columns trivially 0, so read `dropped_extras` (orphan top-1 concentration over survivors + masked/unmasked top-1 sim ratio) and `selection_shift.embed_stale_vs_dropped` instead; raises if survivors < k. `abstain` (C1 fix arm) writes an `abstain` block (no route table): τ calibrated on RETAIN top-1 similarity percentiles (no forget data touches the threshold), abstaining forget orphans whose masked top-1 sim < τ → reports orphan→base vs retain false-abstain per percentile. Result 2026-07-07 REFUTED — orphan/retain sim distributions overlap (means 0.858/0.877); 90% orphan-abstain costs 58% retain false-abstain. **The encoder pin is part of the index cache name:** `encoder_pin:"base"` (config `configs/ramole_tofu_1b_basepin.json`) caches to `expert_index_n{n}_encbase[.._ex…].npy` — without the suffix a base-pinned run silently reloads the FT-built stale cache (`expert_index_n32.npy` was rebuilt in place with the FT retriever on 2026-06-29, so its bytes are ramoleft provenance). `python routing_audit_tofu.py --config configs/ramole_tofu_1b_basepin.json --tag forget10 --policies stale rebuilt dropped key --device cuda --out PATH`. |
| `test_routing_audit_tofu.py` | Routing-audit regression | CPU micro-test (tiny pool + synthetic manifest, reuses `test_ramole_tofu` fixtures): rebuilt-vs-stale divergence, untouched index rows bit-identical, key selection-shift == 0, JSON round-trip, the sorted-tuple-vs-rank top-1 regression, `_encbase` cache isolation (base-pin never touches the stale bytes), and dropped-policy invariants (no masked expert in top-k; survivors<k raises) on a self-consistent k=1 fixture. Run before any SLURM job: `python test_routing_audit_tofu.py`. |
| **Router-leak campaign** (`log/router_leak/`) | — | `routing_audit_tofu.py` gains: `--policies … tombstone` (identity-seal family, three provenance rungs — per-EXPERT stale key / per-AUTHOR sentinel centroid (from the shared Q pass) / NAME-string embedding (`router._extract_author_names`, fallback = author sentinel, counted) — `tombstone_analysis` reports catch/leak/retain-FPR + margins per rung); `--dump_sims` (per-query similarity sidecar `<out>.sims.npz` — forget/retain × stale/sentinel sims + row/author metadata; never mutates the aggregate JSON; makes threshold variants pure-CPU post-processing); `--centroid_mode` (+`--centroid_k --drop_shard --router_encoder --probe_manifest --no_holdout`) = the k=10 shard-CENTROID router audit — the SAME MiniLM router the R2 serving arms deploy — reporting full/sibling/tombstone-rung stats, Mode-B `c_probe` per rung (the H4 mediator), and the holdout10 deletion-disclosure AUC (`_auc`, rank-based) — **per RUNG since 2026-07-23**: `disclosure.auc_forget_vs_holdout` (shard) plus `..._author` / `..._name`, with `sims_{author,name}_sent_holdout` added to the dump (the recommended author rung and the privacy-cleanest name rung were previously unpriced for disclosure). `--config` is now optional (unused in centroid mode). |
| `analyze_orphan_destinations.py` | Orphan-destination concentration (task (a), CPU) | Reduces stored orphan-destination histograms to a concentration profile per (router, drop-set, pool): max_share / top3_share / normalized-entropy (survivor-normalized, cross-checked == stored) / **HHI / Gini / n_eff=1/HHI** + a monotonicity read across drop counts. No inference. Inputs: `--family_json` (k=10/k=200 router-family sweep, `orphan_capture.top1_hist`), `--enc_json`/`--centroid_json` (`sibling.sibling_hist`), **`--legonet_json`** (n=32 legonet/ramole routing audits, `dropped_extras.top1_hist`), and **`--sims_glob`** (router-family `.<strategy>.npz` sidecars → the **per-author landing-determinism** block: for each deleted author, the fraction of its ~20 orphan questions landing on ONE survivor — masks the dropped shard cols via the `get_author_shard` inverse; skips key_exact's score-less `match` matrix). `python analyze_orphan_destinations.py --family_json … --legonet_json … --sims_glob …/rl_family_*.npz --out_md … --out_csv …`. **Result: dense k=10 routers collapse orphans onto n_eff 1.4–2.1 magnet experts (shard-4 hub); semantic/generative spread over 5.5–7.7; the n=32 legonet/ramole pool is two-magnet (n_eff 6.1–6.7, FT retriever puts 50% on 2 experts).** |
| `groupb_realistic_router.py` | Group-B realistic selector (router-leak (b)) | `attach_realistic_router(model, hf_home, unit_to_authors, …)` monkey-patches a Group-B served model's `_route` (SIFT/ClAMU/MemSinks) to a MiniLM centroid router over the SURVIVING selection units, keeping `q2author` only for the OOD gate; records `route_stats`. `per_author_units` (SIFT/MemSinks) / `clamu_cluster_units` (feature clusters + min-member repr). Used by `orphan_route_groupb.py`. |
| `orphan_route_groupb.py` | Group-B leak-inheritance test ((b), CPU routing + GPU serve) | `--mode routing` (no LLM): build surviving-unit MiniLM centroids, route the 400 deleted-author questions, report destination histogram + concentration + confidence AUC + author-sentinel tombstone catch/FPR. `--mode serve` (GPU): serve `*_unlearn` under {oracle, realistic} selector on the 400 forget questions → forget answer-prob + ROUGE-vs-gold (leak) vs base-gen (confabulation) + route stats (the H-GB2 exact-subtraction-robustness test). `--method {sift,clamu,memsinks}`. Driver `submit_groupb.sh [routing\|serve\|all]` (STUB=1; MSINK=1 adds memsinks). CPU gate `test_groupb_router.py`. |
| `analyze_router_leak.py` | Router-leak CPU post-processing | Subcommands over the `.sims.npz` sidecars (no encoder rerun): `roc` = the H1/H2 threshold-detector family (global_top1 / per_expert z / top1−top2 margin / knn_density / tomb_expert|author|name) with an AUTHOR-level calibrate/eval split (query-level would leak identity), ROC-AUC + retain-FPR at 90% orphan catch; `coverage` = R6 registry name-coverage cells (original / paraphrased / name-stripped questions); `table` = markdown deletion-dial table over audit JSONs. `python analyze_router_leak.py roc --npz X.sims.npz [--legonet] --out J`. |
| `router_family_audit.py` | All-router leakage audit (Phase 3, GPU) | The §9-D drop-audit generalized to the WHOLE `router.py` strategy family on one pool: `--pool_dir --base_model --k --strategies key_exact,key_tfidf,centroid_sbert,centroid_sbert_q,centroid_lm,centroid_lm_last,ppl,activation_norm,attn_norm,logit_div --drop_sets "9;9,8;9,8,7,6" --queries {all,sample}` (sample = 400 forget + the `analyze_router_tofu` RandomState(42) 400-retain draw). Emits per strategy × drop set: orphan capture (top1/top3/entropy/hist), family-specific sibling-adequacy ratio (cosine masked/unmasked top-1; ppl unmasked/masked top-1 loss; norm/div score ratio), retain top-1 shift, key_exact no-match operating point, oracle by-construction block, plus `--dump_sims` npz sidecars (`<out>.<strategy>.npz`: scores [n_q,k] higher=routed, ppl NEGATED, logit_div per-drop `scores__d<ids>` recomputed over survivors with NaN dropped cols, key_exact `match`, question-only per-author `author_sent_scores` for feature-space strategies). `--self_check N` (default 50) asserts score-matrix argmax ≡ `router.route()` per strategy — never disable (but its cost is N×k adapter activations, so lower N at high k rather than dropping it). **`--lazy_adapter_cache N` (selector_audit, 2026-08-07)** lifts the k>50 behavioral refusal for `ppl`/`activation_norm`/`attn_norm` ONLY. The split is the access pattern, not resident bytes: `score_norm_ppl_family` loops shards OUTER so the whole run costs k loads, while `score_logit_div` loops query batches outer with every shard inner AND caches one logits tensor per shard (~50 GiB at k=200) — no cache size fixes that, so it stays refused with its own message. Hooks are now registered AFTER `set_adapter` in both norm paths, since a lazily-loaded adapter has no `lora_B` until then. Centroid caches go to `{pool}/centroids/rfa_*` only. `--stub` = offline synthetic pool (CPU gate path). CPU gate: `python test_router_family.py`. |
| `analyze_router_family.py` | All-router leak-table assembly (CPU) | router_leak Phase-3 consumer of the `submit_router_family.sh` sweep: family npz sidecars + aggregate JSONs (+ RouterLoRA/DBpedia/encoder audits, SepMLP leak npz) → unified leak table + H-ARCH/H-DIAL/H-POOL/H-TRAINED/H-DATASET/H-ENC/H-SEAL-GEN verdict JSONs/MD (`--family_json --family_npz --routerlora_json --dbpedia_json --enc_json --enc_roc_json --sepmlp_leak_npz --sepmlp_ref_npz --out_json --out_md [--force]`). `--enc_roc_json` = `analyze_router_leak.py roc` JSONs over the `rl_enc_*.sims.npz` sidecars (H-ENC confidence half; tomb_* rungs excluded). Driver `bash submit_router_family.sh [j1..j6|collect|all]` (`STUB=1` previews, incl. the collect stage; `all` = ONE serial GPU lane); `collect` runs the enc roc halves first, then the analyzer with `--force` (derived summaries refresh on re-collect). CPU gate: `python analyze_router_family.py --self_test` (+ `python test_router_family.py`). |
| `analyze_router_probe.py` | Router-side orphan probe, E1 (CPU) | Asks the question the confidence family cannot: does the selector's POST-DELETION score geometry still identify an orphan for a *learned* reader, with no deletion record consulted? Consumes the same FAMILY NPZ CONTRACT as `analyze_router_family.py`; deleted columns are removed before any feature is computed, features are permutation-invariant over survivor columns (sorted top-m, margins, row moments) so a probe fit on one deletion transfers to an unseen one, and the split is the same author-parity one (even fit / odd eval). Recomputes the confidence family + `tomb_author` sentinel on the identical eval half as comparators — those recomputations reproduce `rl_family_leak_table.md` on all 12 comparable cells, which is the faithfulness check. Controls: author-level label shuffle (chance) and an oracle refit WITH the dropped columns (ceiling). Also ranks the eval half's SOURCES by mean orphan probability = **score-access** deletion attribution (not the black-box §4.4 attack). `--family_npz GLOB --drop_set 180-199 --out_json --out_md`. **`--rung LABEL:GLOB:DROPSET` (repeatable, coarse→fine)** assembles the GRANULARITY LADDER: each rung carries its OWN drop set, because a ladder is only a ladder if the DELETION is held fixed while the unit changes (forget10 = shard 9 at k=10, shards 45–49 at k=50, authors 180–199 at k=200). Monotonicity is computed, not eyeballed, and a strategy already at ceiling is reported `saturated` rather than `monotone` so it cannot pad the claim. ⚠ the published k=50 cells are `d49`/`d49_48` (4 and 8 authors) while `is_forget` marks all 400 forget10 rows — those cells label 16 of 20 authors as orphans with their expert still present, and read 0.593 instead of 0.795. Produces `reports/granularity_ladder.{json,md}`. CPU gate: `python analyze_router_probe.py --self_test` (+ `python test_router_probe.py`). **Result: the probe's median lift over plain confidence is +0.001 at k=200 and +0.05/+0.08 at k=10 — the mechanism is confidence, not a learned residual. The real axis is granularity, and the ladder confirms it monotonically at CONSTANT deletion size: `centroid_sbert` 0.564 → 0.795 → 0.984 and `centroid_lm` 0.502 → 0.628 → 0.761 over k = 10/50/200, with attribution recall 0.300 → 0.700 → 1.000. So the published 0.57–0.61 confidence ceiling is a statement about COARSE units, not about selectors.** |
| `analyze_sequential_deletion.py` | Magnet saturation + RDR (CPU) | Deletes sources ONE AT A TIME from a FAMILY NPZ CONTRACT matrix and reports, per step, where the accumulated orphans land (busiest share, n_eff) and **RDR** — the fraction of RETAINED queries whose unit changed versus no deletion, orphans excluded from the denominator. Verdict (saturating / saturated / dispersing / flat) is computed, not eyeballed. Pair with `analyze_router_shift --dump_npz` to run it on perturbed queries. `--family_npz --delete_order --out_json --out_md`; gate `--self_test`. **Result: saturation is REFUTED at k=200 (busiest share 0.550 → 0.130 as deletions accumulate, since each deleted author has a different nearest survivor) except for `key_tfidf` on name-free descriptive queries, which collapses to 0.902 on the nameless unit 88. RDR is 0.000 gold-form but 0.092 name-stripped — locality is lexical too.** |
| `analyze_router_shift.py` | Routing + orphan detection under QUERY SHIFT (CPU) | The stress test for the granularity ladder. Re-runs the weights-free feature-space routers (`key_exact`, `key_tfidf`, `centroid_sbert` — none reads expert weights, so no GPU) on six query conditions: `original`, `paraphrase` (TOFU's own — a NULL by construction, since those paraphrases keep the author's name: coverage 0.900 vs 0.895), `name_stripped`, `indirect` (definite description from `selector_audit/csar.py`'s distinctive-fact index), `name_injected` (adversarial — true subject kept, attacker's name appended) and `name_swapped` (the name-following upper bound). Reports routing accuracy, attacker capture, and post-deletion detection AUC via `analyze_router_probe.probe_arrays`. Evaluated on the 800 rows TOFU ships paraphrases for, with the perturbation applied to BOTH classes — perturbing only orphans would let the detector learn 'perturbed ⇒ deleted'. ⚠ pick the attacker with care: author 0 is `key_exact`'s fallback shard and 18 authors have no extractable name; both make capture unreadable. `--k --drop_set --attacker_id --out_json --out_md`. CPU gate: `python analyze_router_shift.py --self_test`. **Result: strip the name and the ladder flattens (centroid_sbert 0.560 → 0.572 → 0.623 vs 0.624 → 0.744 → 0.991 gold-form); one injected name captures 97.7% of queries against key_exact.** |
| `collect_wave45.py` | router_leak Waves 4+5 autonomous collector (CPU) | Assembles the Wave-4/5 numbers with NO inference so the cluster itself can finish the campaign independently of any interactive session: Mode-B rho per router family recomputed against the SHARED router-independent `expert_max` ceiling (via `aggregate_rho.rho_table` + the same channel remap as `--ceiling_channel`), the ppl-native seal rho + abstain rate, per-RUNG deletion-disclosure AUCs, and composed-MIA AUCs on the embed-routed sibling/tombstone arms. Degrades gracefully — missing producers are listed under `missing`, never fatal. `python collect_wave45.py --rl_dir D --ent_out D --out_json J --out_md M`; run as a CPU SLURM job chained `afterany` every producer. |
| `aggregate_rho.py` | ρ-vs-R assembly (CPU) | Mechanical ρ = clip((post−floor)/(ceiling−floor),0,1) table from three `eval_entangled_probe` JSONs (`--ceiling --post --floor`) — closes the previously-manual report step so every probe arm (e.g. embed tombstone) gets an identical ρ table. **`--ceiling_channel expert_max`** (2026-07-23) takes the ceiling from a DIFFERENT channel and remaps its keys onto the post channel: a **router-independent** ceiling (max answer-prob over experts, no routing). Required for magnet routers — `activation_norm` misroutes even with `--drop_shard none`, collapsing ceiling≈floor so the ratio degenerates (read ρ=1.0 even at R=1, impossible for a single-owner fact); a shared ceiling also makes ρ comparable ACROSS router families. |
| `dump_generations_routed.py` | R3 sibling-content audit (GPU) | For every forget-shard question: greedy generations under own-expert / embed-routed sibling / base+scaffold arms (one pool load), three ROUGE-L axes (vs deleted gold = disclosure; vs base-gen = generic-vs-confabulation; vs the sibling shard's nearest-question gold = cross-author disclosure) + the confabulation rate (H5). `--max_questions` = smoke cap. **`--strategies s1,s2,…`** (router_leak Wave-2, 2026-07-23) = the PER-STRATEGY sibling-content audit: generate own+base+every survivor shard ONCE per orphan (cached), then for each `router.py` strategy pick its routed sibling (forget shard excluded, via `merge_lora._build_routed_model`) and read the ROUGE axes off the cache → `strategies[strat].{aggregates,confabulation_rate,sibling_shard_hist}`; omit the flag = the single-MiniLM default (unchanged). **Multi-unit deletion (selector_audit, 2026-08-07):** `--forget_author_ids '180-199'` deletes a set spanning many shards (refusing a set that STRADDLES a shard, since dropping it would take retained authors too), `--questions_per_author N` samples every deleted author instead of head-slicing the first two, and `--lazy_adapter_cache N` makes k=200 viable. Generation is now LAZY — own + base up front, then only the shards a router actually picks; the old eager sweep over every survivor was ~200 generations per question at k=200 and would thrash the adapter LRU. Records carry the raw question/gold/generations so `selector_audit/csar.py` reads the same text the ROUGE axes were computed from. **`--query_transform {none,name_stripped,indirect}` (2026-08-07)** rewrites the SERVED query — routing and generation both see it, and both `question` (original) and `question_served` are recorded. `none` leaves every existing arm byte-identical. The others exist because CSAR, like the H3 defence before it, was measured on gold-form questions that name their author in ~90% of rows; a harm measured only on queries naming the deleted person is worth as much as a defence measured that way. Shares its transforms with `analyze_router_shift.py`. **`--serve_rows shift800 --row_shard i/N` + `--query_transform {para_stripped,name_injected,name_swapped}` + `--attacker_id` (plain-FT baselines, 2026-08-17):** `shift800` replaces the forget-only row set with `analyze_router_shift.build_eval_rows`'s 800 rows (400 forget + **400 retain**) — deletion is unchanged, so a retain row is simply one whose own expert survives, and without it the routed side of a named/anonymised × retain/forget table does not exist (this script had only ever served orphans). `row_shard` is strided, not contiguous, so a dead shard removes a slice of both classes rather than one class entirely. `name_injected`/`name_swapped` SERVE finding 5's attacks instead of only routing them, which is the only criterion a routerless baseline can also be scored on; `--attacker_id` must match finding 5's (author 0) and REFUSES an attacker with no extractable name. `para_stripped` keys TOFU's paraphrase by original-question text so the `(q, author)` transform signature is unchanged. Records gain `is_forget`. CPU gate: `python test_dump_generations.py`. |
| `test_router_leak.py` | Router-leak regression | CPU gate: `_auc` midrank ties, `_author_sentinels`, `tombstone_analysis` rung math, `EmbedRoutedModel` policy decisions (full/sibling/tombstone/OOD/guard), `detector_scores` planted separation + operating point, `aggregate_rho.rho`. Run before any router-leak SLURM job: `python test_router_leak.py`. |
| `submit_router_leak.sh` | Router-leak driver (SLURM) | `bash submit_router_leak.sh [phase1\|phase2smoke\|phase2\|content\|collect]` — phase1 = routing-only audits (n=32 basepin/FT + k=10 centroid + f01/f05 deletion dial; plan-manifests generated in-job via `unlearn_legonet.py --plan`); phase2 = the embed-routed serving triple (embedrouted_full / _sibling_del9 / _tombstone_del9) + Mode-B tombstone worlds at extended tier; content = `dump_generations_routed.py` (CAP=n smoke); collect = CPU analyzers. `STUB=1` previews; self-skips existing outputs; `DEP=<jobid>` chains every submission `--dependency=afterany` (the global 4-GPU cap). |
| `analyze_router_tofu.py` | E2 alpha-weight diagnostics (RAMoLE arm, GPU) | Captures the RouterLoRA per-layer attention weights alpha (`RouterController.capture_alpha`, opt-in) over ONE teacher-forced forward per record — never `generate` (KV-cache l=1 breaks position pooling) — for the 400 forget10 questions + a RandomState(42) 400-question retain sample, routed by the model's own key-route. Reports normalized entropy H vs the uniform-1/m anchor, max-share, ideal-expert mass, per-layer/per-completion-decile profiles, sharpness↔(−ln ppl) Spearman, and forget-vs-retain group means; `--unlearn_tag forget10` probes the UNCHANGED router over the post-deletion pool. Shared stats lib = `ramole/analyze_router.py` (same JSON schema; its `--report` mode renders the cross-run markdown). `python analyze_router_tofu.py --config configs/ramole_tofu_1b.json [--unlearn_tag forget10] --device cuda --out PATH`. **Router-leak Phase-3 (`--dropped`, 2026-07-20):** the H-TRAINED drop-audit — routes each query via the EMBED route over the STALE n=32 index, then masks the unlearn manifest's `affected_adapters` out of the active set (softmax renormalizes over survivors; if the whole routed set was affected, falls back to top-`min(k, n_survivors)` survivors by the same index ranking, `fallback_used` counted); TWO captured forwards per query (unmasked + masked, each under the strict one-capture contract). Emits per-query arrays (`h_norm, max_share, top1_share, top1_share_full, …, n_active, is_forget`) + group means + `_auc` AUCs (h_norm as-is, max_share negated; m==1-collapse rows carry `h_norm=1.0` — the family analyzer re-filters on `n_active>1`). `--router_ckpt` serves the s43/s44 seed routers. Consumed by `analyze_router_family.py`. |
| `jd_collection.py` | JD at scale (100s–1000s of adapters) | `build`/`select`/`merge` subcommands. Reads adapter safetensors on CPU (no fp32 GPU cast), caches a compressed JD artifact, materializes a kept-subset merge as a PEFT adapter dir for `eval_tofu.py`. Core math in `jd_compress.py`. Run heavy builds on SLURM; artifacts to `/storage2`. |
| `submit_jd_sweep.sh` | JD selective-keep smoke sweep | One SLURM array (`%4` = ≤4 GPUs total) running the 4 JD labels (merged/remerge × full/diag) over the in-model-safe k∈{4,10,50} Llama-2-7B dirs; c=1 / rank (n/2)+7 (paper §6.5). Skips existing result JSONs. `STUB=1` prints without submitting. k=100/200 need the `jd_collection.py` mode-B path (memory wall). |
| `submit_jd_highk.sh` | JD selective-keep at k=100/200 (mode-B) | GPU build+materialize jobs (`jd_collection build --device cuda`, clustered c=7/10 per §10.1) → dependent eval array (`%4`) that evals each materialized adapter via `eval_tofu --preloaded_adapter`. k=200 uses the r8 shard set. Artifacts → `/storage2/jack/jd_collections`. `STUB=1` to preview. |
| `subspace_overlap.py` | Merge-mechanism Exp 1 (CPU, no GPU) | Pairwise cosine + principal angles + shared-subspace energy of a set of adapters' effective deltas `scaling·BA` (factored, via `jd_collection.build_collection_slots`), vs random-orthogonal / shuffled-factor / replicated nulls. Tests whether fact adapters collide in a shared subspace (they do: col(B) overlap ≫ null). `python subspace_overlap.py --adapters DIR... --rank R --out reports/*.json`. `SUBSPACE_THREADS=N` env raises the default 8-thread cap for large-n SLURM CPU runs (n=200 needs it). Guarded by `test_subspace_overlap.py`. |
| `submit_subspace_k200.sh` | Per-author similarity (k=200, CPU) | sbatch driver for `subspace_overlap.py` over ALL 200 `_k200_r32` per-author shards (numeric `sort -V` order so matrix index == author id). CPU-only but a compute-node job: ~50 GB fp32 adapters + ~103 GB fp64 null copies ⇒ `--mem=220G`, 32 threads, `--n_null 5`. `STUB=1` previews. |
| `author_similarity_report.py` | Per-author similarity report (CPU) | Post-processes a `subspace_overlap.py` JSON: top-N most/least similar author pairs (TOFU author names resolved offline from the cached `full` split), per-author row-mean cosine ranking, name-token overlap test (do name-sharing authors have more similar deltas? permutation p, seed 42), PIL heatmap PNG (no matplotlib in test-env), cross-run trend table vs earlier subspace JSONs (⚠ those are 1B collections — direction only). `python author_similarity_report.py --json reports/subspace_overlap_k200_r32.json --priors k4=... --out_md ... --heatmap ...`. Re-runnable from the JSON alone. |
| `plot_author_tsne.py` | LoRA parameter-space t-SNE figure (CPU, **base python**) | HydraLoRA-Fig-1(b) analog: t-SNE (`metric='precomputed'`, perplexity sweep 5–50, seed 42) over a `subspace_overlap.py` JSON's pairwise delta-cosine matrix (1−cos distances; k200 ⇒ one point per author). Colorings: forget-split nesting (retain / f10∖05 / f05∖01 / f01) + optional `--author_emb` (legonet MiniLM `author_emb.npy`) k-means semantic clusters. Sidecar JSON records silhouette **on the precomputed distances** (never the 2D coords) so the visual can't be over-read; coords CSV for re-plotting. Outputs → `reports/figures/lora_tsne/`. ⚠ base python only (matplotlib). Gate: `/home/jack/anaconda3/bin/python test_plot_author_tsne.py`. |
| `train_peft_shard.py` | peft_compose bake-off trainer (GPU) | One non-LoRA PEFT adapter per shard: `--method {prefix,vera,ia3,dora}` (PrefixTuning / VeRA shared-frozen-basis / IA³ / LoRA+DoRA), hyperparams from `configs/peft_bakeoff_1b.json` (method-standard lrs, deliberately NOT recipe-matched — recorded there). Reuses `train_lora_shard` data path + shard conventions; `--smoke` = 2-step pipeline gate (`shard_i_smoke/` dir). VeRA shards share `projection_prng_key` + `save_projection=True` so the frozen basis is identical across shards (compose precondition). Saves to `{slug}_peft_{method}_k10/shard_i/`. Thread: `log/peft_compose/`. |
| `submit_peft_bakeoff.sh` | peft_compose driver (SLURM) | `bash submit_peft_bakeoff.sh CONFIG [smoke\|train\|compose\|eval\|collect\|all]`. smoke = 4 per-method 2-step micro-trains; train = 40-task GPU array (%4, self-skips); compose = CPU task (VeRA/IA³ compositions + exact-delete asserts + KS-ref copy into each pool); eval = GPU array (%4) over a generated manifest (iso {0,5,9} probes, composed full/unlearn per method, routed_key_exact reference per pool, LoRA additive_mean anchor on the legacy 1B pool; per-task self-skip on existing --out). `all` chains train→compose→eval with afterok (⚠ #SBATCH directives must live in the header — appending them after executable lines silently no-ops). `STUB=1` previews. Prereq: `python test_compose_peft.py` green. |
| `test_compose_peft.py` | peft_compose regression | CPU micro-tests (tiny GQA llama): compose math (mean; O(1) exact-delete identity; geo sign-fallback; shared-key mismatch raises), VeRA/IA³ save→file-compose→reload with n-copy compose ≡ single, prefix N=1 wrapper ≡ peft's own prefix forward + exact drop-shard + generate, and the DoRA `add_weighted_adapter` probe (writes `reports/dora_merge_probe.json`; the submit script gates the dora arm on it). Run before any bake-off SLURM job: `python test_compose_peft.py`. |
| `compose_peft.py` | peft_compose VeRA/IA³ composer (CPU) | Pure file-space compose of per-shard adapters into ONE `--preloaded_adapter`-servable dir: vera = mean of `vera_lambda_*` in the SHARED frozen basis (all other keys assert-equal + copied — catches projection_prng_key mismatches); ia3 = mean of `ia3_l` gates (`--variant geo` = signed geometric mean, arith fallback where signs disagree, fallback count printed). Verifies the O(1) exact-deletion identity `(n·mean−x_drop)/(n−1) ≡ compose(all∖drop)` before writing (`--verify_drop`, assert <1e-6). Hard-fails on missing shard dirs. |
| `prefix_concat.py` | peft_compose prefix-arm serving (GPU) | `PrefixConcatModel`: frozen base + every shard's trained KV prefix CONCATENATED in the cache (attention routes implicitly; deletion = drop the shard's segment — byte-exact O(1)). Drop-in for the PeftModel in eval_tofu (RoutedModel contract: forward with .loss, greedy batch-1 generate, .config). Prefix cache rebuilt per call; real-token positions start after the concatenated prefix (same convention as peft's single-prefix serving). Loader `load_prefix_concat_model` hard-fails on missing shard dirs (no silent skips). Used via `eval_tofu --prefix_pool_dir`. |
| `submit_iso_merged.sh` | Merge-mechanism Exp 3 (GPU) | Per-adapter isolated-vs-merged recall on each shard's OWN authors. One task per `"<label>\t<eval_shard_id>"` line of `ISO_MANIFEST`; runs `eval_tofu.py --eval_shard_id` → `results/.../<label>__own<sid>.json`. Mirrors `submit_eval.sh` (1 GPU/task, sprint4 excluded, `STUB=1`). The Exp-2 λ-sweep reuses `submit_eval.sh` directly via its env hooks + a `merged_additive_s{λ}` manifest. |
| `merge_subset.py` | Merge-mechanism Exp 5 — N-merge interference (CPU) | `plan`/`merge`/`overlap` subcommands, config-driven (`configs/nmerge_interference_7b.json`). Merges arbitrary NESTED author subsets of the `_k200_r32` per-author LoRAs (perm = `RandomState(seed).permutation(199)`, author 199 held out, probes = perm[:5]) and **materializes each merge on CPU as a single PEFT adapter dir** for `eval_tofu --preloaded_adapter` — never pays the fp32 high-k eval law. `additive_mean` = true-scale 1/N factor-cat straight from safetensors (no base model; output config `use_rslora=false, lora_alpha=r` ⇒ scaling 1.0); `additive_sum` = the same factor-cat at weight 1.0 each (the ctv unit-sum condition; label `nmerge_sum_*` — submit_ctv symlinks its ctv labels onto these); N∈{128,200} compress to `--svd_rank` via factored QR+SVD (acceptance-validated vs exact at N=64). `dare_ties` = the in-model `merge_shards` path run on a CPU-loaded base (√r-inflated convention preserved; the r8 N=200 `--cross_check` reproduces the prior in-model mu 0.4201). **Centered merges (2026-07-15, `configs/nmerge_centered_7b.json` → `_nmerge_r32_centered`):** `centered_pool` (S = mean of ALL `pool_authors`, default 0..198; always svd-compressed — cat rank 32·(N+199); capped at `max_n` because subset→pool degenerates back to the mean) and `centered_lowrank --rho R` (S = per-slot rank-ρ SVD of the subset mean; ρ=0 ≡ naive sum, ρ≥cat rank ≡ mean) implement `M = ΣΔᵢ − (N−1)·S` — ⚠ the literal PATHS_FORWARD §6.1 formula (S = exact subset mean) is the algebraic identity ≡ `additive_mean` and is deliberately not a method (the CPU gate proves the identity). `merge_meta.json` gains `center`/`rho`/`pool_size`/`center_energy_*`. `overlap` = per-N subset col(B)/cosine/shared-energy stats (imports `subspace_overlap.py`, ≤48-adapter subsample above the cap). Labels `nmerge_{add,dare,cpool,cr{ρ},sum,sumisqrt,sumL{λ}}[_svd{R}]_N{n}_s{seed}`, `iso_a{author}`. **APA study additions (2026-07-28, `configs/nmerge_sum_expA_7b.json` → `_nmerge_r32_sum`):** `--lam` sets a GLOBAL per-adapter coefficient for `additive_sum` — omit/`1.0` = the literal unit sum (label token `sum`, unchanged), `isqrt` = 1/√N (token `sumisqrt`, the **matched-norm control**: the per-author deltas are near-orthogonal, mean \|cos\| 0.0009–0.0051, so ‖Σ_N‖ grows as √N and 1/√N holds the injected perturbation constant across N — without it, sum-vs-mean at N confounds the aggregation rule with delta magnitude), or a float (token `sumL{λ}`). A FIXED λ keeps drop-a-term exact. Config keys `methods.additive_sum.lam_values` + `lam_n_values` (per-λ ladder restriction; each rank-1024 merge costs 7.7 GiB). `--authors "180-199"` + `--label` materialize an EXPLICIT author set, bypassing the permutation (needed because it covers 0–198 only and `perm(42)[:20]` contains no author from 180–198, yet 180–199 are the only paraphrase-covered authors outside the retain90 oracle). `fixed_probe_authors` + `anchors.at_all_probes` pin one probe set across seeds (per-seed probes are disjoint, and `probe_authors` clamps to `min(n, n_probes)` so N=2 would emit only 2 probe rows). New `norms` subcommand = CPU-only delta-norm ladder (‖Σ‖_F, `kappa` = ‖Σ‖/√Σ‖Δᵢ‖² — 1 ⇒ orthogonal/√N growth — and `rel_pert` = ‖Σ‖_F/‖W₀‖_F), which turns "utility falls" into a magnitude law and puts the sum and mean ladders on one axis. The merge manifest gained a **6th column (lam)**; `submit_nmerge.sh` treats absent/`-` as unset, so pre-2026-07-28 manifests still work. CPU gates: `python test_merge_subset.py` and `python test_expa.py`. |
| `submit_nmerge.sh` | Exp-5/6 SLURM driver | `bash submit_nmerge.sh CONFIG [plan\|merge\|eval\|overlap\|collect\|all]`. merge = CPU array (no gres, 160G/32cpu — N=200 r32 factors + SVD workspace; dare cross-check adds the CPU-loaded 7B base; merge manifest is 5-column `method\tn\tseed\tsvd\trho` — the rho column is absent in pre-2026-07-15 manifests and safely skipped); eval = GPU array over `eval_manifest_nmerge.txt` lines `label\tadapter\tsid` (`BASE` ⇒ `eval_baseline.py --forget_shard_id sid`; else `eval_tofu --preloaded_adapter --eval_shard_id sid`; self-skips existing JSONs; `EVAL_TIME=01:30:00` — big-rank adapters slow the forward); overlap = 1 CPU task; `all` chains merge→eval with afterok (kill_invalid_depend is off — scancel the eval array yourself if merges fail). `STUB=1` previews; `MERGE_ARRAY`/`EVAL_ARRAY` select single tasks (micro-smoke). |
| `analyze_nmerge.py` | Exp-5/6 CSV assembly (CPU) | `--config C` → `reports/nmerge_mu.csv` (label×probe rows; `headline` = the perm[0] probe job so every curve point shares one retain split; label grammar incl. centered `cpool`/`cr{ρ}` → methods `centered_pool`/`centered_lowrank_r{ρ}`), `reports/nmerge_own_recall.csv` (per-probe forget_rouge/tr/ppl + iso reference + drop), `reports/nmerge_overlap.csv` (geometry joined with drops, the H3 table), `reports/nmerge_subset_mu.csv` (subset-conditioned retain from `{label}__subset.json` rows + `ft_r32_sub{n}`/`base_model_sub{n}` ceiling/floor anchors). Checks: real/world components must be IDENTICAL across a label's probe jobs (`--eval_shard_id` only remaps forget + the retain sample); flags NaN mu + retain_ppl explosions. |
| `eval_mmlu.py` | MMLU for a served adapter (GPU) | The "no adapter should affect this" channel for the APA study — TOFU's `real_rouge`/`world_rouge` are saturated (0.98/0.93 across the whole mean ladder) and cannot show graded damage. Imports `_mmlu_prompt`/`_pred_letter` verbatim from `legonet_lora/eval_utility.py` (whose own `mmlu_acc`/`_run_grouped` only serve `"base"`/`"legonet"` and cannot take a `--preloaded_adapter` dir) and builds the model with `eval_tofu.load_single_adapter`, so the MMLU number describes the same artifact as the mu number. Same seeded draw for every condition ⇒ paired tests. Records per-item `letter_probs`: argmax over 4 letters floors at 25%, so **accuracy alone cannot tell "degraded" from "degenerate constant-letter"** — read `pred_letter_entropy` (0.0 = always the same letter) and `mean_letter_entropy` alongside `acc`. `python eval_mmlu.py --model_name M --adapter DIR\|BASE --label L --n_items 2000 --out P.mmlu.json`. |
| `measure_expb_contrib.py` | Per-author SIGNED contribution decomposition (GPU) | The Exp-B/C diagnostic. At `svd_rank=None` a factor-cat merge keeps per-author block identity (`A_cat`/`B_cat` rows/cols `[r·i : r·(i+1)]` ↔ `merge_meta.json["authors"][i]`, PEFT scaling forced to 1.0), so hooking the SERVED model's LoRA modules and splitting `z = A_cat h` gives each author's signed contribution `cᵢ`, with all norms from one Gram `G = B_catᵀB_cat` per slot. Reports per-query-group per-author norms, `diffuseness` (max_share/top3/H_norm/HHI/**n_eff**, same vocabulary as `analyze_orphan_destinations.py`) and the **cancellation index** ‖Σcᵢ‖/Σ‖cᵢ‖ (~1/√n = mutually orthogonal ⇒ injected noise grows √N; ~1 ⇒ reinforcing). Groups cover owned/unowned × original/paraphrase plus `holdout10` and OOD. `--hidden {served,base}` gives both tiers: base-h ratio <2 ⇒ selectivity failure is a TRAINING artifact, ≥5 on base but <2 on served ⇒ an AGGREGATION artifact. **Asserts `Σᵢcᵢ` reconstructs the exact delta and aborts if not**; **refuses any SVD-compressed merge** (`_compress_factored` destroys block identity). Replaces `measure_key_firing.py` for this question — that one hooks a plain base model, reports unsigned norms, and globs `shard_<i>/` dirs. |
| `measure_adapter_selectivity.py` | Exp-B driver: paraphrase robustness + cross-author leakage (GPU) | One served model → every pool author × both question surfaces → raw per-question CSV + per-(author,surface) aggregates; contrasts are assembled offline so no condition knows about another. Reimplements no metric: calls `eval_tofu.get_rouge`/`get_answer_probability`/`get_truth_ratio_scores` through their `per_example` sinks and `question_key`. ROUGE-L is scored against the **original `answer` on both surfaces** (a verbatim-`answer` model scores only 0.380 against `paraphrased_answer`, so the paraphrased gold builds in a ~2.5× penalty before any memorization effect); `paraphrased_answer` is kept as a secondary column. `--build_refs` writes one KS reference per (author, surface) from the retain-only oracle — the repo's cached `retain_tr_scores.npy` is author-199-only and `--eval_shard_id` never re-derives it — and every scoring run verifies the reference's row-set hash and **refuses a mismatch**. Refuses authors outside 0–19/180–199 unless `--allow_uncovered`. |
| `test_cluster_env.py` | Site-abstraction CPU gate | Pins the two properties that make `cluster_env.<site>.sh` safe: (1) on `sprint` the shim exports the six legacy variables with **byte-identical** values to the pre-refactor hardcoded ones, so the 57 unported drivers cannot regress; (2) `tofu_sbatch_resources` honours the site memory policy — sprint emits `--mem`, cispa **drops it** (emitting one there fails the whole campaign at the first `sbatch`). Also covers `gpus=0 ⇒ no --gres`, `${VAR}` config expansion, and that every shipped `nmerge_*.json` still resolves. No SLURM, no network, no GPU. |
| `test_fixtures.py` | Portable test fixtures (import-only) | `resolve_tokenizer(model_id, extra_dirs=…)` finds a real tokenizer offline-first via `$HF_HOME/hub/models--…/snapshots/*` instead of a hardcoded `/storage2` snapshot dir, and `require_tofu(configs)` the dataset splits. Both raise `FixtureMissing` with an actionable message so a gate on a fresh machine **skips loudly** rather than dying on `FileNotFoundError` — or, worse, quietly stopping testing. |
| `tofu_env.py` | Shell→Python site settings (import-only) | `ensure_site_env()` shells out to `slurm_nodes.sh` once and imports the `TOFU_*`/`HF_HOME` exports into `os.environ`, filling only keys that are absent so an explicit override always wins. Exists so `cluster_env.<site>.sh` stays the single source of truth — re-declaring those values in Python would create a second one that silently drifts. |
| `test_expa.py` | APA-study CPU gate | Run before any APA SLURM job. Pins: the `nmerge_sum_*` label grammar (explicitly that it does NOT parse to `centered_lowrank_rm`, which a regex-only patch produces); `do_plan`/spec default-identity for configs without the new keys; `eval_tofu`'s `per_example` sinks bit-identical at `None` and `question_key` actually changing the measurement; the perturbed-coverage and `holdout10`-disjointness data premises; and that the block decomposition is exhaustive with fixed-weight drop-a-term exact. |
| `plot_nmerge.py` | Exp-5 figures (CPU, **base python**) | Reads only the analyze CSVs → `reports/figures/nmerge/fig{1..5}_*.png/pdf` (mu-vs-N with anchors + prior k-scaling context, per-probe recall trajectories, 3×3 mu-component small multiples, geometry + drop-vs-overlap scatter, subset-conditioned retain utility vs N). ⚠ matplotlib lives ONLY in base anaconda (`/home/jack/anaconda3/bin/python`), NOT test-env — run this one script with base python. Safe on partial results. |
| `analyze_merge_mechanism.py` | Merge-mechanism CSV assembly (CPU) | Post-array: `lambda` subcmd → `reports/lambda_sweep_1b.csv` (mu/forget vs λ from `merged/remerge_additive_s{λ}.json`); `iso` subcmd → `reports/iso_merged_drop.csv` (per-shard isolated−merged forget_rouge drop from `*__own<sid>.json`). Findings → `reports/MERGE_MECHANISM_REPORT_2026-06-29.md`. |
| `measure_key_firing.py` | Merge-mechanism Exp 7 — key-firing / lazy-read-keys (§6.2, GPU) | Functional selectivity of per-author adapters over BASE-model hidden states: forward pre-hooks on the 192 target Linears capture h once per prompt (one forward serves ALL adapters), then per adapter ‖Aᵢh‖ + ‖sᵢBᵢAᵢh‖ via the factored Gram trick ‖Bz‖² = zᵀ(s²BᵀB)z — never materializes d_out. Prompts: 5 seeded TOFU questions/author × 200 + world_facts/real_authors/Alpaca OOD (×100), eval-prompt convention (`Question: {q}\nAnswer:`, no chat template). Output `--out` JSON (per-adapter on/off/OOD stats + ratio distributions + the pre-registered LAZY/SELECTIVE gate verdict) + sidecar `_matrices.npz` (adapter×group matrices, 8 agg keys: BA/A × meantok/lasttok × attn/mlp/layer-terciles). Gate: median on/off ‖sBAh‖ ratio <2.0 ⇒ LAZY (negative anchoring §6.3 GO), ≥5.0 ⇒ SELECTIVE. `python measure_key_firing.py --shards_dir D --out reports/key_firing_e5.json --device cuda`. |
| `test_measure_key_firing.py` | Key-firing regression | CPU micro-test (tiny random Llama + 2 saved shard adapters, stub char tokenizer): hook/Gram-trick matrices == a direct dense per-prompt computation across all 8 agg keys (incl. padding/batching), bit-equal determinism, summarize() on/off wiring + all three gate verdicts. Run before any key-firing SLURM job: `python test_measure_key_firing.py`. |
| `test_train_anchor.py` | §6.3 anchor-penalty regression | CPU micro-tests (tiny random Llama, one rslora adapter, dropout 0): `anchor_penalty` (lora_B hooks + scaling-once + fp32 + pad mask) == dense closed form; penalty gradients reach lora_A AND lora_B; `apply_anchor_to_loss` λ=0 identity passthrough (the flag-free frozen-recipe invariant), λ>0 = parent + λ·penalty, deterministic batch cycling, return_outputs contract. Run before any anchored SLURM job: `python test_train_anchor.py`. |
| `submit_anchor_pilot.sh` | §6.3 λ-pilot driver (SLURM) | `bash submit_anchor_pilot.sh [train\|keyfire\|iso]` — train = 15-task GPU array (probe authors {82,15,111,177,76} × λ∈{1,10,100}, e25 recipe + `--anchor_lambda`, self-skips); keyfire = 3 × 1-GPU selectivity re-measure per λ pool (`reports/key_firing_e25_anch{λ}.json`); iso = 15-task own-author recall array (`eval_tofu --preloaded_adapter --eval_shard_id`, copies the e5 KS ref in first). `STUB=1` previews; every stage is GPU — cap-check `squeue` first. Pre-registered λ rule + verdicts: `log/merge_mechanism/2026-07-15_negative-anchor-design.md`. |
| `submit_key_firing.sh` | Key-firing driver (SLURM) | `bash submit_key_firing.sh [e5\|e25\|both]` — one 1-GPU job per adapter set (e5 = 200 per-author r32, e25 = the 20 strong subset(42) adapters), 48G/03:00:00, self-skips on existing `reports/key_firing_<arm>.json`, `STUB=1` previews. ⚠ 1 GPU each — check `squeue` against the global 4-GPU cap before submitting. |
| `skill_data.py` | Part B (facts-vs-skills) data | Super-NaturalInstructions loader (`Muennighoff/natural-instructions` per-task `test/*.jsonl`, input-only) + deterministic train/held-out split; `facts_heldout` = TOFU `full`-split Q&As for an author set. Guarded by `test_skill_data.py`. |
| `train_skill_lora.py` | Part B skill adapter (GPU) | One isolated LoRA per SuperNI task, config-driven (`configs/skills_superni_1b.json`), **recipe identical to `train_lora_shard.py`** (r16/α32/rslora/6-mod) so facts vs skills is controlled. Saves held-out to `a{j}/skill_meta.json`. |
| `eval_skill.py` | Part B uniform NLL (GPU) | For N isolated adapters: mean answer-token NLL (reuses `eval_tofu._answer_avg_loss`) under base / isolated / merged (all-N linear 1/N). `--domain {skills,facts}`; one call = full sweep → `reports/skill_nll_{domain}.json`. |
| `analyze_skill_vs_facts.py` | Part B contrast (CPU) | Normalized retention R=(merged−base)/(isolated−base) per adapter; Mann-Whitney (skills>facts) → `reports/facts_vs_skills_retention.csv`. **Result CORRECTED:** the rsLoRA "facts collapse" (U=400,p=3.4e-8) was a √r merge-inflation artifact; a **true-mean (`--no_rslora`) merge → gap vanishes (p=0.68)**, facts collide only modestly more in weight space. See `reports/FACTS_VS_SKILLS_REPORT_2026-07-01.md` §CORRECTION. Use the `_nr` configs/dirs for the true-mean run. |
| `submit_skills.sh` | Part B driver (SLURM) | `train` (N-adapter array) / `eval` (1-GPU: eval_skill skills+facts + analyze). Facts arm = `train_lora_shard.py --k 20` → `..._facts_n20`. Config `configs/skills_superni_1b.json`. `STUB=1` preview. |
| `train_scaffold.py` | routing+scaffold method: public scaffold (GPU) | Trains one LoRA on ~N public Alpaca QA pairs (`skill_data.load_alpaca`) — shared, never deleted, generic QA competence. Non-rslora defaults so scaffold+expert compose as an additive sum → `..._scaffold_alpaca2k`. |
| `make_scaffolded_base.py` | bake scaffold into base (GPU) | `merge_and_unload` the scaffold into the base → a full model on disk; eval routing against it serves base+scaffold+routed-expert with no eval_tofu change. → `/storage2/.../*_scaffolded_alpaca2k`. |
| `eval_routed_scaffold.py` | routing+scaffold eval, OOD-aware (GPU) | Per-query: TOFU-author (exact `q2author` lookup) → shard expert; OOD (real/world) → scaffold-only (adapters disabled) — fixes the bug where routing a TOFU expert onto general-knowledge queries corrupts them. `--delete_shard j` = exact O(1) deletion demo (its authors serve base+scaffold). `--merged_label merged_*\|remerge_*` = the scaffold×composition control (arm B): build that merge over the loaded experts via `activate_label` and serve ALL TOFU-author queries with the ONE merged adapter (OOD stays scaffold-only; `remerge_*` = merged-deployment deletion, forgotten authors get the retain merge Fig-8-style; mutually exclusive with `--delete_shard`; result row/JSON label = `scafmerged_<label>`; driver `bash submit_scafmerge_armB.sh [smoke\|extended]`, `STUB=1` previews, `SCAFMERGE_LABELS="..."` overrides the label set — string-resolving merge labels only, no data-required/ensemble). CPU gate: `python test_routed_scaffold_merged.py`. **Result: OOD-aware routed+scaffold mu 0.556 > full-FT 0.530** (buggy key_exact version was 0.474). `reports/ROUTING_SCAFFOLD_REPRO_2026-07-01.md`. **Router-leak R2 arm:** `--embed_route {sibling,tombstone}` (+`--router_encoder`) serves TOFU-author queries by nearest shard CENTROID (`EmbedRoutedModel`/`build_shard_centroids`, MiniLM; OOD stays oracle-gated) — with `--delete_shard`: sibling = deleted centroid removed (orphans leak to the nearest surviving sibling), tombstone = centroid kept as identity sentinel (top-1 hits serve base+scaffold); without = the embed-full baseline. Labels `embedrouted_full`/`embedrouted_{sibling,tombstone}_del{j}`; route_stats gains `route_mismatch`. **`--embed_route tombstone_author --tombstone_tau T` (2026-07-24, H3 closer):** survivor-only routing gated by a per-deleted-AUTHOR identity sentinel — abstain to base when (best deleted-author-sentinel sim − best surviving-centroid sim) > T; sentinels are the deleted shard's per-author mean question embeddings (recomputable from the deletion request), T calibrated on RETAIN margins (0.1944 = the k=10 MiniLM 90%-catch / 0.11%-retain-FPR point). This is the thresholded author rung the Phase-1 ROC predicted would drive served retain-Δmu → ~0, vs the shard-argmax rung's −0.0061. Routing depends only on question embeddings — route stats must be bit-identical across expert pools (the R4 assert). Mutually exclusive with `--merged_label`. **`--ood_gate {oracle,route}` (2026-08-07):** the `q2author` lookup deciding *is this query about one of my sources* is an ORACLE — `oracle` (default, unchanged) serves base+scaffold on a miss, which is what every published number assumes; `route` hands the miss to the nearest SURVIVING centroid, which is what happens without that oracle. Source routing stays exact in both, so the delta prices this oracle alone (`stats['ood_routed']`, label `routed_oracle_ungated_ood`). Driver `submit_ood_gate.sh`. **Multi-unit deletion + the E5 reroute arm (selector_audit, 2026-08-07):** `--delete_shards '180-199'` deletes a SET of units (20 per-author units at k=200 = one forget10; `--delete_shard` took one int), and `--reroute_to j` deletes NOTHING — the deleted authors' queries are answered by fixed surviving expert j instead of falling through to base+scaffold. That arm exists to ask whether TOFU's forget metric can tell 'the source is gone' from 'a stranger answers for it'. `route_stats` gains `rerouted`, and the run ASSERTS the served route matches the requested policy (20×20 orphan queries on the deletion path) before any metric is read — a plausible-but-wrong route is the failure mode these arms are most exposed to and no metric would flag it. Driver `submit_e5_reroute.sh`. CPU gates: `python test_routed_scaffold_merged.py`, `python test_router_leak.py`. |

## Architecture

### Shard–adapter model

TOFU has 200 fictional authors. With `k=10`, each shard covers 20 authors (20 Q&A pairs each = 400 samples). `shard_utils.get_author_shard(k, shard_id)` is the single source of truth for which authors belong to which shard. Shard 9 (`shard_id = k-1`) is always the forget shard — it aligns exactly with TOFU's `forget10` split.

Each shard's LoRA adapter is saved to `checkpoints/{model_slug}/shard_{i}/` and loaded into a single PeftModel at eval time:

```python
model = PeftModel.from_pretrained(base, "shard_0", adapter_name="shard_0")
for i in range(1, k):
    model.load_adapter(f"shard_{i}", adapter_name=f"shard_{i}")
```

### Eval label dispatch (`merge_lora.py`)

`activate_label(model, k, forget_id, label, ...)` is the single dispatch point for all eval modes. It returns either a **string** (PEFT adapter name) or a **RoutedModel** instance (for routing labels). Callers must branch on `isinstance(result, str)`.

Label categories and what they create:

| Prefix | Example | What happens |
|---|---|---|
| `shard_{i}_only` | `shard_9_only` | Returns existing adapter, no merge |
| `merged_{method}` | `merged_dare_ties` | Merges all k shards with PEFT `add_weighted_adapter` |
| `remerge_{method}` | `remerge_dare_ties` | Same but excludes `forget_id` shard |
| `merged_jd_{full\|diag}` | `merged_jd_full_c4_r16` | Joint-Diagonalization combine of all k shards; `_c{N}` clusters, `_r{R}` JD rank |
| `remerge_jd_{full\|diag}` | `remerge_jd_full_c4` | JD combine of all-but-forget shard = selective-keep unlearning |
| `subtract_linear` | — | Task-arithmetic subtraction via `cat` merge |
| `subtract_orth` | — | Project merged-all delta out of forget shard's LoRA subspaces (`merge_extra.subtract_orth_adapters`) |
| `tree_root_{method}` | `tree_root_linear` | Binary tree merge; O(log k) re-merge on unlearn |
| `tree_remerge_{method}` | `tree_remerge_linear` | Tree root excluding forget shard |
| `routed_{strategy}` | `routed_centroid_sbert` | Returns `RoutedModel` wrapper (see below) |
| `routed_{strategy}_no{i}` | `routed_key_exact_no9` | Same but shard `i` excluded from routing |
| `ensemble_{probs\|logits}` | `ensemble_probs` | Returns `EnsembleModel` (`ensemble.py`): SISA/S3T prediction-level ensemble — averages constituents' per-token distributions at inference (probs = arithmetic mean of softmax, logits = mean logits). Opt-in only (≈n× eval cost), never in default manifests. Needs `output_dir=` to verify constituents against disk. |
| `ensemble_{mode}_no{i}` | `ensemble_probs_no9` | Same but shard `i` dropped from the ensemble (SISA drop-shard unlearning) |
| `legonet_full` / `legonet_unlearn` | — | LegoNet arm (see Comparison Tracks). **Not** routed through `activate_label` — `eval_tofu.py --legonet_config C [--legonet_unlearn_tag TAG]` builds `legonet_model.LegoNetRoutedModel` directly. `legonet_full` = all authors; `legonet_unlearn` = forget10 removed (the unlearning condition). |

All adapters created by `activate_label` are lazy — they are built in PEFT memory on first use, not saved to disk (except centroid `.npy` files).

### Routing mode (`router.py`)

Nine routing strategies in three categories:

- **Lexical**: `key_exact` (author-name substring), `key_tfidf` (TF-IDF cosine)
- **Centroid**: `centroid_lm` (LM mean-pool), `centroid_lm_last` (LM last-token), `centroid_sbert` (sentence-transformers), `ppl` (min perplexity)
- **Adapter-intrinsic**: `activation_norm` (lora_B delta norm), `logit_div` (logit divergence from mean), `attn_norm` (attention layers only)

`RoutedModel` wraps the PeftModel and is a drop-in for it in eval: `forward()` routes per sample, `generate()` routes on the first token's shard. Centroid `.npy` files cache to `{output_dir}/centroids/{strategy}/shard_{i}.npy`.

### Merge methods (`merge_lora.py` registry, implementations split with `merge_extra.py`)

`MERGE_METHODS` dict keys: PEFT-native `linear`, `dare_linear`, `ties`, `dare_ties`,
`magnitude_prune`, `cat`, `ties_svd`, `dare_ties_svd`; custom (in `merge_extra.py` unless noted)
`regmean` (in `merge_lora.py`), `weighted_avg_ab`, `weighted_ba` (in `merge_lora.py`),
`della_linear`, `della_ties` (MAGPRUNE), `breadcrumbs`, `knots_ties` (true shared-basis KnOTS —
distinct from PEFT's merge-then-compress `ties_svd`), `tsv` (TSV-M whitening), `slerp`
(pairwise: tree merges or k=2 only), `fisher`, `lorahub`, `jd_full`/`jd_diag` (Joint
Diagonalization, *Compress-then-Serve*), `additive`/`additive_mean` (exact-unlearning research doc).

**Additive (`merge_extra.additive_merge_adapters`).** The literal `W + Σᵢ scalingᵢ·BᵢAᵢ`:
a TRUE-scale SUM of effective deltas (weight 1.0 each, not 1/n), at FULL rank (`Σᵢ rᵢ`, a
`cat` scaffold), with NO SVD compression — so `merged_additive` − forget-term ≡
`remerge_additive` exactly (`test_merge_extra.test_additive`). It is the *corrected*
`linear`/`cat`: those apply `sqrt(wᵢ·scalingᵢ)` per factor, double-counting the rslora
`sqrt(r)` and exploding (ppl 10³–10⁷, mu≈0); `additive` adds each effective delta once.
Under standard LoRA (`use_rslora=False`) scaling = `α/r`, so it is exactly the doc's
`W + Σ(α/r)BA`. **The raw weight-1.0 sum overshoots and collapses the model at k=10 (ppl ~10⁴,
mu 0; norm grows ~k× from shared-direction overlap, NOT a √r bug).** Compose with a global
coefficient λ via the `_s{λ}` label suffix (`merged_additive_s0.2`, parsed by
`_split_scale_suffix` → `weights=[λ]·n`): a FIXED λ keeps exact drop-a-term (unlearn = drop
λ·dW_j). λ=1/k (`additive_mean`) already recovers mu≈0.48 (≈dare_ties); sweep λ for the optimum.
This is the method under test for the additive-shard exact-unlearning thread
(`reports/ADDITIVE_SHARD_REPORT_*`); kept out of the default manifests for Phase-A
comparability — eval explicit `merged_additive`/`remerge_additive` labels.

**Joint Diagonalization (`jd_compress.py` library + `jd_collection.py` scale path).** JD
compresses the *collection* into a shared basis `U,V` (per cluster, per module) plus a tiny
per-adapter `Σ_i`, then combines the kept adapters into one delta `Σ_j U_j(Σ_i w_i Σ_i)V_jᵀ`
(rank-compressed to the scaffold). Unlike every other registry method it does not fuse adapters
during compression, so "keep a subset" / deletion is an O(1) `Σ_i` add/drop. Labels take
`_c{N}` (cluster count) and `_r{R}` (JD rank) suffixes, parsed by `_split_jd_suffix`
(`merged_jd_full_c4_r16`). `jd_full`=full `Σ`, `jd_diag`=diagonal `Σ`. In `merge_lora`/`eval_tofu`
JD is recomputed in-memory on the given shards (`merge_extra.jd_merge_adapters`); for
hundreds/thousands of adapters use `jd_collection.py` (build/select/merge subcommands) which reads
adapter safetensors on CPU — no per-adapter fp32 GPU cast — caches a compressed artifact under
`/storage2`, and materializes a kept-subset merge as a normal PEFT adapter dir for eval. JD is
**not** supported for tree merging (pairwise nodes defeat collection-wide compression).

Manifest tiers: `DEFAULT_MERGE_METHODS` (Phase A set, unchanged for comparability) +
`EXPERIMENTAL_MERGE_METHODS` (`della_linear`, `della_ties`, `breadcrumbs`, `knots_ties`, `tsv`,
`jd_full`, `jd_diag`)
both go in smoke and extended manifests; `DATA_REQUIRED_METHODS` (`regmean`, `fisher`,
`lorahub`) need a dataloader — `eval_tofu.py` builds one automatically
(`build_merge_dataloader`: included shards' training text; `remerge_*` excludes the forget shard
so no forget data touches the merge) — and are extended-manifest only. Methods can take a
density suffix: `merged_ties_d0.5`. Stochastic methods (`della_*`, `lorahub`) take `seed`
(default 0).

Tree merging uses `tree_utils.py` for the binary range-split structure. Data-required methods
and `weighted_avg_ab`/`weighted_ba` are unsupported for tree merging; all other custom methods
(incl. `slerp`, which is tree's natural fit) work per node.

### Eval metrics (`eval_tofu.py`)

Metrics are ported to match **open-unlearning** (locuslab) exactly — guarded by `test_ou_equivalence.py`.
Families: **perplexity** (diagnostic), **ROUGE-L recall**, **answer probability** (`P(a|q)^(1/|a|)`;
real_authors/world_facts use `probability_w_options` = correct/(correct+Σwrong) over the `*_perturbed`
splits), **truth ratio** (per-sample `tr = wrong/correct`, wrong = geomean of the perturbed probs).
Aggregation: `forget_truth_ratio` = mean(min(tr, 1/tr)) ∈ [0,1] (→1 = more forgetting); the
retain/real/world utility component = mean(max(0, 1-tr)). `model_utility` = `scipy.stats.hmean` of 9
values (retain + real_authors + world_facts × prob/rouge/truth). **`forget_quality`** (alias `ks_pval`)
= KS p-value of the forget truth-ratio distribution vs the **retain90 oracle** (`retain_tr_scores.npy`);
**higher** p = indistinguishable from a model that never trained on forget = better unlearning.

Smoke caps: ROUGE≤50, retain≤80, truth≤30 samples. Extended caps: ROUGE≤200, retain≤400, truth≤120.

**Raw per-question access (2026-07-28, strictly additive — every default is bit-identical, pinned
by `test_expa.py` and `test_ou_equivalence.py`).** `get_rouge`, `get_answer_probability` and
`get_truth_ratio_scores` each take an optional `per_example=<list>` sink; the caller gets one dict
per scored item (including the **generation**, which `get_rouge` always computed and threw away on
the mean). `get_truth_ratio_scores` additionally takes `question_key` — the only metric that reads
the question out of the row, so it is the one place the paraphrase surface hooks in; ROUGE and
probability already take explicit question lists. Two traps the sinks are designed around:
`get_answer_probability` and `get_prob_w_options` `continue` past rows with no answer tokens, so
**list position ≠ dataset index** — the sink records the source index explicitly and marks skipped
rows `kept: False`. `evaluate_model` itself is unchanged and still writes aggregates only; the
per-question consumers are `measure_adapter_selectivity.py` and the APA collectors.

### Checkpoint layout

```
checkpoints/
  {model_slug}/
    shard_{0..k-1}/          ← LoRA adapter weights
    retain90/                ← retain90 oracle adapter (authors 0-179; forget_quality KS reference)
    results/
      smoke/
        eval_manifest_smoke.txt   ← one label per line
        retain_tr_scores.npy      ← retain90 forget-set truth ratios (forget_quality KS reference)
        {label}.json              ← per-adapter metric dict
      extended/               ← same structure, larger caps
    centroids/
      centroid_sbert/
        shard_{i}.npy         ← cached embedding centroids
```

`model_slug(model_name)` in `model_paths.py` converts `org/ModelName` → `ModelName`.

Directory suffix conventions under `checkpoints/`:

| Pattern | Meaning |
|---|---|
| `{slug}` | SISA-LoRA, k=10 shards (`shard_0..9`) — the main method |
| `{slug}_k4` | SISA-LoRA, k=4 variant |
| `{slug}_k{N}_r{R}_e{E}_lr{LR}` | shard star grid / k-scaling variants (e.g. `_k20_r32_e5_lr1e4`, `_k200_r1_e5_lr1e4`); k ∈ {50,100,200} from `submit_scale_grid.sh` |
| `{slug}_r{rank}_e{epochs}` | rank / epoch ablation variants |
| `{slug}_ft` | full-data k=1 LoRA fine-tune (retain-all reference) |
| `{slug}_ft_unlearn_{ga,gd,kl,idk}` | gradient-baseline unlearning (shard_0 only) |
| `{slug}_s3t_m{M}_L{L}_{armA,armB}` | S³T track (`train_s3t_shard.py`, configs in `configs/s3t_arm*.json`): m shards × L sequential stages with layer-disjoint LoRA; `shard_{i}/stages/stage_{j}/` holds per-stage snapshots. `_del` suffix = deletion state (symlink dir: forget shard → its pre-forget-slice snapshot); `_t0`/`_t2` = retention-curve truncations. Submitted end-to-end by `submit_s3t_overnight.sh` (gates → trains → evals → `s3t_pick_winner.py` → extended). |
| `{slug}_s3t_..._depth{d}` | symlink eval dir for the S³T paper repro: `shard_i -> ..._armA/shard_i/stages/stage_{d-1}` so `F(d)` (ensemble utility when every shard retains d slices) is eval'd with no new training. Built by `s3t_measure_F.py build`. `shard_{i}/seq_{b}/` (under a `_B{B}` dir) holds multi-sequence budget-B>1 training (opt-in). |
| `{slug}_legonet_n{N}_k{K}` | LegoNet arm (`prepare_legonet.py`/`train_legonet_adapter.py`/`unlearn_legonet.py`, config `configs/legonet_tofu.json`): frozen k-means author keys + n cluster adapters under `legonet/`; `legonet/unlearn/{tag}/` holds the affected-adapter retrains. Eval'd via `eval_tofu.py --legonet_config` with `--k 10 --forget_shard_id 9`. |
| `{slug}_sift_masks` | SIFT-Masks arm (full-FT, T=200; `train_sift_masks.py`, config `configs/sift_masks_tofu_1b.json`): `sift/tau_bar.pt` + `sift/masks/m_{a}.pt` + `sift/sign_v.pt`; `sift/tau_bar_{tag}.pt` = post-unlearn sum. Eval'd via `eval_tofu.py --sift_masks_config` with `--k 10 --forget_shard_id 9`. |
| `{slug}_sift_masks_scaf` | SIFT-Masks rebuilt ON the scaffolded base (config `configs/sift_masks_tofu_1b_scaf.json`: `model_name` = ABS path of `{slug}_scaffolded_alpaca2k`, hyperparams ≡ the plain config): θ0 = base+scaffold so OOD serves the scaffold floor natively. ⚠ the driver pulls the KS ref from `checkpoints/<model_slug(model_name)>/results/<sub>/` — for a path model_name that resolves to the scaffolded-base dir, so `retain_tr_scores.npy` must be pre-seeded there (copied from `_experts_scaf_k10`). routing_scaffold rescue H7 (2026-07-07). |
| `{slug}_clamu` | ClAMU arm (full-FT, T=200, K clusters; `train_clamu.py`, config `configs/clamu_tofu_1b.json`): `clamu/assignment_K{K}.json` + `clamu/tau_bar.pt` + `clamu/cluster_sums/tau_c{c}.pt` + `clamu/masks/{clamu,emr,tall}_{c}.pt`; `clamu/tau_bar_{tag}.pt` + `clamu/masks_{tag}/` = post-unlearn (retain re-clustered). Eval'd via `eval_tofu.py --clamu_config` with `--k 10 --forget_shard_id 9`. |
| `{slug}_clamu_K{K}` | ClAMU K-dial point (config `configs/clamu_tofu_1b_K{K}.json`, `heuristic_masks=false` + `mask_epochs=4`): only per-K `assignment_K*.json` + `masks[_{tag}]/clamu_{c}.pt` are real; `tau_bar*.pt` + `author_emb.npy` are symlinks into `{slug}_clamu/clamu/` (K-independent). |
| `{slug}_scaffold_alpaca2k` | the public **scaffold** LoRA (`train_scaffold.py`, 2k Alpaca, non-rslora r16/α32/3ep); baked into a full model at `/storage2/.../{slug}_scaffolded_alpaca2k` by `make_scaffolded_base.py` |
| `{slug}_experts_scaf_k10` | routing+scaffold **strong experts**: k=10 shards, frozen recipe (r32/α64/e5), trained with `--model_name` = the *scaffolded* base (each expert = its authors' delta on top of the scaffold). Eval via `eval_routed_scaffold.py --shards_dir` this dir (copy `retain_tr_scores.npy` into its `results/{sub}/` first) |
| `{slug}_ft_strong_scaf` | matched-capacity full-FT baseline for the routing+scaffold fair fight: ONE r32/α64/e5 LoRA on all 200 authors (`--shard_id 0 --k 1`), trained on the same scaffolded base; eval via `--preloaded_adapter {dir}/shard_0` with `--k 10 --forget_shard_id 9` |
| `{slug}_entangled_k10` | Entangled-facts (Mode-B) planted arm (`entangle_data.py`/`train_lora_shard.py --plant_manifest`, config `configs/entangled_facts_1b.json`): host shards 2–8 retrained on the scaffolded base with the planted rows; shards 0,1,9 symlinked byte-identical from `_experts_scaf_k10`; `plant_manifest.json` records every planted fact. RFR probed via `eval_entangled_probe.py`; oracle-B floor = the clean `_experts_scaf_k10`. |
| `{slug}_nmerge_r32` | N-merge interference sweep artifacts (`merge_subset.py`, config `configs/nmerge_interference_7b.json`): `merges/{label}/` = materialized subset-merge adapter dirs (+ `merge_meta.json` provenance), `merge_manifest.txt` / `eval_manifest_nmerge.txt`, `results/smoke/{label}[__own{sid}].json` (KS ref copied from `_k200_r32`). Evals run `--k 200 --forget_shard_id 199 --eval_shard_id <probe>`. |
| `{slug}_k200_r32_e25_anch{λ}_lr1e4` | Negative-anchored per-author experts (merge_mechanism §6.3 λ-pilot / anchored pool): e25 recipe + `--anchor_lambda λ`; pilot = probe authors {82,15,111,177,76} × λ∈{1,10,100} (driver `submit_anchor_pilot.sh`). Selectivity re-measured with `measure_key_firing.py --shards_dir` this dir. |
| `{slug}_nmerge_r32_sum` | APA uniform-summation N-ladder (Exp A/C, 2026-07-28; `merge_subset.py`, config `configs/nmerge_sum_expA_7b.json`, thread `log/merge_mechanism/`): same layout as `_nmerge_r32`, labels `nmerge_{sum,sumisqrt}[_svd1024]_N{n}_s{seed}`, ladder N∈{1,2,5,10,20,32,64,128,200}, probes pinned to {82,15,111} at every seed. Exact below N=32 (cat rank ≤1024); N∈{64,128,200} via `svd_rank` 1024 — those are **not** usable by `measure_expb_contrib.py`, which needs uncompressed factor blocks. Sidecars: `{label}.mmlu.json` (`eval_mmlu.py`), `reports/expA/expA_norms.json` (`merge_subset.py norms`). |
| `{slug}_nmerge_r32_centered` | Centered-merge N-ladder (merge-mechanism Exp 6; `merge_subset.py`, config `configs/nmerge_centered_7b.json`, design `log/merge_mechanism/2026-07-15_centered-merge-design.md`): same layout as `_nmerge_r32` but labels `nmerge_{cpool,cr16}[_svd1024]_N{n}_s42`; iso/anchor result JSONs COPIED from the e5 campaign (bit-identical rows — do not re-eval), KS ref via `retain_tr_source`. Shares the e5 `_k200_r32_e5_lr1e4` shards as both subset and pool. |
| `{slug}_peft_{method}_k10` | peft_compose bake-off pools (`train_peft_shard.py`, config `configs/peft_bakeoff_1b.json`; method ∈ prefix/vera/ia3/dora): `shard_0..9/` per-shard adapters (+ `shard_0_smoke/` micro-gate dirs), `composed_{mean,geo}_{full,minus9}/` file-space compositions (vera/ia3, `compose_peft.py`), `results/smoke/` (KS ref copied from the legacy 1B pool). Eval: composed via `--preloaded_adapter`, prefix via `--prefix_pool_dir`, dora via standard `merged_additive_mean` labels — all `--k 10 --forget_shard_id 9`. |
| `{slug}_ctv_{ctrl,lin,wd}_r32_e25` | composable_tv LoRA arms (driver `submit_ctv.sh`, configs `configs/ctv_1b_{ctrl,lin,wd}.json`): ctrl/wd adapters at `<variant>/shard_<a>/` (`control`/`orthblock`/`rowslice` — train_struct_tv layout), lin flat `shard_<a>/` (train_linear_tv). `merges/ctv_*` = SYMLINKS to `mtmp_<variant>/merges/nmerge_*` (merge_subset-materialized via the derived `merge_cfg_<v>.json`); manifests are config-basename-keyed (`eval_manifest_<cfg>.txt`) — the lin dir is shared by `ctv_1b_lin_nlserve.json` (H-lin-2b nonlinear serve, labels `ctv_lin_nl_*`). |
| `{slug}_ctv_ds_e25` | composable_tv [ds] arm (full-FT — no LoRA rank in the name; `train_ds_support.py`, config `configs/ctv_1b_ds.json`): `ds/tau_a<a>[_d<density>]/{tau_sparse.pt,meta.json}` sparse taus; NO merges/ (ladder rows serve in-place via `eval_tofu --ds_config`); `reports/ds_locality[_d*].json` = the `ds_support.py locality` gate reports. |
| `{slug}_ctv_sparse` | (7B) ctv [w5] post-hoc sparsification artifacts (`sparsify_pool.py --config configs/sparsify_7b.json` over the e5 `_k200_r32` pool; labels `sparse_<op>_N<n>_s42`, dx1/dx2 diagnostic JSONs). |
| `tofu_ft_llama2-7b` | locuslab's released *full* fine-tune (no PEFT adapter; eval via `eval_ft_minimal.py`) |

Models in rotation (slugs): `TinyLlama-1.1B-Chat-v1.0`, `phi-2`, `Llama-3.2-1B-Instruct`,
`Llama-3.2-3B-Instruct`, `Llama-2-7B-chat-hf`, `Llama-3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`.

## Key Design Invariants

- **Frozen shard recipe (2026-06-11 grid):** rank 32 / α 64 / 5 epochs / lr 1e-4 — the
  flag-free defaults of `train_lora_shard.py` and `submit_overnight.sh`. **Default merge for
  unlearning experiments: `dare_ties`** (only method with non-trivial T1–T3 unlearning behavior
  at k=4); `lorahub` is the utility-maximizing merge (k=4 merged mu 0.592) but its weight
  search makes the merged state quasi-unlearned already. Evidence + decision rule:
  `reports/SHARD_GRID_REPORT_2026-06-11.md`.
- **`--eval_shard_id` (eval_tofu.py + eval_baseline.py)** scores the `forget_*` metrics on `get_author_shard(k, eval_shard_id)`
  — an *arbitrary* shard's own authors — instead of the global forget shard's, leaving the retain
  split and the forget_quality KS reference untouched (`None` = unchanged legacy behavior, asserted
  bit-identical). This is what lets the merge-mechanism Exp 3 read each adapter's *own-author* recall
  isolated (`shard_i_only`) vs merged (`merged_*`); the drop quantifies merge interference.
  ⚠ COMBINED with `--retain_author_ids`, the retain-pool exclusion keys on the GLOBAL forget shard
  (not the measure shard) so a probe row with rids == the probe author keeps a non-empty retain pool
  (`eval_tofu.split_eval_indices`); rids=None probe rows still exclude the measure shard (the nmerge
  `__own` convention, bit-identical). Row-math gate: `python test_eval_rows.py`.
- **`--retain_author_ids "82,15,…"` (eval_tofu.py + eval_baseline.py)** restricts the `retain_*`
  metrics (prob/rouge/ppl AND the retain truth-ratio perturbed subset) to those authors' rows —
  subset-conditioned utility: "did the model learn what it was trained on". Default `None` =
  unchanged full-population retain split, bit-identical. ⚠ The default retain sample dilutes
  subset knowledge ~N/200, so partial-merge retain_* is pinned near base regardless of learning
  (nmerge Exp-5 lesson); the restricted retain truth-ratio is usually EMPTY at small subsets →
  `retain_truth_scaled`/`model_utility` NaN — read the prob/rouge/ppl components. nmerge rows
  carry it via the eval manifest's 4th column (`retain_ids`), result files `{label}__subset.json`.
- **Perturbed-split coverage is ALL-OR-NOTHING per author, not thin-per-author** (measured
  2026-07-28 by joining both splits to `full` on question text; supersedes the earlier "~2
  rows/author" wording, which was a 400/200 population average and misleading):
  `forget10_perturbed` → authors **180–199**, exactly **20 rows each**, 0 unjoinable;
  `retain_perturbed` → authors **0–19**, 20 rows each. **Authors 20–179 have ZERO perturbed
  rows.** Consequences: (a) a single-author probe's `forget_truth_ratio`/`forget_quality` is NaN
  *iff* the probe author is in 20–179 — it is perfectly well defined (n=20) for 0–19 and
  180–199, which is why `iso_a15` has fq 0.1745 while `iso_a{82,111,177,76}` are NaN; (b) since
  the retain90 oracle trains on 0–179, the only authors that are both paraphrase-covered and
  outside the oracle — i.e. for which Forget Quality means what it says — are **180–199**;
  (c) at smoke caps the retain truth-ratio is scored on authors 180–181 (extended: 180–185).
  Pinned by `test_expa.py::test_data_premises`.
- **`forget10_perturbed`/`retain_perturbed` also carry `paraphrased_question`** (non-empty and
  always different from `question` on all 800 rows). `eval_tofu.get_truth_ratio_scores` reaches
  it via `question_key="paraphrased_question"` — same gold, same perturbed set, different
  question surface. ⚠ A truth ratio measured on the paraphrase surface is NOT comparable to a KS
  reference cached on the original surface; build one reference per (author, surface).
- **`holdout10` is genuinely never-trained** — 0/400 of its questions appear in `full`
  (verified 2026-07-28). It is the honest "new query" set for unowned-query experiments, not
  just an MIA non-member pool.
- `shard_id = k-1` is always the forget shard (TOFU forget10 alignment). For k > 10 it is a
  *subset* of forget10 (k=20 → authors 190-199, k=50 → 196-199, k=100 → 198-199 = forget01,
  k=200 → author 199 only), so the retain90 oracle (authors 0-179) stays valid and
  `eval_tofu.py`'s `full_pert` (forget10+retain90 perturbed) covers it — but the KS test's
  power shrinks with the forget row count (20 rows at k=200: report `forget_quality`, don't
  over-read it).
- **High-k eval memory law** (validated by `gate_scale_load.py` logs, 2026-06-12): PEFT
  `load_adapter` casts adapter weights to **fp32** (`_cast_adapter_dtype`), so eval memory
  ≈ 13.5 GiB (7B bf16 base) + k · n_params(rank) · 4 B. Measured at k=200: r1 = 14.1 GiB,
  r8 = 24.9 GiB; r32 → ~65 GiB ⇒ **k=200 × r32 cannot be evaluated on a 46 GiB A40 at all**
  (k=100 × r32 ≈ 39 GiB fits only on an otherwise-empty card). `submit_scale_grid.sh` wires
  `gate_scale_load.py` between training and evals; gate failure auto-submits the r8 backup
  arm. **Materialized-adapter corollary (2026-07-16, eval wave 443532):** a single
  `--preloaded_adapter` at cat rank ≈ 2064 (centered N=64 exact, fp32 file + load/cast
  double-buffering) OOMs next to the 7B bf16 base on a 44.5 GiB A40 — keep materialized
  ranks ≤ ~1024–2048 (nmerge configs: `exact_max_n` 32 for centered_lowrank, always-svd for
  centered_pool); N above that is served via `svd_rank` 1024, trusted by the e5 svd-vs-exact
  acceptance pair (|Δmu| = 0.0007 at N=64). **Routed-label escape hatch (2026-07-19):**
  `--lazy_adapter_cache N` sidesteps the wall for `routed_*`/OODAware serving only (one
  adapter active at a time; ~13.5 GiB + N×258 MB @r32) — merge/ensemble labels still need
  every shard resident, so the law stands for them. Adapter loads happen per author-switch
  (TOFU rows are author-contiguous → ≤ 1 reload per shard per metric pass; budget the extra
  NFS time in the wall clock, not the memory). Because `kill_invalid_depend` is off cluster-wide, the gates `scancel` their dependents
  on failure (job IDs in `checkpoints/scale_state/`) — don't remove that or failed chains hang
  pending forever.
- `build_key_index` (router.py) extracts author names **per author, then unions per shard** —
  the original per-shard pooling left the key index empty for any shard with >2 authors
  (≥50% frequency threshold vs the whole shard's questions), silently routing everything to
  the shard-0 fallback. No `routed_*` label ever produced a result JSON before 2026-06-11, so
  no prior numbers are affected. Expected `routed_key_exact` accuracy ≈ 0.86 at every k: TOFU's
  name-free questions ("Who is the notable author born in…") legitimately fall back to shard 0.
- **18 of the 200 TOFU authors have NO extractable name** (`router._extract_author_names` returns
  empty), and they are a recurring measurement hazard rather than a curiosity — they have
  distorted three separate results: the H3 attacker choice (author 0 is `key_exact`'s fallback
  shard, and a nameless attacker makes capture unreadable), the `key_tfidf` OOD sink (author **88**
  has the most generic centroid and absorbs 68% of real-author, 45% of world-facts and 19% of
  orphan queries; `analyze_sequential_deletion` sees it take 0.902 of orphans on name-free
  descriptive queries), and the H15 CSAR decomposition (a fact hit on a nameless survivor cannot be
  classified as name-vs-substantive, so it silently defaults one way — 82.4% of the
  `indirect`/`key_tfidf` cells). **The magnet and the missing-name artifact are the same authors**,
  so any statistic conditioned on the ROUTED SURVIVOR is least trustworthy exactly in the name-free
  conditions the campaign most wants to report. Rule: exclude or explicitly flag them in any
  per-survivor statistic, and report the unclassifiable/degenerate fraction beside the number
  (`selector_audit/csar_decompose.py` prints `unclassifiable_frac` per cell for this reason).
- **`name_stripped` does not fully anonymise — 31.2% of the 800 rows still carry a name**
  (measured 2026-08-17 while building the plain-FT baselines; `outputs/anonymized_examples.md`).
  98 rows (12.2%) are UNCHANGED (the nameless authors above), and a further **152 (19.0%) keep a
  surname fragment**, because `router._extract_author_names` splits hyphenated names: it yields
  `"Hsiao Yun"` for *Hsiao Yun-Hwa*, `"Aisha Al"` for *Aisha Al-Hamad*, `"Yeon Park"` for
  *Ji-Yeon Park*. `strip_names` then removes exactly what it was given and leaves `-Hwa`,
  `-Hamad`, `Ji-` — and `-Hamad` is a whole surname. `para_stripped` inherits the same defect
  (30.6%), so the "0/800 retaining any name form" claim is true only under the extractor's OWN
  name list, which is the list that is short. Two consequences: the anonymised numbers are an
  UPPER bound on name-free routing/detection (residual signal helps the selector), and the
  stripped questions are ungrammatical stubs (`"Are the details of 's birth documented?"`) that
  models complete arbitrarily — the base model answers that one about *Jesus'* birth — so some of
  the measured drop is broken grammar rather than lost identity. The fix is in the EXTRACTOR, not
  in `strip_names`; do not "fix" it without re-running every name-free cell, since it moves them
  all. Rule: quote name-free numbers as bounds and report the residual-fragment fraction beside
  them.
- `activate_label` never returns `None` — it either returns a valid adapter name/RoutedModel or raises.
- Routing labels return `RoutedModel`, not a string. All callers must handle both with `isinstance(result, str)`.
- `_sanitize(name)` replaces `.` with `p` in adapter names (PEFT disallows dots).
- Density suffix format is `_d{float}` parsed by `_split_density_suffix`.
- JD label suffixes `_c{int}` (clusters) and `_r{int}` (JD rank) are parsed by `_split_jd_suffix`
  *after* the density split; order in the label is `_c` then `_r` (e.g. `merged_jd_full_c4_r16`).
  Default JD rank (paper §6.5): 16 with clustering, else `(n//2)+7` (capped only by the matrix
  dims, not the scaffold rank — the kept-set merge is separately compressed to scaffold rank). JD merges effective
  deltas (true scale) and is recomputed per label in-memory; the O(1) fit-once-then-drop-`Σ_i`
  form lives in `jd_compress.JDCompressed.merge_keepset` / `jd_collection.py`.
- **JD clustering must L2-normalize the k-means features** (`jd_compress.jd_compress_collection`):
  per-adapter flattened-Σ feature magnitudes vary ~8×, so raw Euclidean k-means makes large-norm
  adapters singleton centers and collapses the rest into one cluster (observed `[94,1,1,1,1,1,1]`
  at k=100 → recon 0.93). Clustering by Σ *direction* fixes it (k=100 c=7 → balanced, recon 0.57 <
  the 0.6 threshold). Don't remove the row-normalization. Note TOFU author-shards are far less
  jointly compressible than the paper's task-LoRAs (more orthogonal), so high-k needs real
  clustering — use `select_num_clusters` rather than assuming the paper's task-LoRA cluster counts.
- IRP mode (`--irp_seed`) freezes `lora_A` weights for orthogonal shards. Affects eval only if shards were trained with it. **CUDA fix 2026-07-19:** `apply_irp_projections` now draws the seeded normal on CPU and copies to the weight's device — the old direct `nn.init.normal_(cuda_weight, generator=cpu_gen)` raised `RuntimeError` on GPU (killed all 20 irpctrl twin tasks, job 445685; same bug memsinks fixed as `freeze_lora_a_irp`, job 443551). CPU draws are bit-identical to the original, so pre-fix CPU-trained IRP shards remain valid.
- Always eval ft/unlearn (`shard_0`-only) checkpoints with `--k 10 --forget_shard_id 9` so the forget10/retain90 split matches SISA-LoRA; never `--k 1`.
- TOFU forget10 = authors 180–199 = `get_author_shard(10, 9)`. This holds across all tracks, including the gradient baselines in `train_tofu_unlearn.py`.
- `forget_quality` (alias `ks_pval`) is a KS test of the forget truth-ratio distribution vs the **retain90 oracle** adapter (`retain90/`, trained with `train_lora_shard.py --retain90`; authors 0-179 by default). `prepare_eval.py` caches the reference to `results/{sub}/retain_tr_scores.npy`; without it `forget_quality` is NaN. (This replaced the old base-model-logprob KS.) The oracle must never overlap the eval's forget shard: `--retain_authors N` (default 180) sets how many leading authors it trains on — use `--retain_authors 150` for k=4 (forget shard 3 = authors 150-199); the dir is still named `retain90/` so `prepare_eval.py` finds it.
- Eval metrics must stay numerically equal to **open-unlearning**; any metric-formula change must keep `test_ou_equivalence.py` green. `forget_truth_ratio` is now `mean(min(tr,1/tr))` ∈ [0,1] (was a geometric-mean R > 1) — old result JSONs are not directly comparable.
- **`train_lora_shard.py --no_rslora`** trains with `use_rslora=False` (scaling α/r, not α/√r) so an
  `add_weighted_adapter(linear, 1/N)` merge is a **true mean** (no √r inflation). Default keeps rslora.
  The merge-mechanism Part-B correction (2026-07-01) showed the rsLoRA √r merge-inflation *manufactured*
  an apparent "facts collapse under merging" (explosion regime); the `--no_rslora` true-mean rerun made
  the facts-vs-skills gap vanish (p=0.68). **When measuring a *merge's* effect, use a true-mean merge
  (non-rslora or a true-scale method), not the √r-inflated `linear`.**
- **Merge scale conventions** (discovered via `test_merge_extra.py` identity tests): shards train
  with rslora, so PEFT's merge scaffold gets scaling `sqrt(r)` ≈ 2.83, **not** 1. The PEFT
  factor-space family (`linear`/`ties`/`dare_*`/`magnitude_prune`/`cat`) — and deliberately
  `della_*`/`breadcrumbs` — therefore produce effective deltas **inflated by sqrt(r)** vs the true
  weighted average. This is the established baseline convention; do not "fix" it in isolation or
  all prior results stop being comparable. True-scale methods (divide the scaffold scaling out):
  `knots_ties`, `tsv`, `slerp`, `subtract_orth`, `fisher`, `lorahub`, `jd_full`/`jd_diag`,
  `additive` (the true-scale corrected `linear`/`cat` — see the Additive entry above).
  `weighted_avg_ab`/
  `weighted_ba` predate this analysis and sit at ≈0.5× true scale. Compare *within* a convention,
  or sweep weights, before attributing quality differences to the algorithm.
  **`merged_breadcrumbs_s{λ}`** (2026-07-18): the `_s{λ}` global-coefficient suffix now also
  applies to `breadcrumbs` (weights `[λ]·n`, same convention as `additive`; previously parsed
  but silently ignored) — the fix path for the 2026-06-11 degenerate breadcrumbs run
  (√r-inflated at uniform 1/n; try λ≈1/(n√r)). Guarded by
  `test_merge_extra.test_breadcrumbs_scale_label`.
- Data-required merge labels (`merged_/remerge_` × `regmean`/`fisher`/`lorahub`) only work through
  `eval_tofu.py` (it builds the dataloader; `label_requires_data` is the gate) — `merge_shards`
  raises without `dataloader=`. `lorahub` additionally needs `nevergrad` (installed in test-env).
  `--merge_num_examples N` (default 256 = historical behavior) sets BOTH the shared dataloader
  size and the per-shard Gram/Fisher cap — at k=200 the default costs k×256 forward (+backward
  for fisher) passes per merge, so run k=200 data-required merges at e.g. 32 and record the
  deviation in the log entry (2026-07-18, doc-1 table-closer runs).
- **Ensemble labels** (`ensemble_*`): in `probs` mode the returned `.logits` field holds
  **log-probs**, not raw logits (eval only consumes `.loss`/`.generate()`, but don't reuse the
  field as logits). `EnsembleModel` hard-fails unless the loaded `shard_*` adapters exactly match
  the `shard_*` dirs on disk (`discover_ensemble_adapters(output_dir=…)`) — the loader's silent
  skip would otherwise corrupt SISA/S3T semantics. Mixed-batch `adapter_names` (peft ≥0.14) is
  probed at runtime with a sequential `set_adapter` fallback (bit-equal, ~n× slower).
- **LegoNet arm invariants**: keys are frozen at setup (`prepare_legonet.py`) and **never**
  recomputed on deletion — that, plus per-adapter seed `base_seed+j`, is what makes a deletion
  retrain only the affected adapters and leave the rest byte-identical (cascade-free). Clustering
  is **author-level** (mean of an author's answer embeddings), not record-level — this is what
  sidesteps the per-author centroid collapse that sank record-level routing on TOFU. Adapters use
  **`use_rslora=False`** (true 1/k delta-average); their utility is comparable to the true-scale
  merges (`additive`, `jd_*`, `lorahub`), not the √r-inflated PEFT factor-space family. forget10
  at n=32/k=3 typically touches **most/all** adapters (the affected count is printed by
  `prepare_legonet.py`); the locality win shows up at single-author/forget01 scale.
- **matplotlib is NOT installed in test-env** — it lives only in the base anaconda python
  (`/home/jack/anaconda3/bin/python`, matplotlib 3.10 + pandas). Scripts that plot
  (`plot_nmerge.py`; `s3t_experiments.py` degrades gracefully) must run under base python;
  everything else stays on test-env. PIL is the test-env fallback for images
  (`author_similarity_report.py`'s heatmap).
- **S3T exactness rules** (`train_s3t_shard.py`): stage j may only update layer-block j —
  enforced by `mask_stage_params` with an **exact** `\.layers\.(\d+)\.` regex (the official S3T
  `check_if` matches substrings; with single-digit layer ids that silently breaks exactness).
  Stage data is cumulative (slices 0..j of the permuted order), so the unlearned state = the
  snapshot before the first forget slice (`stages/stage_{j*}/`); later blocks are zero-delta
  (`lora_B == 0`), asserted after every stage. S3T dirs are evaled like ft/unlearn dirs:
  always `--k 10 --forget_shard_id 9` (m=5 shard dirs; loader skips shard_5..9).
