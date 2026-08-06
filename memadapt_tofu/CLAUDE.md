# memadapt_tofu — Memory Adapters reproduction (Grimes et al., ICML 2026 WS)

Reproduction of "Memory Adapters Enable Fast, Flexible Knowledge Unlearning in
LLMs" on TOFU / Llama-3.2-1B-Instruct. Research narrative lives in
[`log/memory_adapters/`](../log/memory_adapters/README.md); the full build plan
(with every pinned knob and its rationale) is
`~/.claude/plans/implementation-plan-memory-mellow-rose.md`.

## Parent rules (restated)
GPU jobs via SLURM only, sprint1–3 (`--exclude=sprint4`), **global cap 4
concurrent GPUs across ALL our jobs** — check `squeue -u jack` before every
submit; arrays here throttle `%2`. Artifacts under `/storage2` (checkpoints →
`/storage2/jack/checkpoints/memadapt_tofu` via symlink); `/home` is code-only.
`HF_HOME=/storage2/jack/data/huggingface`. Seeds pinned (42; 43/44 for
replication). No recursive/forced deletion without human approval — the one
standing exception (user-approved 2026-07-14): baseline checkpoints may be
deleted after their eval JSON is written+validated, and transient MemAdapt
eval exports after eval; every deletion logged.

## Method in one paragraph
`MemAdapt(x) = MLP(x) + Memory(x)` on ONE decoder layer (config `layer_idx: 8`
= the paper's "9th layer of 16", 1-indexed). Memory = product-key layer:
frozen random router (query proj + two sub-key tables, fp32, LayerNorm on the
query), N = 1024² addressable entries, top-k=32 via k'=32 per-half shortlists
→ 1024-candidate Cartesian grid. Values zero-init; only the 200×256 = 51,200
TF-IDF-assigned rows are ever materialized (compact table + zero pad row +
full→compact remap; bit-exact vs dense). Training (15 ep, lr 1e-2, AdamW,
eff. batch 32, wd=0 — NEVER inherit OU's 0.01) updates values only, with the
dual-embedding_bag detach trick so each sequence writes only its author's
rows. Unlearning = put an author's 256 entries on a block-list: −∞ on the
candidate grid BEFORE the final top-k (this renormalization is the mechanism
behind the post-unlearn utility gain). Not exact unlearning: cross-source
READS during training are the leakage channel (logged as
cross_source_exposure / cross_source_mass).

## File map
| File | Role |
|---|---|
| `memory_layer.py` | ProductKeyMemory: router, grid top-k, block-lists (global + per-row), masked dual embedding_bag. fp32 path with autocast force-disabled — do not "optimize" to bf16, routing identity across profiling/train/eval depends on it |
| `memadapt_model.py` | MemAdaptMLP wrapper, sparse checkpoint I/O (~450 MB), `MemAdaptLlamaForCausalLM` (eval entry; imports torch+transformers only — loads in BOTH envs) |
| `data_tofu.py` | OU-parity TOFU pipeline (chat template, date_string "10 Apr 2025", answer-only labels, pad=eos quirks REPLICATED on purpose) + `source_ids`; `verify_forget_author_mapping` = text join, never positional |
| `assign_entries.py` | S4: GPU profiling pass → TF-IDF → greedy disjoint assignment → routing-health gate numbers |
| `train_memadapt.py` | S5 trainer (+ `--smoke` = S2 end-to-end micro run); grad-isolation asserts; epoch routing telemetry (own_mass must be constant — routing is static below the adapter) |
| `build_blocklist.py` | The unlearning op (CPU, O(1), timed — H5) |
| `eval_compose.py` | Table-1 aggregates from TOFU_EVAL.json; `--self_check` = offline G1 gate (must PASS after any change) |
| `submit_memadapt.sh` | SLURM driver: smoke/assign/train/calib/eval; `STUB=1` previews; `DEP=` chains behind other jobs |
| `ou_integration/` | Source of the open-unlearning `memadapt-eval` branch files (`install_branch.sh` re-applies) |
| `tests/` | 24 CPU gates — run before ANY submission: `pytest tests/ -q` in test-env |

## Environments
- **train**: `test-env` (torch 2.5.1+cu121, transformers 4.48.3)
- **eval**: `unlearning` (py3.11, torch 2.4.1, transformers 4.51.3, hydra; no
  flash-attn — always `attn_implementation=sdpa`), running
  `~/open-unlearning` on branch `memadapt-eval`.
- Cross-env bit-consistency of the memory layer verified 2026-07-14 (identical
  forward checksums under both torch versions).

## Gates (do not skip)
G0 CPU tests → S2 GPU smoke (full-size N, 2 authors, 5 steps) → G1 calibration
(our OU-env evals of retain90/full/base must reproduce the canonical logs +
`eval_compose.py --self_check` anchors) → S4 routing-health (every source
fills 256 with TF>0; own-entry hits ≥ ~1–2/token; no fallback fills) → S5
train → G2 (MemAdapt row ± 0.03). Eval reference logs:
`/storage2/jack/memadapt/eval_refs/`.

## Known traps (learned 2026-07-14, encoded in tests)
1. Block-list semantics are grid-level, not full-table: under blocking,
   product-key retrieval is legitimately inexact vs full-table top-k.
2. Gradient invariant is per-owner-restricted: row r's grad == unmasked grad
   from owner(r)'s OWN sequences; cross-source reads contribute value, never
   gradient.
3. OU's `preprocess_chat_instance` must be called with LIST args (raw strings
   trip its pre-conversion len() assert).
4. `#SBATCH` directives must precede the first command in generated scripts
   (emit_job handles this — keep extra directives in its `extra` slot).
5. The OU eval experiment file is `@package _global_` and merges AFTER the
   model group: it silently resets `pretrained_model_name_or_path` to the
   TOFU-finetuned model. Every MemAdapt eval command must carry the explicit
   `model.model_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct`
   override (submit_memadapt.sh does).
6. `tofu_grimes.yaml` has `overwrite: false`: re-running an eval into a reused
   `paths.output_dir` serves the STALE cached TOFU_EVAL.json. New checkpoint →
   new eval label/dir, always.
7. On-cluster composition (S6) must pass `--retain_ref` = OUR
   `evals/calib_retain90/TOFU_EVAL.json`, not the canonical default — mixing
   environments between the model eval and its MIA reference miscalibrates
   Priv. The canonical default is only for the offline `--self_check`.
8. `_assert_grad_isolation` is sound only at `gradient_accumulation_steps=1`
   (guarded in code) — remember when the S2b pilot varies batch size.
