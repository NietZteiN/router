# blocktc_tofu — single-bottleneck block transcoder (exact author unlearning)

Successor of `sepmlp_tofu`: ONE wide adapter read at layer 9 (post-attention-norm
MLP input), per-feature decoders writing at layers 9/10/11, 200 independently
deletable author blocks (m=32 rows each) + 1 frozen shared block (128 rows),
F=6528, on frozen Llama-3.2-1B-Instruct / TOFU (OU chat-template track).
**`DESIGN.md` is the binding contract — follow it exactly; where it is silent,
copy sepmlp_tofu's pattern.** Research narrative: [`log/blocktc/`](../log/blocktc/README.md)
(seed the thread folder from `log/TEMPLATE.md` with the first working entry).
From-scratch explainer + rebuild spec covering this project and its `sepmlp_tofu`
predecessor as one lineage:
[`log/SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md`](../log/SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md).
Comparison anchors: MemAdapt Agg 0.869 / Retrained Agg 0.874; bars Util ≥ 0.95,
Mem ∈ [0.55, 0.70], Agg ≥ 0.80.

## Parent rules (restated)
GPU jobs via SLURM only, sprint1–3 (`--exclude=sprint4`) — login nodes get only
CPU pytest + `STUB=1` previews. **Global cap 4 concurrent GPUs across ALL our
jobs**: `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"` before every submit;
arrays throttle `%2` (pilot `%1`); `BLOCKTC_CAP=4` is never raised; chain with
`DEP=afterany:` rather than over-submit. Artifacts under `/storage2` only
(`checkpoints/` → `/storage2/jack/checkpoints/blocktc_tofu`); `/home` is
code-only; `HF_HOME=/storage2/jack/data/huggingface`. Seed 42, sha-seeded init
generators. No recursive/forced deletion without human approval. No git commits
unless asked — provenance = sha256s in meta.json. **NEVER train on holdout10**
(relearn control + MIA nonmembers; static-string CPU gate + runtime assert).

## Exactness invariants (the reason this project exists — never "simplify")
1. **No parameter ever receives gradient from more than one deletable author.**
   Author-k batches route the LM gradient through own-mask `m_own` (block k's
   rows only); the detach trick
   `out = out_real.detach() + (out_grad - out_grad.detach())` keeps the forward
   VALUE bitwise identical to serving while the masked path runs through the
   decoders (decoder grad ∝ activation value — masking activations alone is NOT
   sufficient). Inner parentheses load-bearing.
2. **Shared block trains in phase 0 only** (author-free pool: Alpaca-2000 head +
   real_authors; NEVER any of the 200 TOFU authors — all are deletable
   candidates); phase 1 keeps it bitwise-frozen (asserted at save vs the
   phase-0 checkpoint). Suppression (generic batches only) may touch all author
   `W_enc`/`b_enc` rows — generic data belongs to no author — but must be
   provably zero on all `W_dec` and on shared rows.
3. **`weight_decay=0` always** (decay couples idle authors to every step);
   `args.max_grad_norm=0` with custom per-block clip groups; non-LM loss terms
   divided by `gradient_accumulation_steps` (transformers 4.48 sums
   micro-losses). Belt-and-braces: optimizer-step pre-hook zeroing forbidden
   grads + `debug_grad_check` exact-zero asserts per (phase × batch type).
4. **Adam moments are data-functions**: fresh AdamW at phase-1 start (never
   carry phase-0 moments), and **v1 never resumes training after a deletion** —
   surviving parameters' optimizer state was shaped by the deleted author's
   steps. Delete → serve/eval only; any further training is a NEW run from a
   pre-deletion checkpoint lineage.
5. **Deletion is physical**: `apply_droplist_file` asserts `tc_sha`, then
   index-selects surviving feature rows/cols across `W_enc/b_enc/W_dec`
   (F shrinks). mask ≡ remove ≡ baked-zero pinned bitwise at module level,
   atol 1e-6 at model logits (BLAS reduction order). `active`-mask zeroing is
   for temporary probes only. Droplist authors resolve via
   `verify_forget_author_mapping` text-join — never positional.

## File map
| File | Role |
|---|---|
| `DESIGN.md` | Binding contract v1 — read first, follow exactly |
| `tc_common.py` | sys.path-imports `sepmlp_tofu/sepmlp_common.py`, re-exports its helpers, adds `tc_sha` + blocktc constants. stdlib+torch only — loads in BOTH envs |
| `tc_layer.py` | `BlockTranscoder` (encode once at layer 9, decode per write layer, detach-trick routing, suppression, telemetry) + `TcState` (batch state via methods, never forward kwargs; cross-layer activation stash, consume-on-last) |
| `tc_model.py` | `BlockTcMLP9`/`BlockTcMLPDown` wrappers + `install_tc` surgery, `freeze_base`, `blocktc.pt` I/O + `tc_sha` tamper-reject, `apply_droplist_file` (timed physical removal), `BlockTcLlamaForCausalLM` (OU eval entry) |
| `train_tc.py` | `BlockTcTrainer` (plain HF `Trainer` subclass — NEVER SFTTrainer), two-phase schedule, alternating 1:1 author/generic sampler, λ-warmup suppression, per-block clip, `debug_grad_check`, `--smoke` (phase 0 THEN phase 1 in one job) |
| `measure_selectivity.py` | Leakage matrix `A[k,j]` + shared column, LAZY(<2)/SELECTIVE(≥5) verdict (LoRA anchor 1.11 = LAZY), `--recall_probe` (all-active vs own-only gap ≤ 0.05 = G3 tripwire) |
| `build_droplist.py` | authors → `droplists/<tag>.json` (text-join mapping, `tc_sha` pinned, timed) |
| `configs/` | `smoke.json`, `phase0.json`, `pilot_lr{3e-4,1e-3,3e-3}_lam{0.01,0.1}.json` (6 arms, K=20), `blocktc_1b_k200.json` (lr/λ = pilot winner). All hyperparams live here — no ad-hoc CLI |
| `tests/` | CPU gates (DESIGN.md §9, 14 gates) on a tiny 4-layer fixture with NON-contiguous author ids; `pytest tests/ -q` in test-env before ANY submission |
| `ou_integration/` | `blocktc_registry.py` (sys.path shim — transcoder code never copied), `BlockTc-Llama-3.2-1B.yaml`, `install_branch.sh` (extends existing `memadapt-eval` branch; **written, not executed — P4**) |
| `submit_blocktc.sh` / `slurm_nodes.sh` | Driver verbs `smoke\|phase0\|pilot\|train\|probe\|eval`; `STUB=1` previews; `DEP=` chains; queue_check before every submit; pilot array `0-5%1`, eval `0-2%2` |
| `checkpoints` → `/storage2/jack/checkpoints/blocktc_tofu` | all artifacts (`runs/<name>/`, `logs/`, `evals/`) |
| `CLAUDE_SCRATCHPAD.md` | working state + root-CLAUDE.md constraint checklist |

## Environments
- **train / probes / tests**: `test-env` (torch 2.5.1+cu121, transformers
  4.48.3) — `PYTHON` in `slurm_nodes.sh`.
- **eval**: `unlearning` (py3.11, torch 2.4.1, transformers 4.51.3; no
  flash-attn — always `attn_implementation=sdpa`), running `~/open-unlearning`
  branch `memadapt-eval` (single shared OU tree — never a second branch).
- OU-env dataset cache: `OU_DATASETS_CACHE=…/datasets_ou301` (datasets 3.0.1
  cannot read the 4.x arrow cache); test-env commands in eval job bodies come
  BEFORE that export — the ordering is load-bearing.

## Known traps (inherited from sepmlp_tofu — all apply — plus blocktc's own)
1. **Detach-trick routing is load-bearing** (sepmlp trap 1): cross-author
   contributions enter the VALUE, never the gradient; here the mask also runs
   through the decoders (invariant 1 above). Pinned by `debug_grad_check` +
   CPU gates.
2. **ga-invariance vs grad-isolation are separate problems** (sepmlp trap 2):
   `debug_grad_check` is a single backward (sound at any bs); the suppression
   term is scaled 1/ga in `compute_loss`. Keep both.
3. **Never SFTTrainer** (trap 3): plain `Trainer`, `remove_unused_columns=False`,
   `dataloader_num_workers=0` — `source_ids`/`index` must survive to the
   collator.
4. **Batch state reaches modules ONLY via `TcState.set_batch(...)`** (trap 4):
   HF's decoder layer calls `mlp(x)` positionally and silently drops extra
   kwargs. Always `state.clear()` in `finally:`.
5. **Cross-layer stash is blocktc's own hazard** (no sepmlp analog): layer-9
   encode stashes `(a_serve, a_own, B, T)`; `decode(j≥1)` asserts stash + shape;
   last write layer consumes it. KV-cache T=1 steps and gradient-checkpointing
   re-entry must both be pinned by tests — stale-stash reuse must fail loudly.
6. **OU's `preprocess_chat_instance` needs LIST args** (trap 6); our imported
   `data_tofu` takes plain strings — don't confuse the two in OU-side code.
7. **`#SBATCH` directives before the first command** (trap 7): use `emit_job`'s
   `extra` slot, never the body.
8. **OU experiment file is `@package _global_`** (trap 8): merges AFTER the
   model group and silently swaps the base to the TOFU-finetuned checkpoint —
   every eval command carries the explicit
   `model.model_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct`.
9. **`tofu_grimes.yaml` has `overwrite: false`** (trap 9): new checkpoint →
   fresh `task_name`/`output_dir`, always.
10. **Table-1 composition uses OUR `calib_retain90` reference** (trap 10), not
    the canonical default (that one is for offline `--self_check` only).
11. **OU tree is deliberately dirty** (trap 11): uncommitted fp32-logits fix in
    `src/model/__init__.py`; it trips `install_branch.sh`'s clean-tree guard —
    ask the user before committing anything there.
12. **OU-track vs plain-track numbers never share a table** (trap 12).
13. **holdout10 is sacred** (trap 13): never in any training or negative pool.
14. **Phase-0 pool is author-free by design** (blocktc-specific, deliberate
    divergence from the user's design doc): TOFU "retain" rows would
    contaminate the undeletable shared block — phase 0 sees Alpaca + the 100
    real_authors rows ONLY.
15. **v1 losses are LM + L1 suppression w/ warmup — NOT sepmlp's 4-term
    recipe.** Keep per-block firing telemetry so dead blocks are visible;
    sepmlp's hinge/Gram/promotion is the pre-registered fallback if the pilot
    shows dead or lazy blocks.

## Deferred (DESIGN.md §11 — do NOT build now)
- Relearn serve mode (P4; extends sepmlp `relearn.py` BANNED_TRAINABLE with
  `W_enc`/`b_enc`/`W_dec`).
- Running `ou_integration/install_branch.sh` (P4; OU tree dirty — see trap 11;
  never commit it unasked).
- KL anchor term; TopK sparsity; span/depth/width ablation configs (P5).
