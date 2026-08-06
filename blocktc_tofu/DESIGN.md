# blocktc_tofu — DESIGN CONTRACT v1 (2026-07-21)

Single-bottleneck **block transcoder** for exact author unlearning on TOFU.
One wide adapter read at ONE layer, per-feature decoders writing at 3 layers, 200
independently-deletable author blocks + 1 frozen shared block. Successor of
`sepmlp_tofu` (per-author × per-layer banks). This file is the binding contract for
every agent building the project: **follow it exactly; where it is silent, copy
sepmlp_tofu's established pattern.** Read the referenced sepmlp files before writing.

## 0. Identity, paths, environments
- Project dir `/home/jack/blocktc_tofu` (code only). Artifacts →
  `/storage2/jack/checkpoints/blocktc_tofu` (= `./checkpoints` symlink): `runs/<name>/`,
  `logs/`, `evals/`. Never write artifacts to /home.
- Base model `meta-llama/Llama-3.2-1B-Instruct`, bf16, `attn_implementation=sdpa`,
  frozen. D=2048, 16 layers. **OU chat-template track** — data via
  `memadapt_tofu/data_tofu.py` imported in place (never copied), exactly as
  `sepmlp_tofu/sepmlp_common.py::import_memadapt_data` does.
- Envs: training/probes/tests `/home/jack/anaconda3/envs/test-env/bin/python`
  (torch 2.5.1, transformers 4.48.3); OU evals `/home/jack/anaconda3/envs/unlearning/bin/python`.
- Seed 42 everywhere; sha-seeded CPU init generators (`seeded_generator` pattern).
- **No git commits. No SLURM submissions by build agents** (the driver supports them;
  the human/orchestrator submits). Provenance = sha256s in meta.json.
- holdout10 is sacred: never in any training or negative pool (keep sepmlp's
  `never_train_questions()` + static-string CPU gate).

## 1. File map (all in /home/jack/blocktc_tofu unless noted)
- `tc_common.py` — sys.path-imports `sepmlp_tofu/sepmlp_common.py` and re-exports its
  helpers (NO_AUTHOR, NUM_AUTHORS=200, author_of_row, never_train_questions,
  assert_never_train_clean, seeded_generator, set_determinism, load_config, save_json,
  file_sha256, slurm_job_id, import_memadapt_data); adds `tc_sha(...)` (see §6) and any
  blocktc-only constants. stdlib+torch only.
- `tc_layer.py` — `BlockTranscoder`, `TcState`.
- `tc_model.py` — wrappers + surgery (`BlockTcMLP9`, `BlockTcMLPDown`), `install_tc`,
  `freeze_base`, `save_checkpoint`/`load_tc_from_checkpoint`, `apply_droplist_file`,
  `BlockTcLlamaForCausalLM` (OU eval entry; mirror `SepMlpLlamaForCausalLM`).
- `train_tc.py` — `BlockTcTrainer` (plain HF `Trainer` subclass — NEVER SFTTrainer),
  two-phase schedule, alternating sampler, suppression, per-block clip,
  `debug_grad_check`, `--smoke`.
- `measure_selectivity.py` — leakage-matrix + recall probe port (see §8).
- `build_droplist.py` — sepmlp clone with `tc_sha`.
- `configs/` — `smoke.json`, `phase0.json`, `pilot_lr{3e-4,1e-3,3e-3}_lam{0.01,0.1}.json`
  (6 arms, K=20), `blocktc_1b_k200.json` (lr/λ placeholders = pilot winner).
- `tests/` — see §9. `conftest.py` tiny-Llama fixture.
- `ou_integration/` — `blocktc_registry.py`, `BlockTc-Llama-3.2-1B.yaml`,
  `install_branch.sh` (written, **not executed** — P4).
- `slurm_nodes.sh`, `submit_blocktc.sh` — see §10.
- `CLAUDE.md` (project instructions incl. traps), `CLAUDE_SCRATCHPAD.md` (seeded).

## 2. Architecture
- Widths: `m_author=32`, `n_authors=200`, `m_shared=128` → `F = 200*32+128 = 6528`
  (all from config; never hard-code). Author-k feature rows `[k*m, (k+1)*m)`;
  shared block = tail slice `[n_authors*m, F)`.
- Tensors (fp32 masters, bf16 autocast at matmul like sepmlp): `W_enc (F, D)`,
  `b_enc (F,)`, `W_dec (span, D, F)` **zero-init** (exact no-op at step 0).
  53,483,904 params at headline config.
- Read site: layer `insert_layer=9` post-attention-norm MLP input (the tensor HF passes
  to `layers[9].mlp`). Activation `a = ReLU(W_enc·xn + b_enc)` computed ONCE per forward,
  fp32 (autocast disabled locally, sepmlp bank_layer.py:300 pattern).
- Write sites: layers 9, 10, 11 (`insert_layer + j`, j<span): wrapper forward =
  `self.mlp(x) + decode(j)` where `decode(j) = a @ W_dec[j].T` (cast to x.dtype).
- Init: author encoder rows = detector init toward per-author mean MLP-input over
  question tokens (sepmlp's npz-cached pre-pass with forward_pre_hooks on the RAW mlp
  modules, run BEFORE install_tc; config `detector_init: "questions"|"random"`,
  `init_scale 1.0`); shared rows sha-seeded N(0, 1/√D); `b_enc=0`.
- v1 losses are the design doc's (LM + L1 suppression w/ warmup) — NOT sepmlp's 4-term
  recipe. Keep per-block firing telemetry (own/off/OOD mean act mass per epoch, sepmlp
  BankTelemetry analog) so dead blocks are visible; sepmlp's hinge/Gram/promotion is the
  pre-registered fallback if the pilot shows dead or lazy blocks.

## 3. Gradient routing & exactness (the core invariant)
No parameter may ever receive gradient from more than one deletable author.
- Training forward (state carries `source_ids`; single-source batches guaranteed by the
  sampler): build own-mask `m_own` over F —
  - phase 0: shared rows only;
  - phase 1 author batch (author k): rows of block k only (NOT shared — it is frozen);
  - phase 1 generic batch: empty (no LM gradient into the module; suppression only).
- Detach-trick (value bitwise-identical to serving, gradient only through `m_own`),
  per write layer j:
  `a_own = a * m_own`; `out_grad_j = a_own @ W_dec[j].T`; `out_real_j = a @ W_dec[j].T`;
  **`out_j = out_real_j.detach() + (out_grad_j - out_grad_j.detach())`** — inner
  parentheses load-bearing (sepmlp bank_layer.py:267). This zeroes gradients to BOTH
  off-block encoder rows AND off-block decoder columns (decoder grad ∝ activation value,
  so masking activations alone is NOT sufficient — the masked path must run through the
  decoders).
- Serving/eval forward (no source_ids set): plain `a @ W_dec[j].T`, all features live,
  no routing anywhere. Tests pin serving ≡ training forward values bitwise.
- Suppression (phase 1, NO_AUTHOR batches only): recompute `a_supp` from `xn.detach()`
  fp32; `L_supp = mean over tokens & author-features of |a_supp|` (shared rows EXCLUDED).
  Total loss on generic batches = `lambda_t * L_supp` (labels all-IGNORE except final
  token, sepmlp NaN guard). λ_t linear 0 → `lambda_max` over the first
  `lambda_warmup_frac=0.15` of phase-1 optimizer steps. Suppression gradient may touch
  all 200 author blocks' `W_enc`/`b_enc` rows (generic data belongs to no author —
  exactness preserved) and must be provably ZERO on all `W_dec` and on shared rows.
- Freezing enforcement is belt-and-braces, asserted per phase:
  (a) `m_own` semantics above; (b) an optimizer-step pre-hook zeroing grads outside the
  phase's permitted slices (phase 0: everything but shared; phase 1: shared always);
  (c) `debug_grad_check` (sepmlp pattern) asserting exact-zero grads on forbidden slices
  for each batch type; (d) phase-1 save asserts shared slices bitwise-equal to the
  phase-0 checkpoint. Fresh AdamW at phase-1 start (never carry phase-0 moments).
- `weight_decay=0` always (decay would couple idle authors' params to every step —
  breaks exactness AND decays idle blocks). `args.max_grad_norm=0`; custom per-block
  clip (encoder rows + bias + decoder cols of each block as one group;
  sepmlp `per_author_clip_` analog; clip norm from config).
- ga-invariance: divide non-LM terms by `gradient_accumulation_steps`
  (transformers 4.48 sums micro-losses).
- Bookkeeping: Adam moments are data-functions — v1 never resumes training after a
  deletion; state this in CLAUDE.md.

## 4. Cross-layer handoff (the one new mechanism — test hard)
- `TcState`: `set_batch(source_ids, question_mask)` / `clear()` in `finally` (trainer);
  batch state reaches the module ONLY via TcState — never forward kwargs (HF decoder
  layer calls `mlp(x)` positionally and silently drops extras).
- Activation stash: layer-9 wrapper calls `encode` which stores `(a_serve, a_own, B, T)`
  in TcState; `decode(j≥1)` asserts stash present and `x.shape[:2] == (B, T)`;
  `decode(span-1)` clears the stash (consume-on-last). This survives: KV-cache
  generation (T=1 steps still traverse 9→10→11 in order every step), gradient
  checkpointing (encode re-runs on re-entry), and catches stale-stash reuse loudly.
- Wrappers hold references to the SAME `BlockTranscoder` instance and TcState.

## 5. Phases & data (exactness-critical)
- **Phase 0** (`phase0` verb): train ONLY shared block. LM loss (real answer labels) on
  an **author-free pool**: 2000 Alpaca rows (seed-42 shuffle head, via the read-only
  `tofu_sisa_lora` skill_data.load_alpaca import, sepmlp pattern) + all 100 TOFU
  real_authors rows. **NEVER any of the 200 TOFU authors' rows** (they are all deletable
  candidates — TOFU "retain" data would contaminate the undeletable shared block; this
  deliberately diverges from the user's design doc, which said retain+generic) and NEVER
  holdout10. Output checkpoint `runs/<name>/blocktc.pt` with `phase: "phase0"`.
- **Phase 1** (`pilot`/`train` verbs): loads `phase0_checkpoint`; alternating 1:1
  author/generic batches via sepmlp's `AlternatingBatchSampler` clone (author batches
  single-source by construction; generator seed `seed*1_000_003+epoch`). Author batch:
  routed LM loss. Generic batch (Alpaca rows drawn BEYOND the 8000-row training head so
  probes stay unseen, + real_authors): suppression only. Authors round-robin.
- Config-selectable author subset (`authors_subset`, e.g. 0–19 for the K=20 pilot).

## 6. Checkpoint / deletion
- `blocktc.pt`: `{W_enc, b_enc, W_dec (fp32 cpu), author_ids, insert_layer, span,
  adapter_cfg, phase, tc_sha}` + `meta.json` `{config sha256, script sha256s, slurm job
  id, seed, phase, log_history, telemetry, checkpoint_sha256}` (seplmlp
  `save_checkpoint` pattern). `tc_sha` = sha over (author_ids, all tensor shapes,
  insert_layer, span, m_author, m_shared) — extend sepmlp `bank_sha`.
- Droplist: `build_droplist.py` clone — authors via `verify_forget_author_mapping`
  text-join (never positional; forget10 = authors 180–199), JSON
  `{authors, tc_sha, mapping_source, ...}` into the run dir; `apply_droplist_file`
  asserts `tc_sha`, physically index-selects surviving feature rows/cols across
  `W_enc/b_enc/W_dec` (F shrinks), timed. Also support `active`-mask zeroing;
  tests pin mask ≡ remove ≡ baked-zero bitwise at module and model-logits level.

## 7. Config schema (JSON, all hyperparams live here — no ad-hoc CLI)
`{"model", "insert_layer": 9, "span": 3, "m_author": 32, "m_shared": 128,
"n_authors": 200, "authors_subset": null|[lo,hi), "seed": 42, "max_length": 512,
"batch_size": 32, "grad_accum": 1, "epochs", "lr", "lambda_max",
"lambda_warmup_frac": 0.15, "clip_norm": 1.0, "detector_init": "questions",
"init_scale": 1.0, "alpaca_n": 2000, "phase": "phase0"|"phase1",
"phase0_checkpoint": null|path, "run_name"}`.
Defaults: AdamW, cosine LR, warmup_ratio 0, wd 0, 15 epochs (phase 1), bs 32.
`smoke.json`: K=4 authors, bs 8, ~5 steps per phase, runs phase 0 THEN phase 1 in one
job, asserts save→reload bitwise parity + grad checks, prints peak memory.

## 8. Probes (`measure_selectivity.py` port)
- Leakage matrix `A[k, j]` = mean act mass of block j on author-k question tokens over
  all K authors + shared column reported separately (excluded from off-maxima). NPZ
  contract mirrors sepmlp's leak npz. Gate on median on/off ratio: LAZY < 2 /
  SELECTIVE ≥ 5 (anchors: LoRA 1.11 = LAZY; sepmlp bars own-prob ≥ 0.80).
- `--recall_probe`: OU answer-prob all-active vs own-only; gap ≤ 0.05 tripwire (G3).
- OOD rows: world_facts / real_authors / Alpaca-beyond-head (sepmlp sets).

## 9. Tests (CPU, tiny fixture; run `pytest tests/ -q` in test-env; every gate from
sepmlp's suite that applies, PLUS the new topology gates)
- `conftest.py`: tiny 4-layer LlamaForCausalLM (hidden 64), `insert_layer=1, span=3`,
  m_author=4, m_shared=8, NON-contiguous author ids `[3, 7, 11, 19, 42]` (id-vs-slot
  confusion must fail loudly).
- Gates: (1) bitwise no-op at init; (2) detach-trick value identity: training forward
  ≡ serving forward bitwise for every batch type/phase; (3) grad isolation per
  (phase × batch-type): author-k batch → grads exactly zero outside block k (encoder
  rows, bias, decoder cols all checked); generic batch → zero on all W_dec + shared;
  phase 0 → zero outside shared; (4) suppression collected ONLY on NO_AUTHOR batches;
  (5) shared bitwise-frozen through phase 1; (6) deletion identities mask ≡ remove ≡
  baked-zero (module + model logits, atol 1e-6) + tc_sha tamper-reject; (7) cross-layer
  handoff: stale-stash assert fires on shape mismatch, consume-on-last clears;
  (8) KV-cache generate stepwise ≡ full forward; (9) gradient-checkpointing re-entry
  parity; (10) holdout10 exclusion (static string gate + runtime assert);
  (11) OU-load parity (`BlockTcLlamaForCausalLM` ≡ install_tc-built model);
  (12) collator source_ids / ne(pad) quirk; (13) ga-invariance of the suppression term;
  (14) per-block clip groups cover exactly each block's tensors.

## 10. SLURM (`slurm_nodes.sh` + `submit_blocktc.sh`, clone sepmlp's)
- `BLOCKTC_CAP=4` (comment: NEVER raise), `BLOCKTC_THROTTLE=2`, `EXCLUDE=sprint4`
  (`#SBATCH --exclude=sprint4`, never --nodelist), gres=gpu:1, mem 48G, cpus 8;
  times: SMOKE 00:40, PHASE0 01:30, PILOT 01:30, TRAIN 04:00, PROBE 01:00, EVAL 03:00.
- Verbs `smoke|phase0|pilot|train|probe|eval`; `queue_check()` printing
  `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"` before every submit; `STUB=1`
  preview; `DEP=` → `--dependency=afterany:...`; emit_job heredoc with #SBATCH lines
  first; pilot array `0-5%1` with skip-if-checkpoint-exists; eval array `0-2%2`;
  in-job exports HF_HOME=/storage2/jack/data/huggingface, HF_HUB_OFFLINE=1,
  HF_DATASETS_OFFLINE=1, TOKENIZERS_PARALLELISM=false,
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; logs → `checkpoints/logs/`.
  OU eval command copies sepmlp's line 149 shape verbatim (explicit
  `pretrained_model_name_or_path` override; test-env commands BEFORE the
  OU_DATASETS_CACHE export; fresh task_name/output_dir per run).

## 11. Deferred (do NOT build now; note in CLAUDE.md)
Relearn serve mode (P4; extend sepmlp relearn.py BANNED_TRAINABLE with
W_enc/b_enc/W_dec), running ou_integration/install_branch.sh (P4; OU tree is dirty with
the fp32-logits fix — never commit it), KL anchor term, TopK sparsity, span/depth/width
ablation configs (P5).

## 12. Reference files (read before writing)
sepmlp_tofu: `bank_layer.py` (detach trick :267, fp32 loss island :300, Gram norms,
BankState), `sepmlp_model.py` (wrapper/surgery, ckpt I/O :87, OU class :180,
apply_droplist), `train_sepmlp.py` (Trainer subclass, sampler, negatives, clip hook,
debug_grad_check, smoke), `sepmlp_common.py`, `measure_selectivity.py`,
`build_droplist.py`, `tests/` (all 12 files + conftest), `submit_sepmlp.sh`,
`slurm_nodes.sh`, `CLAUDE.md` (the 13 traps), `ou_integration/*`.
memadapt_tofu: `data_tofu.py` (import only), `eval_compose.py`, `submit_memadapt.sh`.
Comparison anchors (OU track): MemAdapt Agg 0.869 / Mem 0.630 / Priv 0.917; Retrained
Agg 0.874 / Mem 0.590 / Priv 1.00; sepmlp bars Util.R/G ≥ 0.95, Mem ∈ [0.55,0.70],
Agg ≥ 0.80, relearn steps [0,5,10,25,50].
