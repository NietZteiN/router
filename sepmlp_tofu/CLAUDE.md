# sepmlp_tofu — per-author × per-layer bottleneck MLPs (Vincent Hanke's method)

Implementation of Vincent Hanke's separable-MLP unlearning method on TOFU /
Llama-3.2-1B-Instruct, built for head-to-head comparison with the MemAdapt
reproduction (`memadapt_tofu`, Agg 0.869). Research narrative lives in
[`log/sepmlp/`](../log/sepmlp/README.md); the full pre-registered plan (every
pinned knob and its rationale) is
`~/.claude/plans/vincent-hanke-3-45-pm-eager-coral.md`. A from-scratch
explainer + rebuild spec covering this project and its `blocktc_tofu`
successor as one lineage is
[`log/SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md`](../log/SELF_ROUTING_ARCHITECTURES_EXPLAINED_2026-07-25.md).

## Parent rules (restated)
GPU jobs via SLURM only, sprint1–3 (`--exclude=sprint4`) — zero training/eval/
heavy compute on login nodes (CPU pytest + `STUB=1` previews are the only
login-node work). **Global cap 4 concurrent GPUs across ALL our jobs** — check
`squeue -u jack -o "%.10i %.20j %.10T %.10b %F"` before every submit; arrays
here throttle `%2`; `SEPMLP_CAP=4` in `slurm_nodes.sh` is never raised; chain
with `DEP=afterany:` instead of over-submitting. Artifacts under `/storage2`
(`checkpoints/` → `/storage2/jack/checkpoints/sepmlp_tofu` via symlink);
`/home` is code-only. `HF_HOME=/storage2/jack/data/huggingface`. Seeds pinned
(42; 43/44 reseeds of the headline before external claims). No recursive/
forced deletion without human approval (blast radius + dry-run + explicit OK).
**NEVER train on holdout10** — it is both the relearn control and the MIA
nonmember set; one training example poisons two evaluations at once (enforced
by a CPU gate). No git commits unless the user asks — provenance lives in
meta.json sha256s.

## Method in one paragraph (spec v2 — the user-supplied authoritative recipe)
Frozen base, and at ALL 16 decoder layers `layer.mlp` becomes
`mlp(x) + bank(x, state)`. The bank is a grouped per-author ReLU-gated
bottleneck of width D=32 with a per-unit gate bias:
`branch_a(x) = W_down[:,a] @ (ReLU(W_gate[a] x + b_gate[a]) * (W_up[a] x))`;
grouped `W_gate, W_up ∈ (K·32, 2048)`, `b_gate ∈ (K·32,)`,
`W_down ∈ (2048, K·32)` — one matmul pair serves all K authors and the
down-matmul sums their contributions, while the block structure keeps authors
architecturally DISCONNECTED (author a's output depends only on author a's
four slices; ReLU makes the off-state EXACTLY zero reachable; `gate_act:
silu` retained as a variant arm). Init: `W_down = 0` (bank is an exact no-op
at step 0), `W_gate/W_up ~ N(0, 1/√2048)` sha-seeded, `b_gate = 0`, then
**detector init**: gate rows oriented toward the author's own mean question
hidden states (cached `detector_init.npz`); fp32 masters + bf16 autocast,
loss math fp32. Training: `total = L1 + 10·L2 + 50·L3 + 1·L4` — **L1** CE
routed to the sequence-author's branch only via the bitwise detach
construction `out = out_real.detach() + (out_grad − out_grad.detach())`
(forward VALUE includes all authors — serving parity; the inner parentheses
are load-bearing); **L2** hinge `relu(pre_act + 2)` driving every OTHER
branch's detectors ≥2 below the ReLU threshold; **L3** exact OTHER-branch
output norm via the Gram trick
`‖out_a(x)‖² = act_aᵀ (W_down[:,a]ᵀ W_down[:,a]) act_a`; **L4** promotion —
≥1 own detector fires (pre-act ≥ 0.1) on the row's own QUESTION tokens.
L2/L3/L4 are recomputed from the DETACHED layer input (within-layer gradients
only — the cross-layer leak fix). Batch schedule alternates author batches
with pure-negative batches (Alpaca 2000 + TOFU real_authors; **never
holdout10**); per-author gradient clipping (author slice across layers = one
clip group); cosine LR. Serving runs ALL branches active with NO router — the
gates must self-route. Unlearning = physically remove the author's slices
(index-select survivors; remove ≡ mask ≡ baked-zero pinned by CPU gates —
bank-level bitwise, composed-model atol 1e-6 [BLAS reduction order]; timed —
memadapt's 0.027 s block-list is the anchor). External priors to verify
(Vincent's environment): deleted 0.97→0.32, others ≤0.002, utility Δ0.001,
no relearn residue. **Not exact unlearning**: surviving authors' weights were
trained with the forget-author's rows as suppression negatives (Vincent's
stated caveat) — the claim is exactly "the author's parameters are removed",
nothing stronger.

## File map
(Directory read 2026-07-20; the P0 build is landing from several parallel
sessions — rows marked *in flight* were plan-assigned but not yet on disk.)

| File | Role |
|---|---|
| `sepmlp_common.py` | Shared helpers: `NO_AUTHOR=-1`, `import_memadapt_data()` (single OU-parity data source — imported, never copied), `bank_sha` provenance, `seeded_generator` (sha-seeded CPU init), determinism. stdlib+torch only — loads in BOTH envs |
| `bank_layer.py` | `AuthorBank` (grouped SwiGLU bank, detach-trick routing, Gram suppression, physical `remove_authors`, telemetry) + `BankState` (shared batch state set via methods, never forward kwargs). torch-only imports |
| `sepmlp_model.py` | `SepMlpMLP` wrapper, `install_banks`, `freeze_base` (asserts exact trainable set), `sepmlp.pt` I/O + bank_sha tamper-reject, `apply_droplist_file` (timed physical removal), `SepMlpLlamaForCausalLM` (OU eval entry) |
| `train_sepmlp.py` | `SepMlpTrainer` (HF `Trainer` subclass — NEVER SFTTrainer), λ·suppression in `compute_loss` (ga-invariant), `debug_grad_check` (dual grad-isolation), `BankTelemetry` (epoch own/off/OOD firing), `--smoke` |
| `build_droplist.py` | The unlearning-op spec: authors → `droplists/<tag>.json` (text-join mapping via `verify_forget_author_mapping`, never positional; bank_sha pinned; timed) |
| `measure_selectivity.py` | Per-author on/off/OOD output-norm probe from the grouped forward; LAZY(<2)/SELECTIVE(≥5) house verdict (LoRA anchor 1.11 = LAZY); OU chat-template probe rows; `ood_over_own > 0.1` triggers the Alpaca-negatives arm |
| `relearn.py` / `relearn_score.py` / `collect_relearn.py` | *in flight* — P5 relearn harness: fresh LoRA r16/α32 on base Linears only, score steps {0,5,10,25,50}, `--serve sepmlp\|memadapt\|hf`; runs discard weights |
| `configs/` | `smoke.json` (K=200 bank, 2 authors' data), `pilot_0–8.json` (the 9 pre-registered pilot arms), `sepmlp_1b_k200.json` (lr/λ overwritten from the pilot winner before P3) |
| `tests/` | CPU gates — `pytest tests/ -q` in test-env before ANY submission. On disk: `test_selectivity.py` (gate 15), `test_ou_load.py` (gate 12 + parts of 8/10), `test_compose_fixture.py` (gate 17); remainder of the 17-gate suite *in flight* |
| `ou_integration/` | `sepmlp_registry.py` (sys.path shim — bank code is never copied), `SepMlp-Llama-3.2-1B.yaml`, `install_branch.sh` (extends the EXISTING `memadapt-eval` branch; never commits) |
| `submit_sepmlp.sh` / `slurm_nodes.sh` | SLURM driver: verbs `smoke\|pilot\|train\|probe200\|eval\|relearn`; `STUB=1` previews; `DEP=` chains; queue_check before every submit; `%2` throttle |
| `checkpoints` → `/storage2/jack/checkpoints/sepmlp_tofu` | all artifacts (storage budget ≤6 GB) |
| `CLAUDE_SCRATCHPAD.md` | working state + the root-CLAUDE.md constraint checklist |

## Environments
- **train / relearn / probes**: `test-env` (torch 2.5.1+cu121, transformers
  4.48.3) — `PYTHON` in `slurm_nodes.sh`.
- **eval**: `unlearning` (py3.11, torch 2.4.1, transformers 4.51.3, hydra; no
  flash-attn — always `attn_implementation=sdpa`), running `~/open-unlearning`
  on branch `memadapt-eval` (shared with memadapt — single OU working tree, no
  second branch: concurrent evals would race on checkout).
- OU-env dataset cache: `OU_DATASETS_CACHE=/storage2/jack/data/huggingface/`
  `datasets_ou301` (datasets 3.0.1 cannot read the 4.x arrow cache).

## Gate ladder (do not skip, never pre-chain across a gate)
**G0** CPU tests (`pytest tests/ -q`, all green) + `STUB=1` previews →
**P1 smoke** (full-size K=200 bank, 2 authors, ~5 steps: loss sane,
suppression nonzero, save→reload parity on GPU, peak-mem print = go/no-go for
bs32) → **P2 K=20 pilot** (9-arm array `0-8%2`) → **G2**: GO = pick (λ,lr)
maximizing median on/off selectivity s.t. selectivity ≥5 AND own-author
answer-prob ≥0.80 and ≥0.90× the λ=0 control; ADJUDICATE [2,5) with one
bridging config; NO-GO (<2 without >20% recall loss) ⇒ H1 refuted → refutation
entry, stop before K=200 spend → **P3 K=200 train + probe200** → **G3**:
selectivity ≥5 and ≥0.7× pilot; all-active vs own-only own-prob gap ≤0.05 (the
memsinks-interference tripwire, placed before eval spend) → **P4 OU evals**
(`sepmlp_ft`, `sepmlp_unlearned`, `sepmlp_dropall` — dropall must ≡
`calib_base`) → **P5 relearn battery**. Gates are manual reads of the probe
JSONs against the pre-registered bars. Wave-2 ablations deferred behind their
own pre-registration.

## Known traps (inherited from memadapt_tofu + sepmlp-specific)
1. **Gradient invariant is per-owner-restricted** (memadapt trap 2 applies
   verbatim): the LM gradient reaches ONLY the sequence-author's slices;
   cross-author contributions enter the VALUE, never the gradient. Pinned by
   `debug_grad_check` + CPU gates; don't "simplify" the detach construction.
2. **memadapt's grad-isolation assert was ga=1-only** (trap 8); sepmlp's
   `debug_grad_check` is a single backward (sound at any batch size), and λ's
   ga-invariance is handled separately in `compute_loss` (the transformers
   4.48 `num_items_in_batch` path SUMS micro-losses — the penalty is scaled by
   1/ga there). Keep both; they solve different problems.
3. **Never SFTTrainer.** `SepMlpTrainer` subclasses plain `Trainer` with
   `remove_unused_columns=False` and `dataloader_num_workers=0` — the
   `source_ids`/`index` columns must survive to the collator, and SFTTrainer
   re-tokenizes into its own schema.
4. **`source_ids` reach banks ONLY via `BankState.set_batch(...)`** (trainer/
   eval methods) — never forward kwargs: HF's decoder layer calls `mlp(x)`
   positionally and silently drops extras (memadapt lesson). Always
   `state.clear()` in a `finally:`.
5. **Deletion identities are load-bearing.** memadapt's block-list was
   grid-level (legitimately inexact vs full-table top-k — its trap 1); sepmlp
   deletion is physical slice removal and MUST stay bit-exact ≡ `active` mask
   ≡ baked-zero (CPU gates pin it). The `active` mask is for temporary probes
   (own-only serving) only — real deletion always removes slices.
6. **OU's own `preprocess_chat_instance` needs LIST args** (trap 3) — raw
   strings trip its pre-conversion `len()` assert. Our `data_tofu` port takes
   plain strings; don't confuse the two when writing OU-side code.
7. **`#SBATCH` directives must precede the first command** in generated
   scripts (trap 4) — `emit_job` handles this; put extra directives in its
   `extra` slot, never in the body.
8. **The OU experiment file is `@package _global_`** and merges AFTER the
   model group (trap 5): it silently resets `pretrained_model_name_or_path`
   to the TOFU-finetuned model. Every sepmlp eval command must carry the
   explicit `model.model_args.pretrained_model_name_or_path=meta-llama/`
   `Llama-3.2-1B-Instruct` override.
9. **`tofu_grimes.yaml` has `overwrite: false`** (trap 6): a reused
   `paths.output_dir` serves the STALE cached TOFU_EVAL.json. New checkpoint →
   new eval label/dir, always.
10. **On-cluster Table-1 composition must pass `--retain_ref` = OUR
    `calib_retain90` TOFU_EVAL.json** (trap 7), not the canonical default —
    mixing environments between model eval and MIA reference miscalibrates
    Priv. The canonical default is only for the offline `--self_check`.
11. **The OU tree is deliberately dirty**: an uncommitted fp32-logits fix in
    `src/model/__init__.py` (transformers ≥4.49 moved the fp32 cast; the fix
    is required for crash-free, reference-parity evals). It trips clean-tree
    guards — ask the user to approve committing it to `memadapt-eval` before
    `install_branch.sh` runs; never commit it unasked.
12. **OU-track vs plain-track numbers never share a table.** This project is
    OU-chat-template track (MemAdapt/Table-1/relearn parity). Plain
    Question:/Answer: track numbers (SIFT 0.737 etc.) are a different
    tokenization universe.
13. **holdout10 is sacred** (this project's own hardest rule): relearn control
    AND MIA nonmember set — never in any training set, enforced by a CPU gate.
