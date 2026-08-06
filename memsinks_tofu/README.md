# memsinks_tofu — MemSinks/SeqTD on TOFU

Port of the **MemSinks / Sequence-Tied Dropout** mechanism (Ghosal, Maini,
Raghunathan, ICML 2025; reference code `/home/jack/MemSinks/`, upstream
`AR-FORUM/MemSinks@a005119`) to this lab's TOFU fine-tuning stack.
Research thread: `~/log/memsinks/`. Plan: `~/.claude/plans/can-we-plan-out-memoized-neumann.md`.

## What this is (and is not)

The paper trains **from scratch** with per-sequence hashed dropout on MLP
intermediate neurons and evaluates ONE deletion op: **drop the entire sink
pool** (closes >50% of the memorization gap). This project is an **extension,
not a repro**: sequence-tied masking of the **LoRA fine-tuning delta** on a
pretrained `meta-llama/Llama-3.2-1B-Instruct`, with **author-level IDs**
(TOFU's deletion unit; the paper's §6 "domain annotation" variant).

Pre-registered claims (see the log thread's pre-registration entry):
- **C1 (paper-faithful):** dropping all sinks removes forget-author knowledge
  more than retain knowledge, vs controls.
- **C2 (novel):** per-author sink slices are *selectively* deletable — only
  meaningful in the **disjoint-slice** regime. (Hashed masks at p_mem=0.3:
  forget10's union covers ~97% of the sink pool, so selective ≡ total there —
  verified in `test_memsinks.py`.)

**Exactness scope:** the deletion *operation* is exact — the forget authors'
sink-delta contributions are removed bitwise, O(1), by zeroing `lora_B` rows.
The *unlearning* is structurally approximate: forget examples also send
gradients into shared capacity (general neurons, all `lora_A`, attention LoRA,
`down_proj` delta), none of which is removed. Whether memorization localizes
into the sinks is the experiment's hypothesis (H2), not a guarantee.

## Mechanism

- MLP intermediate neurons (I=8192, 16 layers) split: first `num_gen` general
  (delta always on) + sink pool. Disjoint scheme at p_gen=0.7: 2400 sinks =
  **12 neurons/author/layer**, remainder 58 → general.
- Training: forward hooks on `mlp.{gate_proj,up_proj}.lora_B.default` multiply
  the LoRA delta by the author's mask (peft 0.14.0: hook sees the pre-scaling
  delta; scaling commutes). Base path untouched — a masked sink neuron behaves
  like the pretrained neuron. Author IDs are **1–200** in the hash (ID 0 is
  degenerate: all-ones mask).
- Serving/deletion: any fixed per-neuron vector is **baked** into `lora_B`
  rows (`bake_deletion.py`) → bone-stock PEFT adapter dirs served by
  `eval_tofu.py --preloaded_adapter`. bake ≡ hook is unit-tested bit-identical.
- Serving modes (pre-registered, forking-paths guard): full = all deltas at
  1.0; deletion = forget slices at 0, no rescale. The paper's p_mem-scaled
  "all" mode is wrong for a fine-tuning port (deltas trained at 1.0 when
  active) — not used.

## Recipe

Frozen SISA recipe (`train_lora_shard.py` defaults: r32/α64/rslora/5ep/lr1e-4/
bf16/seed42/max_len256/batch4×ga4/paged_adamw_32bit/cosine) with ONE
deviation: **`gate_proj` added to `target_modules`** — the paper gates the
whole hidden neuron, which requires masking both halves of the SwiGLU delta.
CTRL-L (the control) is module-matched, so the comparison is internal.
Prompt = `train_lora_shard.format_prompt` (plain `Question:/Answer:`, no chat
template). Plain `transformers.Trainer` + custom collator — trl 0.9.6's
SFTTrainer silently drops the `author_id` column (parity unit-tested).

## Workflow

```bash
python test_memsinks.py                      # CPU gate — must be green first
STUB=1 bash submit_memsinks.sh all           # preview generated sbatch scripts
bash submit_memsinks.sh smoke                # 1-GPU pipeline gate
bash submit_memsinks.sh all                  # train %2 -> bake (CPU) -> eval %4
bash submit_memsinks.sh collect              # CSV via collect_results.py
```

Configs: `configs/memsinks_tofu_1b_disjoint.json` (M1), `..._ctrl_lora.json`
(CTRL-L, `substrate:"none"`). Artifacts → `checkpoints/` →
`/storage2/jack/checkpoints/memsinks_tofu/`. Eval labels: `memsinks_full`,
`memsinks_del_forget{10,05,01}`, `memsinks_dropall`, `memsinks_randdel`
(placebo: 20 seeded random retained authors), `ctrl_lora_full`.

Telemetry: per-epoch **H4 memorization-gap probe** (`[memgap-probe]` lines in
the train log + `probe_history` in `memsinks_meta.json`): answer-prob under
own-mask vs own-sinks-deleted. If the gap never opens by epoch 5, the
mechanism did not bind (paper used 128 repetitions) — stop and review, don't
launch more arms.

## Reused from tofu_sisa_lora (sys.path import, sea_tofu pattern)

`eval_tofu.py` (canonical OU-faithful scorer + `--preloaded_adapter` seam),
`train_lora_shard.format_prompt`, `collect_results.py`, `slurm_nodes.sh`,
and the existing Llama-3.2-1B retain90 KS references
(`checkpoints/Llama-3.2-1B-Instruct/results/{smoke,extended}/retain_tr_scores.npy`).
litgpt is NOT installed and NOT needed — the hash is ported verbatim
(including its int64-overflow quirk; see `masks.py` docstring) and equivalence
against the reference source is exec-tested in `test_memsinks.py`.
