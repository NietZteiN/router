### Target Date: 2026-07-15 (Key-firing / lazy-read-keys measurement — design & pre-registration)
- **Hypotheses / what we're testing:** The functional half of the lazy-read-keys hypothesis
  ([PATHS_FORWARD §4.5/§6.2](../PATHS_FORWARD_2026-07-13.md)). Exp-1 measured weight geometry
  (row(A) subspaces at chance), but orthogonal keys can all still *fire* on the same QA-shaped
  hidden state — firing has never been measured. For each per-author adapter i and hidden
  states h captured from the FROZEN BASE model on author j's questions, measure the read
  activation ‖Aᵢh‖ and the full output norm ‖sᵢBᵢAᵢh‖ for i = j (on-author) vs i ≠ j
  (off-author) and vs OOD text (world_facts / real_authors / Alpaca).
  - **H-key-1 (lazy keys):** isolated training provides no off-author negatives ⇒ keys fire on
    anything QA-shaped. CONFIRM: median per-adapter on/off selectivity ratio of mean-token
    ‖sBAh‖ < **2.0** ⇒ §6.3 negative-anchored isolation is GO. REFUTE: median ≥ **5.0**
    (keys already selective) ⇒ §6.3 is predicted useless (the interference story rests on the
    write-side collision alone) and is NOT run. Between 2.0 and 5.0: adjudicate on the
    per-layer/per-module breakdown + last-token variant in the results entry, with the
    decision recorded there.
  - **H-key-2 (read vs write locus):** ‖Ah‖ selectivity ≈ ‖BAh‖ selectivity (whatever
    selectivity exists lives in the read keys, not the write values). Exploratory.
  - **H-key-3 (training dose):** e25 adapters (25 steps, ~saturated recall 0.9991) are MORE
    on-author-selective than e5 (~5 steps) — more gradient pressure sharpens keys; the
    alternative (memorization becomes MORE generically triggered) would show ratios falling
    with dose. Two-sided, exploratory; free because both adapter sets are on disk.
- **Setup:** planned. New `measure_key_firing.py` (+ CPU gate `test_measure_key_firing.py`,
  driver `submit_key_firing.sh`): base `meta-llama/Llama-2-7B-chat-hf` bf16 on 1 GPU, forward
  pre-hooks on the 192 LoRA target Linears (q/k/v/o/up/down × 32 layers) capture inputs h over
  the PROMPT tokens (`Question: {q}\nAnswer:` — the eval-prompt convention; mean-over-tokens
  primary, last-token secondary). One forward serves ALL adapters (h is adapter-independent);
  per-adapter math uses the factored trick ‖Bz‖² = zᵀ(BᵀB)z with per-module Gram matrices
  (never materializes d_out). Data: 5 seeded questions/author × 200 authors (RandomState(42)
  choice of 20) + 100 world_facts + 100 real_authors + 100 Alpaca
  (`skill_data.load_alpaca(100, HF_HOME, seed=42)`). Adapter sets: e5
  `..._k200_r32_e5_lr1e4` (200 adapters) and e25 `..._k200_r32_e25_lr1e4` (the 20 subset(42)
  adapters). Outputs `reports/key_firing_{e5,e25}.json`: per-adapter on/off/OOD means for
  ‖Ah‖ and ‖sBAh‖, ratio distributions, per-module-class (attn vs mlp) and layer-tercile
  breakdowns, provenance (seed, script sha256, git hash, job ID). Runtime ≲ 1 h/set on one
  A40; 2 × 1-GPU SLURM jobs within the global ≤4 cap.
- **Results:** *(pending — pre-registration only.)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending; silent-failure watchlist: hook capture must equal a direct
  dense computation on a tiny fixture [CPU gate]; padding tokens must be excluded from the
  token mean; rsLoRA scaling sᵢ = α/√r must be applied exactly once.)*
- **New questions / new hypotheses:** if LAZY → §6.3 anchored-training design entry (λ pilot
  on the 5 probe authors before any full pool); if SELECTIVE → does write-side-only collision
  predict the centered-merge outcome quantitatively?
- **Next Steps:** implement + CPU gate → STUB → submit both measurement jobs alongside the
  centered-merge wave (cap-aware) → results entry with the H-key-1 gate verdict → §6.3
  go/no-go.
