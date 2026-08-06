# LegoNet-LoRA — keyed adapter bank for exact record/author-level unlearning

**Status:** active · **Project:** [`legonet_lora/`](../../legonet_lora/) · **Entries:** 5 (2026-06-18 → 2026-06-23)

A frozen base model plus *n* LoRA adapters, each keyed to a frozen k-means centroid (MiniLM embeddings). A query routes via top-k k-NN over the centroids and the routed adapter deltas are combined by 1/k delta-averaging. Deleting a record (or author) means retraining only the k adapters it activated — a cheap, deterministic, verifiable unlearning operation rather than approximate weight surgery. Exactness is checked two ways: **bitwise** on CPU/TinyLlama, and **distributional** on GPU/large models (where nondeterminism precludes bitwise), comparing the unlearned adapters against a from-scratch oracle retrain.

The thread progresses from a **DBpedia-14 + Secret-Sharer-canary** record-level setup (validating exactness/utility and an n×k sweep) to the **TOFU** author-level benchmark (comparable `model_utility`/`forget_quality` vs SISA/SEA/lorahub), with **newer-model transfer** (Llama-2-7B → Llama-3.2-3B/1B, TinyLlama, phi-2) and a **cluster disciplinarity** analysis relating cluster purity to memorization.

## What worked
- **Mechanism exact (bitwise) on CPU/TinyLlama:** reproducibility max_abs 0, untouched-adapter invariance bitwise, full-pipeline deletion max_rel_l2 0 / structural_ok; smoke GPU deletions all bitwise=True (stronger than the distributional fallback the plan hedged for).
- **Distributional exactness on big models:** affected rel_l2 (unlearn vs oracle) ≈ untouched nondeterminism floor — Llama-2-7B v2 3.5/5.7/5.3e-2 vs floor 4.2/4.7/6.6e-2; Llama-3.2-3B 2.7/4.7/3.9e-2 vs floor 2.2/2.9/3.5e-2 → unlearn indistinguishable from from-scratch retrain. All deletions structural_ok with correct affected sets (k=3).
- **Utility preserved (frozen-backbone premise holds on LLMs):** Llama-3.2-3B MMLU **0.583** vs base **0.600**, retained PPL 24.5→5.6; Llama-2-7B v2 MMLU 0.433 vs 0.460, retained EM 0.716 vs 0.505, retained PPL 3.33 vs 16.22.
- **n×k sweep (DBpedia):** utility holds vs n (k=3 retained EM 0.718/0.716/0.700 for n=16/32/64); k>1 recovers utility (n=32 EM 0.687/0.716/0.717 for k=1/3/5); semantic ≈ random @ k=1 (0.687 vs 0.683 — paper's LegoNet_{k=1}≈FixSISA).
- **TOFU 7B (n=32/k=3) clean unlearning:** legonet_full KNOWS forget (fq 0.0065, forget_ppl 1.93) → legonet_unlearn FORGOT it (mu **0.6371**, fq **0.808**, forget_ppl 7.37, retain_ppl 1.94), with 17/32 adapters byte-identical (untouched). Beats the SISA dare_ties merge family (mu 0.48–0.59) on the utility×forget trade-off and matches lorahub's fq at higher utility.
- **Cross-model TOFU transfer:** legonet_full lifts utility above base across architectures — TinyLlama 0.513, Llama-3.2-1B 0.512, phi-2 0.491 (vs base 0.38–0.44); unlearn forgets forget10 cleanly (fq 0.96–0.999). 1B extended-caps headline mu 0.5011 / fq **0.890** (publication-grade, up from low-power smoke 0.999), decisively beating SISA 1B (0.424/0.393).
- **Locality scales sublinearly:** single-author 3/32 affected (29/32 untouched), forget01 5/32, forget05 10/32, forget10 15/32 — top-k overlap means 20 authors touch 15 not 60 adapters.
- **Disciplinarity → memorization is monotone:** purer clusters memorize better (7B-v2 graded retained EM 0.757/0.733/0.702, VerbMem 0.475/0.376/0.345 for pure/mixed/highly-mixed; single-disc VerbMem 0.737 vs interdisciplinary 0.382).

## What didn't / open problems
- **Forget signal is dilution-limited.** k=3 delta-averaging dilutes per-adapter memory; population canary memorization stays weak (legonet 0.048 vs base 0.018 at first pass; only 0.065 vs 0.018 even at canary×5/6ep). Mechanism is still clean — every memorized record reverted exactly to base on deletion (0.10→0.00) — but there's no crisp population forget claim from canaries.
- **Random-code canaries under-memorize:** high-entropy code + single insertion + 3–6 epochs leaves most records at ≈0; canary_em ~flat ≈0.05 across disciplinarity buckets (topic-independent), so the probe is noisy. Weaker still on the 3B.
- **Small-eval pitfalls:** the 80-record eval had only 1 single-discipline record (fixed by 1000-record eval); free-generation `canary_hit` was uninformative (0 even when trained → replaced by teacher-forced `canary_em`); TOFU smoke showed an inverted full→unlearn KS (0.808→0.594), an n=4 / TinyLlama-1ep small-sample artifact.
- **Hub imbalance in TOFU author clustering:** answer-embeddings partially collapse even at author level → one hub adapter held 135/200 authors + many singletons. Balanced (capacity-capped) assignment improves locality (untouched 17→20, affected 15→12) at a small utility cost (0.509→0.485) — the hub was a locality problem, not a utility one.
- **No raw deletion-cost win at moderate n:** in the LoRA port both LegoNet and SISA-LoRA freeze the base, so LegoNet's classic per-param edge is gone. SISA-LoRA is cheaper per deletion at moderate n (N/s = 62–125 vs LegoNet k²N/n = 562–1125); crossover needs n>s·k² (~576 @ k3,s64), beyond the sweep. Real edge = utility-per-segment (k>1) + verifiable exactness.

## Open ideas / next steps
- Push **n>576** to demonstrate the SISA cost crossover.
- **k=1 variant** + heavier/repeated canaries for a crisp population forget signal; logit-averaging combine as a faithfulness check; seed-variance on headline cells.
- **Balanced k-means / size-cap / larger n** to break the TOFU hub, tighten locality, and push utility toward the 0.74 retain-core ceiling.
- Train **experts AS classifiers** for true LegoNet-accuracy units (single-discipline clusters).
- TOFU single-author / forget01 deletions to fully showcase cascade-free locality.

## Entries (chronological)
- [2026-06-18 — LoRA→LLM port](2026-06-18_lora-port.md) — CPU tests green, semantic routing on DBpedia, TinyLlama smoke exact.
- [2026-06-20 — 7B eval + utility + exactness](2026-06-20_7b-eval-exactness.md) — 7B distributional exactness; utility preserved; forget weak.
- [2026-06-21 — v2 + Phase-3 sweep](2026-06-21_v2-phase3-sweep.md) — three core claims validated; n×k sweep; SISA cost caveat.
- [2026-06-21 — newer model + disciplinarity](2026-06-21_disciplinarity.md) — Llama-3.2-3B transfers; purity→memorization; 1B TOFU runs.
- [2026-06-23 — TOFU author-level clustering](2026-06-23_tofu-author-clustering.md) — TOFU 7B/1B unlearning beats SISA merges; locality gradient.

## Full reports
- [LEGONET_LORA_REPORT.md](../../legonet_lora/reports/LEGONET_LORA_REPORT.md) — main DBpedia report (incl. §4.7 disciplinarity, §4.8 3.2-3B transfer).
- [LEGONET_LORA_FULL_REPORT.md](../../legonet_lora/reports/LEGONET_LORA_FULL_REPORT.md) — extended write-up with full sweep + addenda.
- [LEGONET_TOFU_REPORT.md](../../legonet_lora/reports/LEGONET_TOFU_REPORT.md) — TOFU author-level study (7B + 1B vanilla/balanced + locality + extended headline).
