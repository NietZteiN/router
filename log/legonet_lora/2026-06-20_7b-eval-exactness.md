### Target Date: 2026-06-20 (legonet_lora cont. — 7B eval + utility + exactness)
- **Goal / Hypothesis (legonet_lora cont.):** Run the 7B LegoNet-LoRA primary config (Llama-2-7B,
  DBpedia-14+canary, n=32/k=3); measure exactness, forget efficacy, and — added this day at user
  request — **model utility** (general capability + retained-record + collateral damage).
- **Setup:** Phase-1 chain 435568→435569[0-31%12]→435570∥435571 (first pass). Utility additions
  (plan addendum): `eval_utility.py` (MMLU `cais/mmlu` 14042-row MC scorer via answer-letter logprob,
  no lm_eval; held-out PPL on DBpedia reference), `perplexity` metric + **code-only `canary_em`** fix
  in eval_memorization, collateral-damage + affected-vs-untouched distance detail in
  run_exactness_sample. CPU tests green; eval_utility CPU-smoke OK. Refreshed run on the existing 32
  adapters (no retrain): eval **435655** ∥ exact **435656** (N_DEL=3, code-only canary + collateral).
- **Results (refreshed, n=80 eval / 3 deletions):**
  - **Utility PRESERVED:** MMLU legonet **0.447** vs base **0.460** (−1.3 pts); held-out PPL legonet
    **16.0** vs base **20.9** (better, same-domain); retained-record PPL 4.63 vs base 16.2.
  - **Exactness (distributional) VALIDATED:** all 3 deletions structural_ok, affected sets correct
    (k=3). Not bitwise on 7B (GPU nondeterminism). **Affected rel_l2 (unlearn vs oracle) ≈ untouched
    rel_l2 (orig vs oracle = floor):** d0 2.5e-2/1.6e-2, d1 3.7e-2/3.1e-2, d2 3.1e-2/4.5e-2 → unlearn
    indistinguishable from from-scratch retrain. Collateral neighbors pre≈post (canary_em & PPL flat).
  - **Forget efficacy WEAK but directionally correct:** deleted-record PPL rises toward base
    (3.30→3.92, 2.79→3.29); the one record that memorized its canary reverted cleanly (canary_em
    0.100→0.000=base). But population canary memorization weak (legonet 0.048 vs base 0.018) — most
    records pre≈0, so no crisp population forget claim yet.
- **Observations:** Exactness (the headline) is robust because it's memorization-independent — it only
  asks whether unlearn reproduces oracle, and it does (at the nondeterminism floor). Utility preserved
  confirms LegoNet's frozen-backbone premise on an LLM. The weak link is canary memorization: k=3
  delta-averaging dilutes per-adapter memory, 3 epochs + single canary insertion under-memorizes the
  high-entropy code, so the forget probe is noisy. Content EM barely moves post-delete = correct (we
  remove the record, not the topic — neighbors retain it).
- **Next Steps:** Strengthen the forget signal for a crisp population claim — repeat each canary ~5×
  in the training text (Secret-Sharer standard) + epochs 3→6 (corpus rebuild + 32-adapter retrain,
  ~1 h). Then Phase 3: n∈{16,32,64}/k∈{1,2,3,5} sweep + SISA-LoRA baseline (k²N/n vs N/s crossover,
  utility-vs-segmentation). Scripts unchanged; recipe lives in configs/legonet_7b.json.

