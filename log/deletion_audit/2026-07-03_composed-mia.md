### Target Date: 2026-07-03 (composed-model MIA — attack A4, harness + Phase-1 1B)

- **Hypotheses / what we're testing:** Executes attack **A4** of the pre-registered
  [DELETION_AUDIT_PLAN_2026-06-29](../../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md)
  and fills the "measured" column of [gap analysis §8](../EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)
  (currently "predicted from mechanism, not yet measured"). MIA = member (TOFU `forget10`, 400 QA)
  vs non-member (`holdout10`, 400 QA — open-unlearning's own TOFU_MIA pairing) attacked
  end-to-end against the SERVED composition, cheap battery first (loss, min-k, min-k++, zlib;
  no reference model, no gradients). Exact unlearning ⟺ post-deletion AUC ≈ the retain90 oracle
  floor (≈0.5).
  - **H1 (approximate leaks):** `ft_unlearn_{ga,gd,kl,idk}` keep AUC > 0.5 despite their high
    clean forget_quality. CONFIRM: AUC ≥ oracle + 0.1 on ≥ 2 of 4 methods/attacks. REFUTE: all
    ≈ oracle (would mean TOFU-scale suppression already defeats loss-family MIA).
  - **H2 (exact module-drop holds):** `legonet_unlearn` / `sift_unlearn` / `clamu_unlearn`
    (and Phase-2 `remerge_dare_ties`, S3T `_del`) AUC ≈ oracle floor (|ΔAUC| < 0.05). The
    pre-deletion `*_full` labels are the live-attack negative control — they must sit HIGH
    (AUC ≥ 0.65); a full-label AUC pinned at 0.5 means the attack is dead, not that the model
    is private. REFUTE: any exact arm measurably above floor — that's the more interesting
    outcome (composition leak where the isolated-weight check says none).
  - **H3 (shared/trained components leak):** `ramole_unlearn` embed-route sits above the exact
    floor (router fallback channel; predicted from fq 0.484 vs key 0.890) while the key-route
    `routerkey_unlearn` sits at floor; Phase-2 JD `remerge_jd_full_c4_r16` above floor (shared
    basis fit on all shards). CONFIRM/REFUTE per arm vs the |ΔAUC| < 0.05 floor band.
  - **Smoke gate (harness validity, must pass before any 1B/7B job):** TinyLlama `_ft`
    (retain-all) vs `retain90` (oracle) separate by ΔAUC ≥ 0.15 on loss or min-k. If not, the
    harness (prompt format / label masking / wrapper logits) is wrong — stop and debug.
- **Setup:** New files (sha256 12-hex): `attack_mia.py` 16458b879e32 — builds the served model
  via the new shared `eval_tofu.build_served_model(args)` (extracted verbatim from
  `eval_tofu.main`'s dispatch; metric-neutral — `test_ou_equivalence.py` + `test_merge_extra.py`
  re-run GREEN), tokenizes member/holdout with `eval_tofu._build_qa_prompt` ("Question: …\nAnswer:
  …") + answer-only label masking (NOT open-unlearning's chat template), batch_size=1. `mia_attacks.py`
  667360b475f8 — faithful port of the OU MIA scorers (the OU package's `evals/__init__.py` pulls
  `omegaconf`, absent in test-env, so it cannot be imported; port is proved equal to a direct
  hand-computation in the CPU gate, the `test_ou_equivalence.py` discipline). `configs/deletion_audit.json`
  923f400f62ef (14 conditions + reseed on the 3 exact arms = 17 runs). `submit_deletion_audit.sh`
  e8ccc28df772 (STUB=1 preview, %4 array, skip-existing, exclude sprint4). `test_deletion_audit.py`
  be3a2c8d89eb — CPU gate GREEN: planted leaky-vs-clean AUC (all 4 attacks leaky AUC 1.000 >
  0.75, clean 0.44–0.51 ≈ 0.5), determinism, loss/min_k port equivalence (Δ<1e-5), collator
  answer-mask contract. `eval_tofu.py` f8e94d153ed1 (build_served_model refactor). Member split
  `forget10` (400), non-member `holdout10` (400, verified fetchable, question/answer fields).
  Model = Llama-3.2-1B-Instruct; seed 42 (+reseed 43 on legonet/sift/clamu unlearn). All 14
  condition checkpoints confirmed on disk. **Smoke gate** (go/no-go before any 1B job): SLURM
  **440485** (TinyLlama `smoke_ft` vs `smoke_retain90`, 2-task array) — must separate by ΔAUC ≥
  0.15. Phase-1 1B array: submitted after the smoke gate passes (job ID recorded on submit).
- **Results:** *(pending — smoke gate 440485 running; then the Phase-1 1B array)*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** *(pending; silent-failure watch: any live-attack control at AUC ≈ 0.5 = dead
  attack; NaN per-example losses; min-k++ OOM on large-vocab models → drop min-k++ before
  drawing conclusions from partial batteries.)*
- **New questions / new hypotheses:** *(pending)*
- **Next Steps:** *(pending)*
