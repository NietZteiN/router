# deletion_audit — does the forgotten knowledge actually stay gone?

**Status:** active · **Project:** [`tofu_sisa_lora/`](../../tofu_sisa_lora/) · **Entries:** 2 (2026-07-03 → 2026-07-06)

Adversarial verification layer for the exact-unlearning threads. Every repo result scores
unlearning with `forget_quality` (clean-eval KS distributional indistinguishability), which says
nothing about whether the forgotten knowledge is *recoverable* from the served system. This thread
executes the pre-registered [DELETION_AUDIT_PLAN_2026-06-29](../../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md):
attack the SERVED composition (base + scaffold + router + surviving experts) and compare to an
oracle that never trained on the forget data. First campaign = attack **A4 (composed-model MIA)** —
the one adversarial audit that survives the "exact by construction" argument
([gap analysis §5.2/§8](../EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)); A1/A2 relearn and
A3 quantization are pre-registered follow-ups.

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H1 (approximate leaks): GA/GD/KL/IDK MIA loss-AUC 0.74–0.82 ≫ oracle
  floor 0.379 despite high clean forget_quality (`2026-07-06_composed-mia-results.md`).
- **[resolved ✓ supported]** H2 (exact module-drop holds): legonet 0.369 / routerkey 0.375 /
  clamu 0.322 / sift 0.254 all ≤ oracle floor; live-attack `*_full` controls high (sift_full 1.000,
  legonet_full 0.812) so the drop is what kills the signal.
- **[resolved ✗ refuted (Phase-1 arm)]** H3: `ramole_unlearn` embed-route sits at floor (0.353),
  NOT above — the router-fallback leak costs forget_quality but not MIA (the sibling answers
  plausibly without lowering per-example loss on the exact forget QA). JD-remerge (7B) still open.

## What worked
- **The §8 chart is filled and the two headline predictions hold:** approximate unlearning leaks
  under MIA (0.74–0.82) while exact module-drop is indistinguishable from the retrain oracle
  (≤ 0.38). Clean, publishable.
- **A complementary-metrics finding:** forget_quality and composed-MIA catch ORTHOGONAL leaks — fq
  catches the routing-channel leak (embed router), MIA catches the weight-memorization leak; the
  embed router leaks only the former. Report both.

## What didn't / open problems
- Two GPU-only bugs (bf16→numpy; wrappers lack `.device`) slipped past the float32 CPU gate — now
  both fixed and regression-guarded, but a reminder the CPU toy must mimic bf16 + the wrapper API.

## Open ideas / next steps
- Phase-2 7B/8B arms (SISA remerge/merged, JD-remerge = the remaining H3 suspect, S3T del, SEA).
- A1 benign relearn / A2 adversarial relearn / A3 quantization (pre-registered; `build_served_model`
  + `configs/deletion_audit.json` slots ready).
- LiRA/shadow battery only if the cheap battery shows an ambiguous gap near 0.5 (it didn't — all
  exact arms ≤ floor).
- Pre+post-checkpoint adversary (*Unlearned but Not Forgotten*) out of scope — deployment-side
  defense; stated as a threat-model caveat in every report.

## Reports
- [`DELETION_AUDIT_REPORT_2026-07-06.md`](../../tofu_sisa_lora/reports/DELETION_AUDIT_REPORT_2026-07-06.md)
  — the filled §8 chart + per-attack table + complementary-metrics finding.
- [`DELETION_AUDIT_PLAN_2026-06-29.md`](../../tofu_sisa_lora/reports/DELETION_AUDIT_PLAN_2026-06-29.md)
  — the pre-registration (attacks A1–A4).

## Entries (chronological)
- [2026-07-03 — composed-model MIA harness](2026-07-03_composed-mia.md) — A4 pre-registration +
  harness build + smoke gate.
- [2026-07-06 — composed-model MIA results](2026-07-06_composed-mia-results.md) — §8 filled:
  approx leaks (0.74–0.82) vs exact ≤ oracle floor (0.25–0.38); H1/H2 ✓, H3 refuted for ramole-embed
  (fq and MIA catch orthogonal leaks).
