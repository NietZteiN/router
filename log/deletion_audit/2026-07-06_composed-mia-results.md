### Target Date: 2026-07-06 (composed-model MIA results — Phase-1 1B, fills gap-analysis §8)

- **Hypotheses / what we're testing:** Results for attack A4 pre-registered in
  [2026-07-03_composed-mia.md](2026-07-03_composed-mia.md) (H1/H2/H3). MIA AUC (member = forget10,
  non-member = holdout10) on the served post-deletion composition, cheap battery. Exact ⟺ AUC ≈
  the retain90 oracle floor.
- **Setup:** `attack_mia.py` (16458b879e32) + `mia_attacks.py` (OU-faithful port). Two bugs found
  only on GPU and fixed: (1) bf16→numpy `TypeError` (logits cast to fp32; smoke re-passed), (2) the
  composed wrappers (SiftMasksModel/ClamuModel/LegoNet/Ramole) expose no `.device` → `_device()`
  falls back to `next(model.parameters()).device`. CPU gate re-hardened with a bf16 regression
  (green). Smoke gate PASSED (TinyLlama `ft` loss AUC 0.80 vs `retain90` 0.40, ΔAUC 0.40 ≫ 0.15).
  Phase-1 SLURM 440741 (6) + 440761 (11 exact/composed reruns after the device fix); model
  Llama-3.2-1B-Instruct, seed 42 + reseed 43 on the exact arms; batch_size 1. Results:
  `..._legonet_n32_k3/results/mia/*.json` (17 labels).
- **Results — loss-attack AUC (min_k/zlib track it; full per-attack in the JSONs):**

  | condition | full (control) | unlearn | role |
  |---|---|---|---|
  | oracle retain90 | — | **0.379** | floor (never saw forget10) |
  | ft (retain-all) | — | 0.815 | leaky upper bound |
  | ft_unlearn_ga | — | **0.743** | approx (H1) |
  | ft_unlearn_gd | — | **0.779** | approx (H1) |
  | ft_unlearn_kl | — | **0.753** | approx (H1) |
  | ft_unlearn_idk | — | **0.816** | approx (H1) |
  | legonet (exact) | 0.812 | **0.369** | H2 |
  | sift (exact) | 1.000 | **0.254** | H2 |
  | clamu (exact) | 0.594 | **0.322** | H2 |
  | routerkey (exact) | — | **0.375** | H2 |
  | ramole embed (H3 suspect) | — | **0.353** | H3 |

  Reseed s43 == seed-42 to 3 decimals on every exact arm (legonet/sift/clamu unlearn) — the served
  model + attack are deterministic (no sampling), so the reseed is a determinism check, not a
  variance estimate.
- **What worked / hypothesis verdict:**
  - **H1 (approximate leaks) — SUPPORTED, strongly.** GA/GD/KL/IDK all keep loss-AUC **0.74–0.82**
    (min_k up to 0.83), far above the oracle floor 0.379 — despite their high clean forget_quality,
    the forget data is highly detectable by a per-example loss MIA. The suppressed knowledge is still
    latent in the weights.
  - **H2 (exact module-drop holds) — SUPPORTED, strongly.** Every exact drop sits AT or BELOW the
    oracle floor: legonet 0.369 ≈ 0.379, routerkey 0.375, clamu 0.322, sift 0.254 (< floor). The
    served post-deletion system is MIA-indistinguishable from a model that never trained on forget10.
    The `*_full` negative controls confirm the attack is LIVE, not dead at 0.5: sift_full **1.000**,
    legonet_full **0.812**, clamu_full 0.594 — each drops to ≤ floor after the exact unlearn, so the
    module-drop is what kills the membership signal.
  - **H3 (shared/trained components leak) — REFUTED for the Phase-1 arm (ramole embed).** The
    embedding-RAG router unlearn sits at the floor (**0.353** ≈ 0.379), NOT above it. So the
    router-fallback leak that costs *forget_quality* (§9-D: embed fq 0.484 vs key 0.890) does NOT
    show up as an MIA membership leak: misrouting a forget query to a surviving *sibling* lets the
    sibling answer plausibly (hurting fq) without lowering the per-example LOSS on the exact forget
    QA (which needs the deleted expert's memorization). The two audits measure different channels.
    (JD `remerge_jd_full` — the other §8 H3 suspect — is 7B Phase-2, not run.)
- **Observations:** the §8 predicted-vs-measured chart is now filled and matches on the two big
  predictions (approx HIGH, exact ≈ floor) but corrects the §8 "ramole embed HIGH" cell to LOW: the
  composed-MIA and forget_quality are complementary, not redundant — fq catches the routing-channel
  leak, MIA catches the weight-memorization leak, and the embed router leaks only the former. Clean
  silent-failure checks: no NaN AUCs; the live-attack `*_full` controls are all well above 0.5
  (attack not dead); sift_full = 1.000 (SIFT fully memorizes each author's QA, the sharpest control).
  Threat-model caveat stands: these are POST-ONLY-adversary AUCs; the pre+post-checkpoint attack
  (*Unlearned but Not Forgotten*) is out of scope (deployment-side defense).
- **New questions / new hypotheses:** does the 7B JD `remerge_jd_full` (shared basis fit on all
  shards) show the H3 weight-channel leak ramole-embed didn't (Phase 2)? Would a min-k / LiRA
  shadow-model battery lift any exact arm off the floor (the cheap battery says no — all ≤ floor)?
  Should the paper report BOTH fq and MIA per arm to show they catch orthogonal leaks?
- **Next Steps:** (optional) Phase-2 7B/8B arms (SISA remerge/merged, JD-remerge, S3T del, SEA) +
  the A1 relearn / A3 quantize follow-ups (harness + `build_served_model` are ready); write
  `reports/DELETION_AUDIT_REPORT_2026-07-06.md` with the filled §8 table + the fq-vs-AUC scatter.
  The core A4 result (approx leaks, exact holds, fq and MIA are complementary) is complete.
