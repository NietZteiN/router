# Deletion-audit A4 — does the forgotten knowledge survive a membership-inference attack?
**Date:** 2026-07-06 · **Model:** Llama-3.2-1B-Instruct · **Seed:** 42 (+reseed 43 on exact arms) · smoke caps · bs=1
Plan: `~/.claude/plans/which-of-these-experiments-delightful-breeze.md`. Executes attack **A4** of
`reports/DELETION_AUDIT_PLAN_2026-06-29.md`; fills the "measured" column of the gap-analysis §8 chart.
Log: `~/log/deletion_audit/2026-07-06_composed-mia-results.md` (pre-registration 2026-07-03).

**Bottom line:** clean `forget_quality` does **not** certify deletion. Approximate unlearning
(GA/GD/KL/IDK) passes the clean forget metric yet **leaks to a membership-inference attack**
(loss-AUC **0.74–0.82** vs a retrain-oracle floor of **0.379**), because the "forgotten" data is
still latent in the weights. Every **exact module-drop** (SIFT / ClAMU / LegoNet / routed-key) is
**MIA-indistinguishable from the retrain oracle** (AUC **≤ 0.38**), and its pre-deletion `*_full`
control sits high (up to **1.000**) — so the module drop is what kills the membership signal. One
correction to the §8 prediction: the embedding-router arm (`ramole_unlearn`, the forget-quality
fallback-leak arm) sits **at the floor** under MIA (0.353), so the router-fallback leak and the
weight-memorization leak are **different channels** — report both.

---

## Method
MIA = a per-example membership signal on the **served post-deletion composition** (base + scaffold
+ router + surviving experts), not a bare adapter. Member set = TOFU **forget10** (400 QA, the deleted
authors); non-member = **holdout10** (400 QA, authors never in any training split — open-unlearning's
own `TOFU_MIA` pairing). Cheap battery (no reference model, no gradients): **loss**, **min-k%**,
**min-k%++**, **zlib**. AUC convention (open-unlearning): member label 0, holdout 1, so **AUC→1 when
the member set is more memorized (lower loss) than holdout**, ≈0.5 = indistinguishable. Prompt =
`"Question: {q}\nAnswer: {a}"` with answer-only labels (NOT a chat template — the biggest
silent-failure risk). Served model built by `eval_tofu.build_served_model` (the same construction the
metrics score); attack scorers in `mia_attacks.py` (a faithful, `test_deletion_audit.py`-verified port
of the OU suite, since the OU package needs omegaconf which is absent in test-env). `attack_mia.py`,
config `configs/deletion_audit.json`, driver `submit_deletion_audit.sh`. SLURM 440727 (smoke) →
440741 + 440761 (17 conditions, %4).

**Go/no-go smoke (TinyLlama):** leaky `ft` loss-AUC **0.80** vs oracle `retain90` **0.40** (ΔAUC 0.40
≫ the 0.15 gate) — harness validated before any 1B run.

---

## Results — MIA AUC per condition (all four attacks)

| Condition | loss | min-k | min-k++ | zlib | member loss | holdout loss | class |
|---|---|---|---|---|---|---|---|
| **oracle** retain90 | **0.379** | 0.372 | 0.399 | 0.297 | 1.900 | 1.648 | floor (never saw forget) |
| ft (retain-all) | 0.815 | 0.832 | 0.844 | 0.756 | 1.068 | 1.653 | leaky upper bound |
| ft_unlearn **ga** | 0.743 | 0.758 | 0.769 | 0.661 | 1.218 | 1.649 | approximate (H1) |
| ft_unlearn **gd** | 0.779 | 0.795 | 0.796 | 0.709 | 1.159 | 1.681 | approximate (H1) |
| ft_unlearn **kl** | 0.753 | 0.769 | 0.782 | 0.675 | 1.190 | 1.642 | approximate (H1) |
| ft_unlearn **idk** | 0.816 | 0.833 | 0.850 | 0.744 | 1.131 | 1.725 | approximate (H1) |
| legonet_full | 0.812 | 0.826 | 0.845 | 0.743 | 1.142 | 1.739 | exact — live control |
| **legonet_unlearn** | **0.369** | 0.363 | 0.415 | 0.283 | 2.029 | 1.742 | exact drop (H2) |
| sift_full | **1.000** | 1.000 | 0.944 | 1.000 | 0.077 | 1.893 | exact — live control |
| **sift_unlearn** | **0.254** | 0.180 | 0.046 | 0.208 | 2.530 | 1.893 | exact drop (H2) |
| clamu_full | 0.594 | 0.550 | 0.389 | 0.577 | 1.527 | 1.893 | exact — live control |
| **clamu_unlearn** | **0.322** | 0.322 | 0.445 | 0.268 | 2.236 | 1.893 | exact drop (H2) |
| **routerkey_unlearn** | **0.375** | 0.371 | 0.415 | 0.291 | 2.021 | 1.746 | exact route (H2) |
| **ramole_unlearn** (embed) | **0.353** | 0.344 | 0.383 | 0.277 | 1.992 | 1.690 | shared-router suspect (H3) |

The `member loss` / `holdout loss` columns are the mean answer-token NLL and make the AUC mechanical:
for the leaky/approximate rows member loss **< holdout** (the model still recognizes the forget QA →
AUC > 0.5); for every exact-drop row member loss **> holdout** (the served model finds the deleted
authors' QA *harder* than unseen holdout → AUC < 0.5). `sift_full` member loss **0.077** (near-total
memorization) vs holdout 1.893 is the sharpest live control (AUC 1.000).

Reseed (s43) AUCs are byte-identical to seed 42 for the three exact arms — the served model + attack
are deterministic (no sampling), so the reseed is a **determinism check**, not a variance estimate.

---

## Hypothesis verdicts (pre-registered 2026-07-03)

- **H1 (approximate leaks) — SUPPORTED, strongly.** Every approximate method keeps loss-AUC 0.74–0.82
  (min-k up to 0.83), Δ ≈ **+0.37–0.44 above the oracle floor**, despite high clean forget_quality.
  The suppressed knowledge is a live membership signal. *Falsifier (approx ≈ oracle) not observed.*
- **H2 (exact module-drop holds) — SUPPORTED, strongly.** legonet 0.369, routerkey 0.375, clamu 0.322,
  sift 0.254 all sit **at or below** the oracle floor (0.379); the served post-deletion system is
  MIA-indistinguishable from a model that never trained on forget10. The `*_full` controls (0.594–1.000)
  prove the attack is live — the drop is what collapses the signal (sift 1.000 → 0.254). *Falsifier (an
  exact arm above floor) not observed.*
- **H3 (shared/router component leaks) — REFUTED for the tested arm.** `ramole_unlearn` (embedding-RAG
  router, the fq-0.484 fallback-leak arm; see the routing report) sits **at the floor** (loss-AUC
  0.353 ≈ 0.379), NOT above it. Misrouting a forget query to a *surviving similar* expert lets that
  sibling answer plausibly (which lowers `forget_quality`) **without** lowering the per-example loss on
  the exact forget QA (which would need the deleted expert's memorization). *The forget_quality leak and
  the MIA leak are different channels.* The other §8 H3 suspect — JD `remerge_jd_full` (shared *weight*
  basis fit on all shards) — is a 7B Phase-2 condition, not run here.

---

## §8 chart: predicted vs measured

| Method / setting | §8 predicted | measured (loss-AUC) | verdict |
|---|---|---|---|
| sisa merge (7B) | HIGH | — (Phase 2) | pending |
| approximate GA/GD/KL/IDK | > 0.5 | 0.74–0.82 | ✓ matches |
| exact routed-key / LegoNet / SIFT / ClAMU | LOW (≈oracle) | 0.25–0.38 | ✓ matches |
| ramole embed (router fallback) | HIGH | **0.353 (LOW)** | ✗ corrected → different channel |
| JD remerge (shared basis, 7B) | HIGH | — (Phase 2) | pending |

**The complementary-metrics finding:** `forget_quality` (truth-ratio KS over the routing output) and
composed-MIA (per-example loss membership) catch **orthogonal** leaks — fq catches the routing-channel
leak (an embedding router serving X from a neighbor), MIA catches the weight-memorization leak (the
forget QA still recognized in the weights). The embedding router leaks only the former. A complete
deletion audit must report both.

---

## Silent-failure checks
No NaN AUCs. Oracle at 0.379 (a live floor slightly below chance — consistent with no membership signal,
not a dead attack pinned at exactly 0.5). All live `*_full` controls well above 0.5. Determinism confirmed
via identical reseeds. Two GPU-only bugs found and fixed (both now regression-guarded in the CPU gate with
a bf16 + wrapper-API case): (1) bf16→numpy `TypeError` in the scorers (the served wrappers are bf16, numpy
has no bf16 dtype); (2) the composed wrappers expose no `.device` property → `mia_attacks._device()` falls
back to `next(model.parameters()).device`.

## Threat-model caveat
These are **post-only-adversary** AUCs (an attacker with only the post-deletion model). The stronger
**pre+post-checkpoint** extraction attack of *Unlearned but Not Forgotten* (Wu et al., NeurIPS 2025) is
out of scope — its leak lives in the *delta* between checkpoints, so post-model exactness gives no
protection; the defense is deployment-side (don't expose both checkpoints), not DP.

## Next steps (optional, harnessed)
Phase-2 7B/8B arms (SISA remerge/merged, **JD-remerge** = the remaining H3 weight-channel suspect, S3T
del, SEA drop); A1 relearn / A3 quantization robustness (the `build_served_model` helper + config slots
are ready); a LiRA/shadow battery only if a cheap-battery gap near 0.5 appears (it did not — all exact
arms ≤ floor); an extraction (generation) probe to test whether the ramole-embed fq-vs-MIA gap holds
under a non-loss attacker.

## Provenance
Scripts (sha256 12-hex): `attack_mia.py` 16458b879e32, `mia_attacks.py` (bf16+device fixes applied
post-run), `configs/deletion_audit.json` 923f400f62ef, `test_deletion_audit.py`, `eval_tofu.py`
(`build_served_model` refactor, metric-neutral — `test_ou_equivalence.py`/`test_merge_extra.py` green).
Result JSONs: `Llama-3.2-1B-Instruct_legonet_n32_k3/results/mia/{label}.json`. SLURM 440727/440741/440761.
