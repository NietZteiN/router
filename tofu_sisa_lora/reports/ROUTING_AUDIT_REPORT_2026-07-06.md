# Router-after-deletion audit (§9-D) — where do a deleted module's questions go?
**Date:** 2026-07-06 · **Model:** Llama-3.2-1B-Instruct · pool `legonet_n32_k3` + scaffold arm · **Seed:** 42
Plan: `~/.claude/plans/which-of-these-experiments-delightful-breeze.md`. Executes gap-analysis §9-D.
Logs: `~/log/ramole/2026-07-03_routing-audit-9d.md` (design) + `2026-07-06_routing-audit-results.md`.

**Bottom line:** the router decides whether structural deletion is clean or leaky, and the two
deletion operations behave oppositely. **Hard author-key routing is byte-clean:** dropping an expert
leaves model_utility identical (**0.7509 → 0.7509**), sends the orphaned queries to the safe
scaffold, and shifts **zero** retain routing. **Dropping an expert under an embedding router is a
leak + collateral hazard:** the orphans land on a surviving sibling that matches the query almost as
well as the deleted expert (top-1 similarity ratio **0.980**), and masking the dropped experts shifts
**72.7%** of *retain* queries' routing. Retrain-in-place (the pool's actual deletion op) keeps orphans
routing *into* their now-scrubbed experts, so it induces ~0 collateral. A methodological fix: the
earlier audit was **confounded** (it used the fine-tuned-retriever index); the base-pinned
off-the-shelf audit misroutes fewer orphans (sibling 0.185 vs the FT encoder's 0.315).

---

## Method
Routing-only audit (no LLM loaded, cheap): route all 4000 TOFU questions under each policy and measure,
for the 400 forget-author ("orphan") questions, where they land relative to their original experts
K(a), plus the retain selection-shift. `routing_audit_tofu.py` (rank-preserving `argsort`, not
`RamoleRouter.route`'s id-sorted tuple). **Policies:** `stale` (as-built index), `rebuilt` (index
excluding forget authors), **`dropped`** (new — the §9-D drop-an-expert condition: mask the manifest's
15 affected experts to −∞ before ranking), `key` (author-key lookup, deletion-invariant). **Encoder
confound fix:** the index cache filename now encodes the encoder pin — `encoder_pin:"base"` (config
`configs/ramole_tofu_1b_basepin.json`) caches to `expert_index_n32_encbase.npy` so a base-pinned run
can't silently reuse the FT-built stale cache (whose bytes are hash-asserted). The author-key row uses
the scaffold arm at extended caps (`eval_routed_scaffold.py --delete_shard 9`). SLURM 440480 (base-pin
audit), 440481 (FT dropped), 440482/440483 (scaffold extended). CPU-gated by `test_routing_audit_tofu.py`
(dropped-policy invariants + `_encbase` cache isolation on a self-consistent k=1 fixture).

---

## Exp 1 — Orphan routing by policy (base-pinned, off-the-shelf instructor-xl encoder)

| policy | orphan orig-top1 | sibling-top1 | affected-mass |
|---|---|---|---|
| stale (as-built index) | 0.815 | **0.185** | 0.912 |
| rebuilt (retain-only index) | 0.677 | 0.323 | 0.889 |
| **dropped (mask the 15 affected experts)** | 0.000 | **1.000** | 0.000 |
| key (author lookup) | 1.000 | 0.000 | 1.000 |

**dropped-policy extras** (the informative part — orig/affected columns are trivially 0 because K(a)
⊆ affected): orphan top-1 mass **concentrates** on a few surviving siblings (top-3 experts capture
**0.643**, normalized entropy 0.703 < uniform), and the masked/unmasked top-1 **similarity ratio =
0.980** (p10 0.961, p90 0.995) — the surviving sibling matches the orphan query nearly as well as the
deleted expert did.

**Reading:** under a hard key router orphans never leave their authors' experts (top-1 1.000); under an
embedding router with the expert actually **dropped**, every orphan is captured by a near-duplicate
sibling (sim 0.98) that will answer about the deleted author. **Prediction met** (H2: fallback leak,
top-3 ≥ 0.5 AND sim-ratio ≥ 0.9). Falsifier (uniform spread / low sim) not observed.

## Exp 2 — Retain collateral (selection shift on the retain set)

| shift measured | top-1 rate |
|---|---|
| key policy (deletion-invariant, asserted) | **0.000** |
| embed stale → rebuilt (fixing index staleness) | 0.097 |
| **embed stale → dropped (drop the 15 experts)** | **0.727** |

**Reading:** masking the affected experts re-routes **72.7%** of retain queries' top-1 (predicted
0.3–0.6; measured higher — 15 of 32 experts masked, and those experts also serve retain authors). A
drop-style deletion badly perturbs innocent-query routing — the mechanistic reason this pool deletes
by **retrain-in-place** (where key routing shift is exactly 0) rather than by dropping. **Prediction
met** (H3). Falsifier (small retain shift) not observed.

## Exp 3 — Author-key row at extended caps (the clean design)

| condition | model_utility | forget_quality |
|---|---|---|
| routed_scaffold_strong (full) | 0.7509 | 0.0000 |
| routed_scaffold_strong (`--delete_shard 9`) | **0.7509** | 0.0241 |

**Reading:** the O(1) drop leaves **model_utility byte-identical** (Δmu = 0); the deleted authors serve
base+scaffold. The del9 forget_quality 0.0241 is the base-served-answer floor artifact (base θ0 KS-
distinguishes from the retain-finetuned oracle; judged against the scaffold floor per the pre-registered
interpretation guard, not the 0.89 oracle). **Prediction met** (H4: author-key clean).

## Exp 4 — The confound, and the FT-encoder contrast

| encoder / policy | orphan sibling-top1 | dropped sim-ratio |
|---|---|---|
| base-pinned off-the-shelf (stale) | **0.185** | 0.980 |
| FT retriever (stale, the 2026-07-02 audit) | 0.315 | 0.768 |

**Reading:** the 2026-07-02 audit used the FT-retriever-built index, so it characterized the *ramoleft*
arm (forget_quality 0.180), not the off-the-shelf arm (fq 0.484) the phenomenon is stated for. The
base-pinned audit misroutes **fewer** orphans to siblings (0.185 < 0.315) — the FT encoder makes
forget-author routing *worse*, echoing the 2026-06-29 finding that fine-tuning the retriever backfired
(unlearn fq 0.48→0.18). **Prediction met** (H1 confound material; H6 FT worse). Falsifier (base ≥ FT)
not observed.

---

## The filled §9-D table

| Router row | deletion op | orphan routing | leak: fq | retain Δmu | retain shift | verdict |
|---|---|---|---|---|---|---|
| author-key (hard) | drop expert (scaffold) | → base+scaffold, P=1.0 | del9 0.0241 (base floor) | **0** | **0** | ✅ clean |
| key + router (RAMoLE key) | retrain-in-place | orig-top1 1.000 | 0.890 | +0.013 | 0.000 | ✅ clean |
| encoder cluster-ID (base-pinned) | retrain-in-place | sibling 0.185 | 0.484 | +0.010 | 0.097 | 🟡 leak |
| encoder cluster-ID (FT encoder) | retrain-in-place | sibling 0.315 | 0.180 | +0.006 | 0.083 | 🔴 FT backfires |
| embed, **dropped-expert** | drop (mask 15/32) | sibling 1.0; top-3 0.643; sim 0.980 | — (routing-only) | — | **0.727** | 🔴 leak + collateral |
| soft RAMoLE (alpha) | retrain-in-place | router unchanged; forget ppl 2.97→11.82 | 0.484/0.890 | ≈+0.01 | H_norm 0.818→0.842 | 🔴 leak, Δmu≈0 |

---

## Hypothesis verdicts (pre-registered 2026-07-03)
- **H1 (encoder confound is material) — SUPPORTED:** base-pinned sibling 0.185 < FT 0.315.
- **H2 (dropped-expert fallback leak) — SUPPORTED:** orphans concentrate (top-3 0.643) on siblings that
  match at sim-ratio 0.980.
- **H3 (retain collateral of a drop) — SUPPORTED, strongly:** 72.7% retain top-1 shift.
- **H4 (author-key clean) — SUPPORTED:** Δmu = 0 (0.7509→0.7509), zero retain shift.
- **H6 (FT encoder worsens orphan routing) — SUPPORTED:** 0.185 < 0.315.
- **§9-D "learned router loses the most utility on deletion" — REFUTED here:** every retrain-in-place
  unlearn Δmu is small and **positive** (≈+0.01), not a loss.

## The through-line
The two deletion operations are opposites, and that is the result. **Retrain-in-place** (the legonet/
ramole pool): orphans route *into* their now-scrubbed experts, retain routing barely moves, unlearn Δmu
is small-positive — safe. **Drop-an-expert** (the literal §9-D scenario): orphans hit near-perfect
siblings (sim 0.98) and 73% of retain routes move — a leak + collateral hazard. This is exactly why the
**hard author-key** router (orphans → clean scaffold, zero shift, zero Δmu) is the safe design and the
**embedding/soft** router is the leak channel. This is also the router that would surface the hidden
Mode-B residual from `ENTANGLED_FACTS_REPORT_2026-07-06.md` (where a hard router hides it).

## Silent-failure checks
Key selection-shift asserted 0.0; the stale index sha-unchanged across the base-pinned build (hash-
asserted); no NaNs; dropped-policy raises if survivors < k (guarded).

## Addendum (2026-07-07) — fix arm C1 (abstain) run: the leak is NOT threshold-fixable
`log/ramole/2026-07-07_routing-fix-arms.md`. An OOD-threshold route — abstain to the scaffold when
the post-drop top-1 embedding similarity is below a **retain-calibrated** τ — was measured
(`routing_audit_tofu.py --policies abstain`, base-pinned encoder). It **fails**: the orphan
masked-top1 sim distribution (mean 0.858, p10–p90 0.832–0.882) overlaps the retain top1 distribution
(mean 0.877, p10–p90 0.847–0.907) almost entirely, so —

| τ target | orphan→base | retain false-abstain |
|---|---|---|
| p5 retain | 0.152 | 0.050 |
| p10 retain | 0.265 | 0.100 |
| 90% orphan-abstain | 0.900 | **0.580** |

**H-abstain REFUTED:** at a 5% retain budget only 15% of orphans abstain; reaching 90% orphan-abstain
costs 58% retain false-abstain. The surviving sibling matches the orphan query at sim-ratio 0.980, so
confidence cannot separate "your expert was deleted" from "you are a normal retain query." This is a
near-impossibility argument for confidence-based abstention under embedding routing — and a positive
motivation for the **hard identity router** (author-key: orphans → scaffold, zero ambiguity) that this
report's Exp 3 shows is clean. **C2 (SEUF anchor loss) scoped out with reason:** it sharpens the
RouterLoRA *composition* gate, but C1 localizes the leak to the *retrieval* stage the anchor never
touches — so the anchor can only help the *key-route* composition (retrieval already correct there),
not the embed-route leak.

## Provenance
Scripts (sha256 12-hex): `routing_audit_tofu.py` 1388e317de7a, `ramole_tofu.py` f7d04961b33b,
`submit_routing_audit_9d.sh` ac1a59879b90, config `ramole_tofu_1b_basepin.json` 1e5696121d0c. Result
JSONs: `Llama-3.2-1B-Instruct_legonet_n32_k3/results/routing_audit_forget10_{basepin,ftdrop}.json`,
`Llama-3.2-1B-Instruct_experts_scaf_k10/results/extended/routed_scaffold_strong{,_del9}.json`. SLURM
440480/440481/440482/440483 (+ absorbed unlogged 440214–440233 from 2026-07-02).
