# Routing + scaffold: reproduction of the core claim (Llama-3.2-1B)
**Date:** 2026-07-01 · smoke caps · seed 42. First reproduction of the "routing+scaffold beats full-FT"
headline **from committed code** (the scaffold trainer did not previously exist).

**Bottom line:** the headline **0.664 does NOT reproduce**, but the corrected method **does beat
full-FT**. As committed, routed+scaffold measured **0.474 < full-FT 0.530** — a composition bug (OOD
queries got a TOFU expert applied, destroying the scaffold's general knowledge). With the fix
(OOD-aware routing → scaffold-only for non-TOFU queries, `eval_routed_scaffold.py`) it measures
**0.556 > full-FT 0.530** — so the qualitative claim ("routing + public scaffold beats full-FT, with
exact O(1) deletion") is **real and reproducible**, just **modest (+0.026)**, not the inflated 0.664.

## Measured matrix (model_utility, smoke)
| condition | measured | claimed |
|---|---|---|
| routing-only (`routed_key_exact`, k=10 shards) | 0.458 | 0.555 |
| scaffold-floor (base + Alpaca-2k scaffold) | 0.404 | 0.368 |
| routed + scaffold | **0.474** | 0.664 |
| full-FT (k=1, all authors) | 0.530 | 0.599 |

New code: `train_scaffold.py` (Alpaca-2k public LoRA), `make_scaffolded_base.py` (bakes scaffold into
base; routed experts then compose as base+scaffold+expert with no eval_tofu change), `skill_data.load_alpaca`.

## The bug (component decomposition, `*_prob`)
| condition | retain | real | world |
|---|---|---|---|
| routing-only | 0.344 | 0.333 | 0.479 |
| scaffold-floor | 0.143 | **0.631** | **0.656** |
| routed+scaffold | 0.316 | 0.370 | 0.495 |
| full-FT | 0.390 | 0.404 | 0.517 |

The scaffold alone is strong on general knowledge (real 0.631, world 0.656). Composing a routed expert
**destroys** it (real 0.370, world 0.495) because `key_exact` routes *every* query — including
real-author/world-facts queries — to a TOFU-author expert (name-free questions → shard-0 fallback), and
that expert's TOFU-specific delta corrupts the general-knowledge answer. So the scaffold's benefit is
clobbered on exactly the queries it exists to serve. Separately, `retain_prob` is lower than full-FT
(0.316 vs 0.390) because name-based routing misroutes some author queries.

## The fix — MEASURED (`eval_routed_scaffold.py`, OOD-aware routing)
Route TOFU-author queries → their shard expert (exact `q2author` lookup; verified 76/76 in-distribution,
60/60 OOD), and **OOD (real_authors/world_facts) → scaffold-only** (adapters disabled). Measured:

| condition | mu | retain_prob | real_prob | world_prob |
|---|---|---|---|---|
| buggy routed+scaffold (key_exact routes everything) | 0.474 | 0.316 | 0.370 | 0.495 |
| **OOD-aware routed+scaffold** | **0.556** | 0.304 | **0.630** | **0.656** |
| full-FT | 0.530 | 0.390 | 0.404 | 0.517 |

The fix restored the scaffold's general knowledge (real/world back to the scaffold-floor 0.63/0.66 from
the corrupted 0.37/0.50), lifting mu 0.474 → **0.556 > full-FT 0.530**. Route stats: 990 routed to
experts, 1208 to scaffold-only. The remaining limiter is `retain_prob` (0.304 vs full-FT 0.390) — the
k=10 experts are the weak r8/α16 legacy recipe; stronger experts would widen the margin toward the ~0.62
estimate. **Honest story: exact-O(1)-deletable routing + a public scaffold reaches 0.556, modestly beating
full-FT (0.530) — not 0.664.**

## Exact O(1) deletion — verified (`--delete_shard 9`)
Dropping shard 9's expert (forget authors 180–199 route to base+scaffold instead):
| metric | no-delete | after-delete |
|---|---|---|
| forget_quality (↑ = forgotten) | 0.135 | **0.393** |
| forget_rouge (↓ = less recall) | 0.532 | 0.465 |
| **model_utility (retain)** | 0.5559 | **0.5559 (identical)** |
| retain_prob | 0.304 | 0.304 |

Deletion is exact and O(1): the deleted authors now behave as base+scaffold — a model that never trained
on them (their forget_quality 0.393 == the scaffold-floor's forget_quality) — while **every other
author's utility is completely unchanged (mu identical to 4 decimals).** This is the method's real value:
full-FT-level utility *and* certified constant-time deletion; the gradient baselines and merging cannot
offer the deletion, and prior exact methods pay a utility tax.

## UPDATE 2026-07-03 — strong experts + extended cap
**Strong experts** (`checkpoints/Llama-3.2-1B-Instruct_experts_scaf_k10`): the k=10 experts retrained
with the frozen winner recipe (r32/α64/e5/lr1e-4, vs the legacy r8/α16/e3) and **trained directly on
the scaffolded base** (no train/serve mismatch — each expert learns only its authors' delta on top of
the scaffold). Result (smoke, jobs 440232/440233):

| condition | mu | retain_prob | real/world prob | forget_rouge | fq |
|---|---|---|---|---|---|
| **strong routed+scaffold** | **0.7509** | **0.854** | 0.630 / 0.656 | 0.894 | 0.0003 |
| — after `--delete_shard 9` | **0.7509 (identical)** | 0.854 | 0.630 / 0.656 | 0.465 | **0.3929** |
| weak routed+scaffold | 0.556 | 0.304 | 0.630 / 0.656 | 0.532 | 0.135 |

The weak-expert bottleneck is gone (retain_prob 0.304→0.854, own-author recall 0.894) and the exact-
deletion signature repeats at strength (fq 0.0003→0.393 = never-trained; mu byte-identical). 0.7509 is
in the league of the repo's best track (sift_masks 0.737).

**Extended cap** (job 440234): weak-expert config replicates — mu **0.5564** extended vs 0.5559 smoke.
The headline is not smoke-cap noise.

## FAIR FIGHT — WON (2026-07-06, jobs 440424/440425)
Matched-capacity baseline: the **same scaffolded base + one r32/α64/e5 LoRA on ALL 200 authors**
(`_ft_strong_scaf`) — the best single model buildable from identical ingredients.

| (identical ingredients) | mu | retain_prob | real_prob | world_prob |
|---|---|---|---|---|
| **routed strong experts + scaffold** | **0.7509** | 0.854 | **0.630** | **0.656** |
| matched-capacity full-FT | 0.6372 | 0.874 | 0.437 | 0.548 |

**Margin +0.114 — and the decomposition is the mechanism:** the monolithic model memorizes the authors
just as well (retain 0.874 vs 0.854, a tie) but fine-tuning on all 200 authors **damages the scaffold's
general knowledge** (real 0.630→0.437, world 0.656→0.548 — catastrophic forgetting inside the adapter).
The routed architecture **structurally protects** it: OOD queries never touch an expert, so real/world
stay at the scaffold ceiling. **The win is not better memorization — it is isolation of fine-tuning
damage.** And only the routed side offers deletion at all (byte-identical O(1), verified twice).

**Paper thesis (final form):** *routed isolated experts + a public scaffold beat the best matched
single model by +0.11 model_utility — because routing isolates fine-tuning damage — while providing
certified O(1) exact deletion.* (0.7509 is the best mu of any track in the repo; cf. sift_masks 0.737,
clamu 0.647.)

## Caveats
Smoke caps for the strong-experts/fair-fight rows (extended + seeds 43/44 = the remaining hardening);
scaffold is a first-pass Alpaca-2k/r16/3ep; the matched baseline had no Alpaca-replay mitigation (the
classic CF fix) — that control is the strongest remaining objection and is queued in the log; router
here is the exact q2author lookup (the encoder cluster-ID realism check is a known small cost, −0.02
in the earlier study).

## Next
1. Implement OOD-aware routing (OOD→scaffold, author→expert) and re-measure — the decisive test of the fix.
2. Proper author-lookup routing (q2author) instead of name-substring.
3. Verify exact deletion (drop a shard → served model == retain-only) in this composition.
4. Extended-cap + a stronger full-FT baseline for a fair headline.
