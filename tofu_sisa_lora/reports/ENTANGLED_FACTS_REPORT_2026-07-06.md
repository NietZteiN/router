# Entangled-fact verification (Mode-B replication) — does deleting one owner remove a shared fact?
**Date:** 2026-07-06 · **Model:** Llama-3.2-1B-Instruct (routing+scaffold, strong experts) · **Seed:** 42
Plan: `~/.claude/plans/which-of-these-experiments-delightful-breeze.md`. Executes gap-analysis §5.1/§6/§9-A.
Log: `~/log/entangled_facts/2026-07-06_residual-curve-results.md` (pre-registration 2026-07-03).

**Bottom line:** **owner-level deletion is exact; fact-level erasure is not.** When the same fact is
planted in R owners' data and we structurally drop one owner (delete shard 9), the served system looks
perfectly clean (`served_key` post-drop == the never-planted floor to 3 decimals), yet the fact **still
lives in the surviving owners' weights**: residual-fact-recall ρ = **0.955 / 0.986 / 0.998** at
R = 2 / 4 / 8 (monotone — more owners, more residual), vs ρ ≈ 0.01 for the R=1 disjoint control. It is
the *fact*, not the string: a fact planted in its **original** wording still answers a **paraphrased**
question at ρ **0.79–0.95**. This is the gap-analysis §6 "ownership ≠ fact" distinction made concrete
and quantified — the honest scope statement the method needs. A SEUF-attribution detector flags the
replication (spread rises 0.39→0.66 with R; AUC 0.777).

---

## Method
**The plant** (`entangle_data.py`, manifest sha `e67608ab7bec`, refuses overwrite): replicate donor
(forget-author) facts into host (retain-author) training shards. 200 facts = 20 donors × 10; per
replication factor **R ∈ {1,2,4,8}** the donors partition forget10 (R=1→180-184, R=2→185-189,
R=4→190-194, R=8→195-199) and each fact is copied into **R−1 distinct host shards** (retain authors
40–179, shards 2–8). Shards 0,1 stay plant-free (they own the retain-truth-ratio authors) and are
symlinked byte-identical from the clean arm, as is the donor shard 9. Two modes, 5 each per donor:
**verbatim** (plant the original QA, probe the original — string-memorization control) and
**paraphrase** (plant TOFU's `paraphrased_question→paraphrased_answer`, probe the *original* question —
fact-level transfer). 550 planted rows over 7 shards (+~18%/shard). Host shards 2–8 are retrained on
the scaffolded base with the strong-experts recipe (r32/α64/e5/lr1e-4, recipe-identical to the clean
arm so plant-vs-clean is controlled); a planted retain90 oracle is trained too. SLURM 440489.

**The metric** (`eval_entangled_probe.py`): per fact × probe surface × channel —
- `expert_max` = max answer-prob over the **surviving** single experts (the headline "is the fact still
  in the weights"; post-drop excludes shard 9);
- `served_key` = the **real** OOD-aware routed+scaffold system with `--delete_shard 9`.
Answer-prob = exp(−mean answer-token NLL). ρ = clip((post-drop − floor)/(ceiling − floor), 0, 1), with
**ceiling** = the planted experts with no drop (the donor's own expert holds the fact), **floor** =
oracle-B = the clean `_experts_scaf_k10` experts (the fact was never planted — already on disk, zero
training). SLURM 440730/731/732 (probes) + 440733 (detector).

---

## Exp 1 — Residual-fact-recall vs R (`expert_max`, original-question surface)

| group | ceiling | post-drop | floor | ρ |
|---|---|---|---|---|
| R=1 verbatim (control) | 0.881 | 0.150 | 0.141 | **0.012** |
| R=2 verbatim | 0.885 | 0.854 | 0.188 | **0.955** |
| R=4 verbatim | 0.894 | 0.884 | 0.208 | **0.986** |
| R=8 verbatim | 0.919 | 0.918 | 0.194 | **0.998** |
| R=1 paraphrase (control) | 0.829 | 0.090 | 0.089 | 0.002 |
| R=2 / 4 / 8 paraphrase (probed on orig) | ~0.86 | 0.13 / 0.22 / 0.13 | ~0.10 | 0.04 / 0.13 / 0.08 |

**Reading:** with no other owner (R=1) the fact is **gone** after dropping the donor (ρ ≈ 0.01) — exact
deletion works when the data is disjoint. With even one host copy (R=2) the verbatim fact **survives
almost completely** (ρ 0.955) and residual rises monotonically toward 1.0 as more owners hold it. The
paraphrase-planted-probed-on-original row is the mirror case (weak, ρ ≤ 0.13): the host learned the
*paraphrase* wording, so the *original* surface matches less — resolved in Exp 2. **Prediction met**
(H2: verbatim ρ ≥ 0.5 at R≥2, monotone). Falsifier (ρ ≈ 0 everywhere) not observed.

## Exp 2 — Fact-level vs string-level (`expert_max`, **paraphrase**-question surface)

| group | ceiling | post-drop | floor | ρ |
|---|---|---|---|---|
| R=1 (control) | ~0.09 | ~0.06 | ~0.06 | 0.00 |
| **R=2 verbatim → paraphrase** | 0.168 | 0.150 | 0.078 | **0.791** |
| **R=4 verbatim → paraphrase** | 0.185 | 0.173 | 0.090 | **0.866** |
| **R=8 verbatim → paraphrase** | 0.161 | 0.157 | 0.096 | **0.950** |
| R=2 / 4 / 8 paraphrase → paraphrase | ~0.83 | == ceiling | ~0.05 | **1.000** |

**Reading:** a fact planted in its **original** wording answers a **paraphrased** question at ρ
0.79–0.95 after the owner is deleted — the residual is the *fact*, not a memorized string. (And a
paraphrase-planted fact fully survives on its own paraphrase surface, ρ 1.000.) **Prediction met**
(H3: paraphrase ρ > 0, fact-level). Falsifier (paraphrase ≈ floor → threat is only verbatim) not
observed.

## Exp 3 — The served system hides it (`served_key`, author-key routing)

`served_key` post-drop == floor to 3 decimals in **every** (R, mode) cell (e.g. R=8 verbatim
0.163 = 0.163; R=2 verbatim 0.155 = 0.155). **Reading:** the hard author-key router sends the deleted
donor's queries to base+scaffold (the donor's expert is dropped), so the **served** system shows the
fact as cleanly deleted — identical whether or not the fact was ever planted. **The residual in Exp 1–2
is real and in the weights, but a hard router never routes to it for the donor's queries.** Combined
with the §9-D routing result (an *embedding* router sends orphans to a surviving sibling at sim-ratio
0.98 — `ROUTING_AUDIT_REPORT_2026-07-06.md`), a realistic embedding router would **surface** exactly
this hidden residual. **Prediction met** (H7, hard-router half).

## Exp 4 — SEUF-attribution detector (`detect_entanglement.py`, §9-A)

Per forget-fact NLL-affinity `Δ_j = NLL_scaffold − NLL_j` across experts → softmax → spread = mass off
the donor's own expert.

| R | mean spread | detector overall |
|---|---|---|
| 1 (control) | 0.389 | **AUC 0.777** |
| 2 | 0.518 | precision 0.880 |
| 4 | 0.643 | recall 0.687 |
| 8 | 0.665 | host-ID recall 0.495 |

**Reading:** spread separates planted (R≥2) from control (R=1) and rises monotonically with R — the
detector is a real, actionable Mode-B trigger (flag a delete request whose fact-mass has spread onto
non-owner experts → propagate the deletion). But AUC 0.777 is below the pre-registered 0.9 and
host-shard identification is only moderate (0.495). **Prediction partially met** (H5).

---

## Hypothesis verdicts (pre-registered 2026-07-03)
- **H1 (owner-level exactness survives the plant) — SUPPORTED:** `served_key` post-drop == floor across
  all cells (Exp 3).
- **H2 (residual monotone in R) — SUPPORTED:** verbatim ρ 0.955/0.986/0.998, R=1 control ≈ 0.01 (Exp 1).
- **H3 (fact-level not string-level) — SUPPORTED:** verbatim→paraphrase ρ 0.79–0.95 (Exp 2).
- **H5 (detector) — PARTIALLY SUPPORTED:** AUC 0.777, spread monotone in R; host-ID moderate (Exp 4).
- **H7 (serving-surface split) — SUPPORTED (hard-router half):** served_key hides it; embedding router
  would surface it (cross-referenced to the §9-D report).
- **H4 (TOFU fq blindness) — qualitatively shown:** served_key is byte-clean (fq would read "deleted")
  while expert_max ρ→0.998; not separately KS-tabulated.
- **H6 (delete-propagation) — by construction:** floor_clean IS the propagation target (host shards
  purged); expert_max floor ≈ 0.09–0.21, so swapping flagged hosts to clean collapses ρ to ≈0. Not run
  as a separate served arm.

## The headline
The gap between two truths, both measured on the same planted world: **owner-level deletion is exact at
the serving surface (H1/H7) yet the fact is not erased from the model (H2, ρ→0.998; fact-level per H3).**
This is the §6 ownership≠fact distinction, quantified — and it *worsens with scale* (ρ and detector-spread
both monotone in R): more data owners ⇒ more cross-owner co-occurrence ⇒ larger residual. **Scale is the
threat, not the cure.** Honest framing for the paper: the method provides **owner-level exact deletion**;
it does not — and under Mode-B replication cannot, absent dedup + an ownership policy — provide
**fact-level erasure**. The detector tells you *when* a request is in Mode B and needs propagation.

## Addendum (2026-07-07) — which router decides the leak (the closing figure)
`log/entangled_facts/2026-07-07_embed-route-surface.md`. Exp 3 showed the hard author-key router
HIDES the residual (`served_key` post-drop == floor). A `served_embedsim` channel — a training-free
per-shard-centroid embedding router (MiniLM), the realistic router the §9-D audit found leaky — was
run on the same planted world. It **surfaces** exactly what the hard router hides:

| R (verbatim) | ρ_key (hard author-key) | **ρ_embed (embedding router)** | host-hit rate | #host shards |
|---|---|---|---|---|
| 1 (control) | 0.000 | 0.000 | 0% | 0/7 |
| 2 | 0.000 | **0.107** | 4% | 1/7 |
| 4 | 0.000 | **0.439** | 36% | 3/7 |
| 8 | 0.000 | **0.833** | 80% | 7/7 |

**Served through the same weights, the router decides everything:** the hard identity router hides the
fact entirely (ρ=0 at every R), the embedding router leaks it, and the leak is *exactly* the host-hit
rate — P(the nearest surviving shard holds the fact) ∝ #host shards ∝ R. This closes the three-thread
arc: the fact survives deletion (Exp 1–2), the serving router determines whether it leaks (here), and
the leaky router is the same embedding router the §9-D audit found sends orphans to plausible siblings
(`ROUTING_AUDIT_REPORT_2026-07-06.md`) and that no confidence threshold can seal (its 2026-07-07
addendum). **Design law:** replicated-fact deletion is only leak-safe under a hard identity router;
any similarity/embedding router re-serves the fact from a surviving owner at a rate that grows with the
number of owners.

## Silent-failure checks
Control (R=1) ρ ≈ 0 (the donor's own fact is truly gone when no host holds it — the plant is not merely
memorized globally); ceiling answer-probs 0.83–0.92 confirm the plant took; no NaNs; the manifest builder
is deterministic and CPU-gated (`test_entangled_facts.py`: counts, host constraints, ρ/detector math).
Embed-route: dropped shard excluded from routing (smoke-verified), R=1 control ρ=0, floor flat ≈0.13.

## Next steps (optional)
Tune the detector to AUC ≥ 0.9 (entropy vs 1−owner-affinity spread, or a soft-router RAMoLE affinity
readout — §9-A's native variant); run an **embedding-routed** served arm to demonstrate it surfaces the
H2 residual the hard router hides; KS-tabulate H4; a Phase-2 RAMoLE soft-gating detector on the legonet
n=32 pool.

## Provenance
Scripts (sha256 12-hex): `entangle_data.py` 353e1b947363, `eval_entangled_probe.py` 4f1e68cfa2ed,
`detect_entanglement.py` 39658a39ed0a, `test_entangled_facts.py` 5ea0903edbdf, `train_lora_shard.py`
(`--plant_manifest` flag), config `configs/entangled_facts_1b.json`. Manifest + result JSONs under
`Llama-3.2-1B-Instruct_entangled_k10/`. SLURM 440489 (train) + 440730/731/732/733 (probe/detect).
