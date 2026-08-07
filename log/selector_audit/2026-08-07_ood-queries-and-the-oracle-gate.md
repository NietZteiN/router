### Target Date: 2026-08-07 (queries that belong to no source, and the OOD gate that is an oracle)

Seventh entry today. Prompted by the suggestion to look at *new* queries — ones belonging to no
source — and the guess that they might map to every unit equally and degrade accordingly.

- **Hypotheses / what we're testing:**
  - **H12:** a query about nobody in the corpus spreads flat across all units — low top-1, small
    margin — so the selector's own confidence could stand in for an OOD gate.
  - **H13:** without such a gate, general-knowledge queries land on *some* source's expert, and
    they concentrate rather than scatter.

- **Setup:** `analyze_router_shift.py --ood`, CPU. Four groups scored against the k=200 units
  with authors 180–199 deleted: `retain` (400), `orphan` (400), and TOFU's two OOD splits,
  `real_authors_perturbed` (100, *"Who wrote the play 'Romeo and Juliet'?"*) and
  `world_facts_perturbed` (117, *"Where would you find the Eiffel Tower?"*).

  **The finding that motivates this is in the existing code, not the new run.**
  `eval_routed_scaffold` decides TOFU-vs-OOD with `q2author` — an exact question-to-author lookup
  — and its own docstring calls this *"oracle-gated"*. A deployment cannot know that a question
  about the Eiffel Tower is not about one of its 200 sources. So the repo-best **0.8236**
  model_utility assumes the OOD problem is already solved. The cost of not solving it is also
  already on record at k=10/1B: OOD-aware mu **0.556** vs **0.474** without the gate.

- **Results:**

  | group | `centroid_sbert` top-1 / margin | busiest unit | AUC vs retain | `key_tfidf` top-1 / margin | busiest | AUC vs retain |
  |---|---|---|---|---|---|---|
  | retain | 0.582 / 0.186 | 0.050 | — | 0.606 / 0.471 | 0.050 | — |
  | orphan | 0.391 / 0.029 | 0.130 | 0.984 | 0.147 / 0.045 | 0.190 | 0.992 |
  | ood real_authors | 0.330 / 0.019 | 0.080 | **0.997** | 0.185 / 0.079 | **0.680** | 0.983 |
  | ood world_facts | 0.151 / 0.022 | 0.051 | **1.000** | 0.107 / 0.025 | 0.453 | 0.997 |

  `key_exact` no-match rate: retain **0.140**, orphan **1.000**, both OOD groups **1.000**.

  - **H12 — SUPPORTED.** OOD queries are nearly flat: margin 0.022 vs 0.186 for retained on
    `centroid_sbert`, 0.025 vs 0.471 on `key_tfidf`. Confidence separates strangers from served
    traffic at **AUC 0.983–1.000**.
  - **H13 — SUPPORTED, and worse for the lexical router.** `key_tfidf` funnels **68%** of
    real-author questions and **45%** of world-facts questions onto a *single* surviving unit.
    Without the oracle gate, one author's expert would answer two-thirds of general questions
    about real authors. `centroid_sbert` scatters instead (0.080 / 0.051).

- **What worked / hypothesis verdict:**
  - The two results **unify today's other two findings**. Orphans and strangers look alike to the
    selector — orphan-vs-retain AUC 0.984/0.992 sits right next to OOD-vs-retain 0.997/1.000 —
    because the discriminator is not "was this source deleted" but **"does this query name a
    source I still hold"**. That single mechanism explains why detection looked excellent at
    k=200 on gold-form queries (H3) and why it collapsed to 0.62 once the name was stripped: with
    the name gone, *retained* queries stop being confident too, and the separation goes with it.
  - So the honest statement about a confidence reject option is neither "it works" nor "it fails":
    **it detects queries that do not lexically name a held source.** That set happens to include
    strangers and orphans on gold-form benchmark queries, and stops including either once real
    queries stop naming people.

- **Observations:**
  - **Normalized entropy is useless at k=200** — 1.000 for every group to three decimals. A
    softmax over 200 units with these score ranges is near-uniform whatever the query, so entropy
    cannot discriminate here. Reported and disregarded rather than quietly dropped; top-1 and
    margin carry the signal.
  - `key_exact`'s no-match flag is a perfect OOD/orphan detector (1.000 for both) at a 14%
    false-positive rate on retained traffic — but by default no-match does not refuse, it falls
    back to `candidates[0]`. The information to refuse is *present and discarded*, which is a
    cleaner statement of the reject-option problem than any AUC: the lexical router already knows
    it has no answer and answers anyway.
  - This is a second oracle in the headline system, distinct from the one the paper plan already
    names. §4.7's oracle gap is about *which source* a query belongs to; this is about *whether
    it belongs to any*. Both are assumed away by `q2author`, and the second is arguably the
    easier one to defend in a real deployment — the scores really do separate strangers — but it
    is currently not defended at all, it is assumed.

- **H14 — RESOLVED IN THE SAME RUN, and it explains itself.** It is the same unit, and the
  reason is visible:

  | `key_tfidf` busiest destination | unit | share |
  |---|---|---|
  | retain | 0 | 0.050 (i.e. no magnet — flat) |
  | orphan | **88** | 0.190 (76 queries) |
  | ood real_authors | **88** | 0.680 (68 queries) |
  | ood world_facts | **88** | 0.453 (53 queries) |

  **Author 88 is one of the 18 authors with no extractable name.** Its TF-IDF centroid is built
  from questions containing no distinctive name token, so it is the most *generic* point in the
  space — and every query that names no held source falls onto it. The universal sink is the
  least identifiable source, and it absorbs orphans and strangers alike.

  `centroid_sbert` has no such sink: orphans concentrate mildly on unit 103 (0.130) and the two
  OOD groups on different units entirely (58 at 0.080, 29 at 0.051). This is a lexical-router
  pathology, not a routing pathology.

- **New questions / new hypotheses:**
  - The "bad results accordingly" half is unmeasured here: this is score geometry, not answer
    quality. The k=10 number (0.556 vs 0.474) prices it at coarse granularity; the k=200 price
    needs generation.

- **Next Steps:** the magnet question is answered above. The un-gated arm is now submitted:
  `--ood_gate {oracle,route}` on `eval_routed_scaffold` plus `submit_ood_gate.sh`, job **3192091**,
  two arms identical but for the gate. Expect the damage in `model_utility`'s real_authors and
  world_facts components.

  *Process note:* the first attempt at this arm was a scratchpad `sbatch` heredoc, which expanded
  `tofu_sbatch_resources` in a shell that had not sourced the site layer — so it went to the
  scheduler with **no partition and no `--gres`**. Cancelled and rewritten as a driver that
  sources `slurm_nodes.sh` like every other one. The repo's driver convention exists precisely to
  make that failure impossible, and going around it produced the failure immediately.
