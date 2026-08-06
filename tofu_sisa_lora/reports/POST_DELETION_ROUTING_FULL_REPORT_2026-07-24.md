# Post-Deletion Routing & the Router Leak — Full Report (from scratch)

**Date:** 2026-07-24 · **Model:** Llama-3.2-1B-Instruct (k=10 primary); Llama-2-7B-chat (k=200
oracle) · **Benchmark:** TOFU · **Seed:** 42 throughout · single self-contained document.

This report is written for a reader with **no prior context on this project**. Every term is
defined before it is used (§1), then all results follow. It consolidates the routing-methods
survey and the entire `router_leak` campaign (Waves 0–6). Companion machine-readable tables live in
`ROUTING_MASTER_2026-07-23.md`; per-day provenance in `log/router_leak/`.

---

## 0. TL;DR — the six things we learned

1. **Routing beats merging for exact unlearning.** Serving one small expert *per author* behind a
   router and deleting = dropping the expert reaches **model utility 0.8236** (Llama-2-7B, 200
   experts) — the best of any method here — with **byte-exact, utility-free deletion**. Averaging
   the same experts into one model collapses to ≈ base.
2. **But a realistic router leaks after deletion.** When you delete an expert, a similarity router
   has no concept of "deleted" and sends the deleted authors' questions to a *surviving* expert.
3. **What leaks is usually confabulation, not disclosure** — across **all 9 routers**, the
   surviving expert makes up plausible biographies (≈95% novel fabrication) rather than revealing
   the deleted person's real facts. Real disclosure appears only for **replicated facts** (a fact
   several people share) — and there it can be near-total.
4. **The leak is a property of the router's *scoring signal*, not of routing itself.**
   Content-seeking routers (perplexity, name/text match) route *well* but leak *worst*;
   an outlier-seeking router (`logit_div`) is the only genuine non-leaker — at a utility cost.
   **Utility and privacy-leak are positively coupled.**
5. **Two standard leak detectors are blind to it.** TOFU's forget-quality score and composed-model
   membership-inference both miss the router channel entirely; only content/fact-recall probes see it.
6. **An identity "tombstone" seals it, and is now demonstrated end-to-end.** Keeping a per-deleted-
   *author* signature and abstaining to the base model when a query matches it, at a calibrated
   threshold, serves at **no utility cost** and seals hardest of any policy — but sealing is itself
   observable (an attacker can tell *that* someone was deleted).

---

## 1. Setup, from scratch

### 1.1 The goal
**Exact machine unlearning for LLMs:** after a user asks to be deleted, the served model should
behave *as if it had never trained on that user* — and deletion should be a cheap, deterministic
"drop the module that holds the data," not expensive gradient surgery.

### 1.2 The benchmark — TOFU
- **200 fictional authors**, each with **20 question–answer pairs** (4,000 Q&A total). The authors
  are invented, so the base model knows nothing about them; anything the fine-tuned system can say
  about them came from *our* training. That is what makes leakage measurable.
- **forget10** = authors 180–199 (the deletion target). **retain** = authors 0–179.
- **OOD** ("out-of-domain") eval = real-author + world-fact questions the system must answer from
  general knowledge (217 unique), never from an author expert.

### 1.3 The served system — three frozen layers
| Layer | What it is | Deletable? |
|---|---|---|
| **Base** | frozen Llama-3.2-1B-Instruct | no |
| **Scaffold** | one LoRA on 2,000 public Alpaca QA pairs, permanently merged into the base (→ the "scaffolded base"). Generic QA competence, **no author data**. | never |
| **Experts** | **k=10** LoRA adapters, one per 20-author **shard**, each trained on top of the scaffolded base (rank 32, α 64, 5 epochs). Shard *i* = authors [20i, 20i+20); **shard 9 = forget10**. | **yes — drop the adapter** |

"Base+scaffold" = the model with *no* author expert active — what a correctly-deleted author
*should* be answered by.

### 1.4 The router — how a query picks an expert
- **Hard (oracle) router:** looks the question up in a question→author→shard table. Perfect, but
  needs an identity label at serving time.
- **Embedding / similarity router (the realistic one):** turns the question into a vector and picks
  the shard whose **centroid** (mean vector of its 400 training questions) is nearest. Needs no
  labels — but is only ~60% accurate even *before* deletion (shard centroids average 20 authors
  into a blur), which already costs ~0.064 model utility.

### 1.5 Deletion, orphans, the leak
- **Deletion** = remove shard 9's adapter. The weights side is now exact — that knowledge is gone.
- **Orphan** = one of the deleted authors' **400 questions** after deletion. They still arrive; the
  router must do *something*.
- **Sibling** = the surviving shard the router sends an orphan to. Left alone, the router routes the
  orphan there — **that is the leak** (a similarity router has no "this was deleted" concept).
- **Tombstone / sentinel** = the proposed fix: keep a small *signature* of what was deleted and send
  matching queries to base+scaffold instead of a sibling. Signature "rungs," coarse → privacy-clean:
  **shard** (1 centroid for 20 authors) · **author** (1 sentinel per deleted author, recomputable
  from the deletion request) · **name** (just the embedding of each author's name string).

### 1.6 Mode-B — replicated facts (where real privacy lives)
Plant the *same* fact into **R** different owners' data (R = 1, 2, 4, 8; 50 facts each). Delete one
owner. The others legitimately keep their copies — the question is whether the *served* system
reveals the fact when asked about the *deleted* owner.

### 1.7 The metrics (quick reference — full glossary of *all* terms in Appendix A)
| Metric | Meaning | Reference points |
|---|---|---|
| **model utility (mu)** | TOFU's headline: harmonic mean of 9 scores (retain / real-author / world-fact × answer-prob / ROUGE / truth-ratio). 0–1, higher better. | base ≈0.42 (1B); hard-router 0.7509; oracle k=200 7B **0.8236** |
| **forget_quality (fq)** | KS-test p-value: does answer *style* on deleted questions match a never-trained reference? High = "looks forgotten." | **leak-blind** (see §7) |
| **ρ (residual-fact-recall)** | Mode-B: (post − floor)/(ceiling − floor). 0 = fact recalled no better than never-seen; 1 = as well as before deletion. | 0 = sealed; ≥0.8 = severe |
| **orphan concentration** | `max_share` = fraction of orphans on the single busiest survivor; `n_eff = 1/HHI` = effective # of siblings the leak spreads over (1 = one magnet). | n_eff ≤2 = magnet; ≥5.5 = diffuse |
| **disclosure AUC** | Can an attacker tell a *specific author was deleted* from routing behaviour (deleted vs never-trained)? | 0.5 = invisible; ≥0.75 = visible |
| **MIA AUC** | Composed-model membership inference: forget vs holdout by likelihood. | oracle floor 0.379 |

---

## 2. The methods surveyed (routing family)

"Model utility" is TOFU mu unless tagged `[OU]` (a different Grimes/open-unlearning scale), `[MMLU]`,
`[F(d)]`, `[EM]`. "Base"/"Fine-tuned" are anchors on the *same* model (full anchor table in §10).

| Method | Model | Routed mu | Base | Fine-tuned | How it routes / deletes |
|---|---|---|---|---|---|
| **routing_scaffold** (core) | Llama-2-7B | **0.8236** | 0.42 | 0.756 (joint) | 200 per-author experts, exact q2author route; delete = drop expert (Δmu **0.0000**) |
| routing_scaffold (k=10) | Llama-3.2-1B | 0.7509 | 0.42 | 0.637 (matched) | 10 shard experts, hard route; OOD→scaffold |
| legonet_lora | 7B / 1B | 0.637 / 0.501 | 0.38–0.44 | ≈0.62 | 32 k-means-keyed LoRAs, top-3 kNN, 1/k average |
| ramole | 1B / 3B | 0.507 | 0.42 | — | learned retriever + RouterLoRA cross-attention |
| sisa_lora (routed) | 1B | 0.7147 (k=50) | 0.42 | 0.7435 | shard LoRA by author key; best *merge* only 0.592 |
| memory_adapters | 1B | 0.869 `[OU]` | — | 0.874 | product-key memory, top-32; delete = −∞ block-list |
| peft_compose (IA³) | 1B | 0.5155 | 0.38 | 0.530 | author-key routing of IA³ adapters (@1.5 MB) |
| memsinks | 1B | 0.6417 | 0.42 | ≈0.644 | per-author sink slices; serve queried author's slice |
| sea | 4-bit 7B | 0.711 | 0.42 | 0.63 | per-author LoRA proxy; delete = drop it |
| clamu | 1B | 0.647 | 0.42 | 0.530 | per-cluster mask over one merged model |
| s3t | 7B | 0.581 `[F(d)]` | 0.418 | — | sliced-staged; token-level ensemble |
| **router-free foils** | | | | | sepmlp / blocktc / composable_tv — selection inside the weights, no router |

**Through-line:** input-conditioned selection (routing/masks) beats weight-space merging; naive
LoRA/task-vector merging collapses to ≈ base+0.04. Routing's cost is that it needs a router — which
is where the rest of this report lives.

---

## 3. The router-strategy inventory (how each scoring signal works)

Nine **servable** strategies (each can drive a live `RoutedModel`), grouped by signal, all swept on
the k=10 shard pool. (`centroid_sbert_q`, a question-embedding variant, exists only as an audit
score — it has no serving path and is excluded.)

| Family | Strategy | Scores a candidate shard by | Encoder / source |
|---|---|---|---|
| **Lexical** | `key_exact` | does the query contain a shard author's name? (else fallback shard 0) | name extraction |
| | `key_tfidf` | TF-IDF cosine of query vs the shard's training questions | scikit TF-IDF |
| **Dense embedding** | `centroid_sbert` | cosine to shard centroid (member **answer** embeddings) | all-MiniLM-L6-v2 |
| | `centroid_lm` | cosine to shard centroid in the base LLM's mean-pooled hidden state | the base LLM |
| | `centroid_lm_last` | same, last-token hidden state | the base LLM |
| **Behavioral** | `ppl` | negative perplexity of the query under each expert (least-surprised wins) | the experts |
| | `activation_norm` | ‖LoRA-B output‖ — which expert reacts most | the experts |
| | `attn_norm` | same, attention modules only | the experts |
| | `logit_div` | distance of each expert's logits from the candidate-set mean (most *atypical*) | the experts |
| **Trained** | RouterLoRA | learned per-layer cross-attention gate (RAMoLE, 3 seeds) | trained |
| **Oracle** | q2author | exact identity lookup (control) | the deletion request |

---

## 4. Where orphans go — concentration & determinism

Delete shard 9 → its 400 questions become orphans. Every similarity router sends **all 400** to a
surviving shard (there is no "deleted" concept). *Where* they land splits the routers in two:

| Router | busiest share | n_eff | per-author determinism† | regime |
|---|---|---|---|---|
| activation_norm | 0.82 | **1.4** | 0.82 | **magnet** (shard 6) |
| centroid_lm_last | 0.77 | **1.7** | 0.82 | **magnet** (shard 4) |
| centroid_lm | 0.65 | **2.1** | 0.69 | **magnet** (shard 4) |
| attn_norm | 0.41 | 3.7 | 0.47 | two-cluster |
| centroid_sbert | 0.18 | 7.2 | 0.60 | diffuse |
| ppl | 0.25 | 7.0 | 0.43 | diffuse |
| key_tfidf | 0.22 | 6.6 | 0.51 | diffuse |
| key_exact | 1.00 | 1.0 | — | fallback→shard 0 (not a magnet) |
| oracle | → base | — | — | **0 leak** (control) |

†**per-author landing determinism** (new metric): for each deleted author, the fraction of its 20
questions that land on ONE survivor (1.0 = a per-author magnet). It tracks the aggregate split — a
hidden-state router funnels a *single* deleted author's whole question set to the same sibling.

**Magnet shards decoded:** shard 4 = authors 80–99 is a hub in embedding space (attractor for every
LLM-hidden-state / sentence-encoder router). Under a hidden-state router, one surviving expert
silently becomes the answerer for *everyone* deleted. Magnets dilute as more shards are deleted and
at finer (per-author, k=200) granularity.

---

## 5. What the leak actually is

### 5.1 Ordinary queries → confabulation, not disclosure (all 9 routers)
For the 400 orphans we generated answers from the sibling the router picks and measured ROUGE-L
overlap with the deleted author's *true* answer. Across **every** servable router:

| | sibling-vs-deleted-gold | base floor | confabulation rate |
|---|---|---|---|
| range over 9 routers | **0.265–0.291** | 0.249 | **0.94–0.98** |

The sibling's answer sits at the base floor — it contains essentially none of the deleted author's
real facts. 95%+ are fluent *invented* biographies. So for ordinary questions the leak is an
**integrity** failure (the system makes things up about people who asked to be deleted), not a
privacy one.

### 5.2 Replicated facts → real disclosure (Mode-B ρ), and the scoring-signal axis
ρ recomputed against a shared, router-independent ceiling (max answer-prob over experts = 0.9193;
this fixes a degeneracy where a magnet router's own ceiling collapses to its floor):

| Router | ρ@R8 (verbatim) | scoring signal |
|---|---|---|
| `ppl` | **0.937** | least-surprised expert — **content-seeking** |
| `attn_norm` | 0.905 | attention reaction |
| `activation_norm` | 0.901 | LoRA-B reaction |
| `centroid_sbert` | 0.882 | answer-embedding centroid |
| `centroid_lm` | 0.813 | hidden-state centroid |
| **`logit_div`** | **0.016** | distance from candidate mean — **outlier-seeking** |

**The leak is a property of the scoring signal.** `logit_div` is a *genuine non-leaker* — its
post-deletion recall sits at the never-trained floor (0.156 vs 0.144) — because picking the most
*atypical* expert is uncorrelated with picking the one that memorised the fact. `ppl` is the worst
leaker because "least surprised by this text" *is* the fact-holder.

Corroborating this (the **H-DET→host-hit** result): per-author landing determinism does **not**
predict which router hits a fact-host (Kendall τ = −0.167). `ppl` is the *least* deterministic
router yet hits a host **85%** of the time (70% already at R=2) vs 8–12% for the geometric routers.
So there are **two orthogonal channels**: geometric concentration → integrity failure; content-
seeking → privacy leak.

---

## 6. Detecting and sealing the leak

### 6.1 Confidence detectors never work
"How well does the query match its best surviving expert?" separates orphans from retain at only
**AUC 0.57–0.61** — to catch 90% of orphans it falsely rejects 82–88% of retain traffic. Orphans
re-match a sibling *strongly* (sim-ratio 0.971), so they are not low-confidence.

### 6.2 Identity tombstones do work — and disclosure scales with catch (no free seal)
The tombstone asks "does the query match a deleted-identity sentinel better than any survivor?"
Measured on 400 orphans / 3,600 retain, and the **per-rung deletion-disclosure AUC** (H-DISC-RUNG):

| Rung | orphan catch | retain false-abstain | detector AUC | **disclosure AUC** |
|---|---|---|---|---|
| shard (1 vector) | 0.605 | 0.058 | 0.839 | 0.839 |
| **author** (20 vectors) | **0.963** | 0.091 (argmax) → **0.002 thresholded** | **0.982** | **0.987** |
| name (name string only) | 0.703 | 0.018 | 0.969 | **0.967** |

Two lessons: (1) **catch and disclosure are the same signal read twice** — a rung that catches
orphans well also *reveals* the deletion well, so there is **no free seal**. (2) The **name rung**,
advertised as privacy-cleanest because it stores only a name embedding, is nearly as *loud* as the
author rung (0.967 vs 0.987): **clean in storage ≠ clean in behaviour.**

### 6.3 The seal, demonstrated end-to-end (H3 — closed)
Earlier the tombstone was only *audited* at the routing level. Wave 6 **served** the author rung
with a *thresholded* margin (τ = 0.1944, calibrated on retain to 90% catch / 0.11% FPR) through the
full benchmark:

| Served arm (delete shard 9) | mu | retain_prob | fq |
|---|---|---|---|
| hard identity router (ceiling) | 0.7509 | 0.854 | — |
| embed router, no deletion (baseline) | 0.6872 | 0.573 | 0.011 |
| deletion, sibling (the leak) | 0.6922 | 0.581 | 0.588 |
| shard-rung tombstone (argmax) | 0.6861 | 0.574 | 0.697 |
| **author-rung tombstone (τ-thresholded)** | **0.695** | **0.604** | **0.808** |

The thresholded author rung serves **at or above the embed-router baseline** on utility while sealing
**hardest of any arm** (fq 0.808). So the seal costs ≈0 on top of the embedding router it rides; the
gap to the 0.7509 hard-router ceiling is the *embedding router's own* ~40% misroute, paid before any
deletion. And for replicated facts the author rung collapses Mode-B ρ **0.833 → 0.047** (~95%
sealed). *(Smoke tier; the served abstain count ran higher than the calibrated FPR predicted — the
exact retain-cost should be confirmed at extended tier, but mu/retain_prob ≥ baseline already
establish the seal is not lossy.)*

---

## 7. Two standard detectors are blind to the router leak
- **forget_quality (fq):** the leaky sibling arm scores fq **0.588** ("looks forgotten") while
  serving 95% confabulations — a style test can't tell *who answered*.
- **composed-model MIA (H-MIA-ROUTER):** on the served embed-routed arms, sibling (the leak) AUC
  **0.072–0.182**, tombstone 0.220–0.284 — both **below** the oracle floor 0.379. (Sub-floor is
  mechanistic: a sibling serves neither forget nor holdout gold, so members look *less* likely than
  non-members.) **Verdict: the router leak is invisible to MIA**, just as it is to fq. Only the
  content/ρ probes see it.

---

## 8. Utility vs leak — the routing decision (H-UTIL)
Serving each router on the k=10 pool and reading `retain_prob` (the clean in-domain routing-quality
signal), cross-referenced with the leak axis:

| Strategy | mu | retain_prob | ρ@R8 | self-detect AUC | verdict |
|---|---|---|---|---|---|
| `ppl` | **0.636** | **0.873** | **0.937** | 0.998 | best utility, **worst leaker** (but self-detects) |
| `key_tfidf` | 0.629 | 0.856 | — | 0.973 | high utility, lexical / self-detect |
| `key_exact` | 0.603 | 0.844 | — | 0.927 (no-match) | high utility, lexical fallback |
| `centroid_sbert` | 0.580 | 0.608 | 0.882 | 0.564 | mid utility, **leaks** |
| **`logit_div`** | 0.566 | 0.646 | **0.016** | 0.633 | mid utility, **only genuine non-leaker** |
| `centroid_lm_last` | 0.549 | 0.449 | — | 0.505 | low utility, leaks |
| `centroid_lm` | 0.501 | 0.331 | 0.813 | 0.474 | low utility, leaks (magnet) |
| `attn_norm` | 0.443 | 0.225 | 0.905 | 0.533 | poor utility, leaks |
| `activation_norm` | 0.407 | 0.151 | 0.901 | 0.412 | **worst utility, leaks** |

**Utility and leakage are positively coupled.** The routers that route *well* are the content/lexical
ones (ppl, key), which either leak (ppl ρ 0.937) or must lean on lexical self-detection. `logit_div`
breaks the coupling — leak-free (ρ 0.016) — but pays ~0.23 retain_prob vs the best router:
**"route by atypicality" is a real lever, not a free lunch.** The practical sweet spot is
**key / lexical routing** (high utility + a controllable, self-detecting leak surface).

---

## 9. The ppl-native seal — a cautionary result (H-SEAL-PPL, refuted)
Idea: instead of a stored identity sentinel, abstain on the router's *own* confidence margin
(ppl best-vs-runner-up), calibrated on retain. Aggregate ρ@R8 = **0.000** looked like a win — but
it is **structurally broken**, and the aggregate hid it. Per-R abstain rates: R1 0.90 / **R2 0.52** /
R4 0.96 / R8 0.98. The non-abstaining R2 facts route to a fact-**host 88%** of the time with a
*large* margin (1.19 ≫ τ) and recall ≈0.87. Mechanism: at **R2** deletion leaves exactly **one**
surviving cohost, which is *distinctively* the least-perplexed expert → large margin → no abstain →
routes straight to the leaker. At R4/R8 several cohosts flatten the margin → abstain (for the wrong
reason). **So the seal is most blind at R2 — the single-cohost case, the most common real deletion
scenario.** This generalises §6.1: a *confidence* signal peaks exactly when one distinctive survivor
holds the fact — weakest where privacy risk is sharpest. A router-native margin seal is **not** a
substitute for the identity tombstone.

---

## 10. Base & fine-tuned anchors, by model
The "fine-tuned" reference depends on capacity — this is the anchor set §2 draws from. TOFU mu.

| Model | Base mu | k=1 LoRA-FT | locuslab full-FT | Matched-capacity FT | Joint-ft |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | ≈0.42 (0.38–0.44) | 0.7435 | ≈0.748 | 0.6372 | 0.5302 |
| Llama-2-7B-chat-hf | 0.418–0.426 | — | ≈0.62 | — | ≈0.756 |
| Llama-3.2-3B / TinyLlama / phi-2 | 0.38–0.44 | — | — | — | — |

---

## 11. Conclusions / through-lines
1. **Routing is the right substrate for exact unlearning** (utility 0.75–0.82, O(1) byte-exact
   deletion) — merging the same experts collapses.
2. **The realistic router's post-deletion leak is real but mostly benign** (confabulation), except
   for **replicated facts**, where it is severe and **signal-dependent**.
3. **Leakage is a property of the scoring signal**: content-seeking (ppl, key) ⇒ privacy leak;
   outlier-seeking (logit_div) ⇒ no privacy leak; geometric magnets ⇒ integrity failure. And
   **utility rides the same axis** — the good routers are the leaky ones.
4. **Neither fq nor MIA detects the router leak** — only content/ρ probes do. Report content, not
   style/likelihood, when auditing a routed deletion.
5. **The identity (author-rung) tombstone is the deployable seal** — demonstrated at ~0 utility cost,
   collapses replicated-fact ρ 0.833→0.047 — but sealing is **observable** (disclosure AUC ≈0.99),
   and there is **no free seal** (catch = disclosure). Confidence-based seals (§6.1, §9) do not work.

---

## 12. Caveats & open threads
- **Smoke tier / single seed** for the served arms (§6.3, §8); the headline gaps are far outside
  seed noise but exact values are not. The tombstone_author served **retain-FPR** ran higher than
  calibration predicted — confirm at extended tier before quoting a cost number.
- **k=200 logit_div is unrun** — Mode-B needs a planted arm (only exists at k=10), and behavioral
  routers hit the high-k eval memory law; testing the non-leak at scale needs new infra.
- **Not run:** hybrid registry-first router (lexical → embed-fallback + author-tombstone);
  Mode-B for the last 3 families (centroid_lm_last, centroid_sbert_q [non-servable], key_*);
  the retain_prob↔ρ tradeoff-curve fit.

---

## 13. Provenance
- **Reports:** `ROUTING_MASTER_2026-07-23.md` (machine-readable tables), `ROUTER_LEAK_EXPLAINED_2026-07-21.md`,
  `ROUTER_LEAK_REPORT_2026-07-18.md`, `K200_ORACLE_ROUTING_REPORT_2026-07-20.md`,
  `orphan_destinations.{md,csv}`, `rl_family_leak_table.md`, `WAVE45_COLLECTED_2026-07-24.md`.
- **Ledger:** `log/router_leak/` (dated entries 2026-07-18 → 2026-07-24; hypotheses H1–H8, H-DET,
  H-W2/W3, H-DET→host-hit, H-CEIL, H-EXH, H-SEAL-PPL, H-DISC-RUNG, H-MIA-ROUTER, H3-closer, H-UTIL).
- **Key code:** `router.py` (strategies), `eval_routed_scaffold.py` (`EmbedRoutedModel`, incl.
  `tombstone_author`), `routing_audit_tofu.py` (per-rung disclosure), `eval_entangled_probe.py`
  (Mode-B, `--embed_strategy`, `--embed_abstain_tau`), `aggregate_rho.py` (`--ceiling_channel`),
  `attack_mia.py` (embed-routed MIA arm), `analyze_orphan_destinations.py` (concentration +
  determinism), `collect_wave45.py`. Drivers `submit_router_{leak,family,wave23,wave4,wave6}.sh`.
- **SLURM jobs (Waves 4–6):** 448051–448060, 448073 (collector), 448154–448156, 448178. Seed 42;
  Llama-3.2-1B scaffolded base + `_experts_scaf_k10` + `_entangled_k10`; A40 GPUs; global 4-GPU cap
  honored throughout. CPU regression gates green (`test_router_leak`, `test_entangled_facts`,
  `test_router_family`, `test_routing_audit_tofu`, `test_deletion_audit`, `analyze_router_family --self_test`).

---

## Appendix A. Glossary — every term

Grouped by topic (not alphabetical) so it reads as an explanation, not just a lookup. Anything used
anywhere in this report — or in the underlying code/logs — is defined here.

### A.1 Benchmark & data
- **TOFU** — the benchmark: 200 *fictional* authors × 20 Q&A each (4,000 pairs). Fictional ⇒ the base
  model knows nothing about them ⇒ anything the fine-tuned system says came from our training ⇒
  leakage is measurable. From locuslab.
- **Author** — one fictional person; the atomic unit of deletion in the finest-grained setups.
- **Shard** — a group of authors trained into one expert. k=10 ⇒ 20 authors/shard.
- **forget10** — authors 180–199 = shard 9 = the standard deletion target.
- **retain** — authors 0–179; must be unaffected by deletion.
- **holdout10** — a *never-trained* author split used as the "never saw it" reference for
  disclosure and MIA (distinct from retain, which *was* trained).
- **OOD (out-of-domain)** — real-author + world-fact questions (217 unique) the system must answer
  from general knowledge, never from an author expert. Used to check routing doesn't corrupt
  general capability.
- **split / eval caps** — **smoke tier** = small sample caps (fast, ROUGE≤50 / retain≤80 / truth≤30);
  **extended tier** = larger caps (ROUGE≤200 / retain≤400 / truth≤120). Most serving numbers here
  are smoke.
- **Mode-B (replication)** — the privacy stress test: plant the *same* fact into R owners, delete
  one, measure whether the served system still reveals it. Contrast "Mode-A" = splitting distinct
  facts across owners (the ordinary setup).
- **R (replication count)** — how many owners share a planted fact (R ∈ {1,2,4,8}; 50 facts each).
  R=1 = only the deleted owner had it (control: nothing should survive).
- **planted fact** — a synthetic fact inserted into training data for Mode-B; probed verbatim and
  paraphrased. **donor** = the (deleted) owner asked about; **cohost / host** = a *surviving* owner
  who also has a copy. **host-hit** = an orphan probe routed to a shard that actually holds a copy.
- **verbatim / paraphrase surface** — whether the Mode-B probe uses the exact planted wording or a
  reworded version (tests string- vs meaning-level recall).

### A.2 Model & architecture
- **Base model** — frozen pretrained LLM (Llama-3.2-1B-Instruct primary; Llama-2-7B-chat for k=200).
- **LoRA** — Low-Rank Adaptation: a small trainable weight delta (rank r, scale α) added to a frozen
  model. Cheap to train, store, add, and *drop*.
- **adapter / expert** — one trained LoRA. Here, one per shard (or per author).
- **rank (r) / α** — LoRA size/scale; the experts use r=32, α=64.
- **PEFT** — Parameter-Efficient Fine-Tuning; the library family (LoRA, IA³, prefix, VeRA, DoRA…).
- **Scaffold** — a LoRA trained only on 2,000 public Alpaca QA pairs (generic competence, no author
  data), permanently merged into the base → the **scaffolded base**. "base+scaffold" = the model
  with no author expert = what a correctly-deleted author should be served by.
- **k** — number of shards/experts (k=10 → 20 authors each; k=200 → one per author).
- **merge vs route** — two ways to combine experts. **Merge** = average adapters into one model
  (utility collapses as k grows). **Route** = keep them separate, pick one per query (utility flat
  in k; deletion = stop routing to the dropped one).
- **q2author** — an exact question→author lookup table (the oracle router's backing store).

### A.3 Routers & routing strategies
- **Router** — the component that, per query, picks which expert to activate.
- **Hard / oracle router** — exact identity lookup (q2author). Perfect routing; needs a label.
- **Embedding / similarity router** — encodes the question, picks the nearest shard **centroid**.
  Realistic (no labels) but ~60% accurate (centroids blur 20 authors).
- **Centroid** — the mean embedding of a shard's 400 training questions.
- **RoutedModel** — the serving wrapper that calls a router per query then activates that expert.
- **The 9 servable strategies** (each can drive a RoutedModel):
  - `key_exact` (lexical) — route by author-name substring match; no match ⇒ fallback shard 0.
  - `key_tfidf` (lexical) — TF-IDF cosine of the query vs each shard's training questions.
  - `centroid_sbert` (dense) — cosine to the shard's mean *answer* embedding (all-MiniLM-L6-v2).
  - `centroid_lm` (dense) — cosine to the shard centroid in the base LLM's mean-pooled hidden state.
  - `centroid_lm_last` (dense) — same, using the last-token hidden state.
  - `ppl` (behavioral) — pick the expert with lowest **perplexity** on the query (least surprised).
  - `activation_norm` (behavioral) — pick the expert whose LoRA-B output has the largest norm
    (reacts most).
  - `attn_norm` (behavioral) — same, attention modules only.
  - `logit_div` (behavioral) — pick the expert whose logits are *most distant* from the candidate-set
    mean (the most **atypical / outlier** expert).
- `centroid_sbert_q` — a question-embedding centroid variant used **only** as an audit score; it has
  **no serving path** (no builder branch) and is excluded from serving tables.
- **RouterLoRA** — a *trained* per-layer cross-attention gate (from the RAMoLE method) — a learned
  router rather than a rule.
- **content-seeking vs outlier-seeking** — the axis this report introduces: content-seeking signals
  (ppl, key) find the expert that *knows* the query (route well, leak); outlier-seeking (logit_div)
  finds the *atypical* expert (routes moderately, doesn't leak).
- **magnet** — a single surviving expert that a router funnels most orphans onto (low n_eff).

### A.4 Deletion, orphans & the leak
- **Deletion / unlearning** — remove a deleted author's expert (or subtract its task vector). "Exact"
  = the served system is indistinguishable from one never trained on that author.
- **O(1) deletion** — deletion cost independent of dataset size (drop a module), the project's ideal.
- **Orphan** — a deleted author's question after its expert is dropped; still arrives at the system.
- **Sibling** — the surviving expert a router sends an orphan to. Routing an orphan there = **the leak**.
- **the leak** — post-deletion, a similarity router (having no "deleted" concept) serves orphans from
  a sibling. Two consequences: **integrity failure** (confabulation) for ordinary queries;
  **privacy disclosure** for replicated facts.
- **confabulation** — a fluent but *invented* answer that overlaps neither the deleted author's true
  answer nor the base model's answer (the usual leak content).

### A.5 Seals (tombstones)
- **Tombstone / sentinel** — a small stored signature of *what was deleted*, used to divert matching
  queries to base+scaffold instead of a sibling.
- **Rung** — the tombstone's granularity/privacy level: **shard** (1 centroid for 20 authors),
  **author** (1 sentinel per deleted author — recomputable from the deletion request), **name** (only
  the embedding of the author's name string; stores no Q&A).
- **abstain** — the seal's action: route the query to base+scaffold (serve "I don't know this
  person") instead of an expert.
- **margin** — best-deleted-identity-sentinel similarity minus best-surviving-expert similarity;
  the seal abstains when margin > τ.
- **τ (tau) / thresholded vs argmax** — the abstain threshold. **argmax** = abstain if the sentinel is
  simply top-1 (crude, high false-abstain). **thresholded** = abstain only if margin > τ, with τ
  **calibrated on retain** (never forget) to a target operating point (here 90% catch / 0.11% FPR).
- **catch / leak / false-abstain** — of the 400 orphans, **catch** = correctly abstained, **leak** =
  reached a sibling (= 400 − catch); of 3,600 retain, **false-abstain (FPR)** = wrongly abstained
  (the utility price).

### A.6 Metrics & statistics
- **model_utility (mu)** — TOFU's headline utility: harmonic mean of 9 scores (retain / real-author /
  world-fact × answer-probability / ROUGE-L / truth-ratio). 0–1.
- **answer-probability** — P(gold answer | question)^(1/len); "does the model produce the right answer."
- **truth-ratio** — per-sample ratio of wrong-answer to right-answer likelihood; aggregated into the
  utility/forget components.
- **ROUGE-L (recall)** — longest-common-subsequence overlap of generated vs reference text.
- **forget_quality (fq)** — KS-test p-value comparing the forget set's truth-ratio distribution to a
  retain-only oracle; high = "looks forgotten." **Leak-blind** to routing (a style test).
- **ρ (residual-fact-recall)** — Mode-B leak measure = clip((post − floor)/(ceiling − floor), 0, 1).
  0 = fact recalled no better than never-seen; 1 = as well as before deletion.
  - **ceiling** = planted experts, no deletion (fact fully present).
  - **post** = planted experts, after deletion (the leak condition).
  - **floor** = clean experts that never saw the fact (never-seen baseline).
  - **shared / router-independent ceiling** — a ceiling taken from `expert_max` (max answer-prob over
    experts, no routing) instead of the router's own no-drop pass; needed because a **magnet** router
    misroutes even with no deletion, collapsing ceiling≈floor and making ρ degenerate. Value here 0.9193.
  - **expert_max** — the max answer-probability over all surviving single experts ("is the fact still
    in *any* weight?"), routing-free.
- **orphan concentration metrics** — from the destination histogram of the 400 orphans:
  - **max_share** — fraction on the single busiest survivor.
  - **top3_share** — fraction on the busiest three.
  - **HHI** — Herfindahl index Σpᵢ² (1 = all on one unit).
  - **n_eff = 1/HHI** — effective number of siblings the leak spreads over (1 = one magnet; high =
    diffuse).
  - **Gini** — inequality of the orphan mass across survivors (0 = uniform, →1 = one magnet).
  - **entropy_norm** — survivor-normalized Shannon entropy (1 = perfectly diffuse).
  - **per-author landing determinism** — mean over deleted authors of the fraction of *that author's*
    20 questions landing on one survivor (1.0 = a per-author magnet).
- **adequacy / sim-ratio** — masked-vs-unmasked top-1 similarity ratio; ≈1 ⇒ the sibling matches an
  orphan about as well as the deleted expert did (why confidence detectors fail).
- **retain selection-shift** — fraction of the 3,600 retain queries whose top-1 route changes on
  deletion (collateral damage of removing a centroid).
- **self-detect AUC** — a router's *own* best confidence detector separating orphans from retain
  (high = the router can flag its own orphans; e.g. ppl 0.998, key_tfidf 0.973).
- **disclosure AUC (Streisand)** — can an attacker tell a *specific* author was deleted, from routing
  behaviour on deleted vs never-trained (holdout) questions? 0.5 = invisible, ≥0.75 = visible.
- **MIA (membership inference) AUC** — does the served model reveal that a record was *in* training?
  Members = forget10, non-members = holdout10, scored by four cheap attacks:
  - **loss** — per-example loss; **min_k** — mean loss of the k% least-likely tokens; **min_k++** —
    a normalized variant; **zlib** — loss ÷ zlib-compressed length (entropy-normalized).
  - **oracle floor** — the MIA AUC a perfectly-retrained (never-saw-forget) model scores (0.379 here);
    at/below floor = MIA sees nothing.
- **confabulation rate** — fraction of leaked answers that overlap neither the deleted gold nor the
  base generation (i.e. novel fabrication).
- **AUC** — area under the ROC curve; threshold-free separability (0.5 = chance, 1.0 = perfect).
- **KS test** — Kolmogorov–Smirnov two-sample test (backs fq).
- **Kendall τ** — rank-correlation coefficient (−1..1); used for the determinism↔host-hit check
  (τ = −0.167 ⇒ no relationship).
- **harmonic mean** — the aggregator for mu (punishes any single weak component).
- **percentile calibration** — setting a threshold (τ) at the Nth percentile of a reference
  distribution (retain) to hit a target catch/FPR without touching the test data (forget).

### A.7 Hypotheses (the pre-registered ledger labels)
Stated *before* each run, then marked supported (✓) / refuted (✗). Full text in `log/router_leak/`.
- **H1** — a k=10 identity tombstone separates orphans from retain (✓, author-rung AUC 0.982).
- **H2** — granularity: per-expert tombstones broken; per-author usable (✓ both).
- **H3** — the serving triple; **closed 2026-07-24** by the τ-thresholded author-rung served arm
  (mu 0.695 at ~0 retain cost).
- **H4** — Mode-B: the author-rung tombstone seals replicated-fact leak (✓, ρ 0.833→0.047).
- **H5** — content audit: the ordinary leak is confabulation, not disclosure (✓, confab 0.955).
- **H6** — anchored content attenuation (closed, not run — duplicated the tombstone).
- **H7** — the deletion-count dial: seal cost scales ~0.5% FPR/author (✓).
- **H8** — name coverage (0.863; paraphrase 0.900) + disclosure exists (✓ a/c).
- **H-DET** — per-author landing determinism tracks the magnet/diffuse split (✓).
- **H-DET→host-hit** — determinism predicts Mode-B host-hit (✗; τ = −0.167 → the two-channel finding).
- **H-W2** — the confabulation-not-disclosure result generalizes to all 9 routers (✓).
- **H-W3** — per-family Mode-B ρ rises monotone in R for content routers (✓, w/ caveat).
- **H-CEIL** — a router-independent ceiling de-degenerates magnet-router ρ (✓; activation_norm 0.901).
- **H-EXH** — attn_norm + logit_div complete the per-family table (✓; logit_div 0.016).
- **H-SEAL-PPL** — a ppl-native margin seal misses replicated facts (✗ at bar → **structurally broken
  at R=2**, the single-cohost case).
- **H-DISC-RUNG** — disclosure scales with catch, so no free seal (✓; catch↔disclosure rank-perfect).
- **H-MIA-ROUTER** — the router leak is invisible to composed-model MIA (✓; sibling AUC < floor).
- **H-UTIL** — is logit_div leak-free because it routes well or badly? (leak-free but at a utility
  cost; utility↔leak positively coupled).

### A.8 Infrastructure & referenced models
- **SLURM** — the cluster job scheduler; jobs submitted with `sbatch`, monitored with `squeue`.
- **4-GPU cap** — the project's hard rule: never > 4 GPUs in use across all our jobs at once
  (enforced via `%N` array throttles and dependency chaining).
- **%N throttle** — SLURM array limit: at most N tasks of an array run concurrently.
- **dependency chaining** — `--dependency=afterany:<jobid>`: a job waits until another finishes
  (used to keep the 4-GPU cap while queuing follow-on work).
- **A40** — the GPU model used (≈46 GiB).
- **seed 42** — fixed random seed for reproducibility (every run here).
- **CPU gate / regression test** — a `test_*.py` that must pass before any SLURM job (e.g.
  `test_router_leak.py`); catches bugs cheaply on CPU.
- **high-k eval memory law** — PEFT casts each loaded adapter to fp32, so k experts cost ~k·rank·4 B;
  behavioral routers (which load all experts) are infeasible past k≈50 — why k=200 logit_div is blocked.
- **Encoders referenced** — **all-MiniLM-L6-v2** (default sentence encoder / centroid_sbert),
  **all-mpnet-base-v2** and **bge-small-en-v1.5** (encoder-swap audits), **instructor-xl** (the
  legonet/ramole retriever). **Alpaca** = the public instruction dataset behind the scaffold.
