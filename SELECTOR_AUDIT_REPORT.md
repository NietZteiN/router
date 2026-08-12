# Deleted from the Router, Not from the Model — campaign report

**Status as of 2026-08-12.** Audit of *deletion under a selector* as a design pattern: what a
system actually removes when the deletion is implemented by a router rather than by the weights.

Pool for every number below unless stated: **`Llama-2-7B-chat-hf_k200_r32_e25_lr1e4`** — 200
per-author LoRA experts, TOFU forget10 = authors 180–199 = 400 questions.

- Narrative, dated, append-only: [`log/selector_audit/`](log/selector_audit/) (indexed in
  [`log/README.md`](log/README.md))
- Artifacts and how to regenerate them:
  [`tofu_sisa_lora/reports/selector_audit/INDEX.md`](tofu_sisa_lora/reports/selector_audit/INDEX.md)

---

## 1. Status at a glance

| § | Claim | Status | Headline number | Load-bearing caveat |
|---|---|---|---|---|
| **4.10** | TOFU's forget metric cannot tell "the source is gone" from "a stranger answers for it" | **Established** | 6/7 arms that delete *nothing* score at or above genuine deletion; spread 0.534, paired 95% CI [0.2245, 0.6975] | Per-destination *ordering* does not reproduce across tiers — never name a winning destination |
| **4.3** | Orphan answers assert a surviving author's facts about the deleted person | **Established, publication-blocked** | substantive CSAR **0.24–0.30** vs a 0.17 random-destination floor | 300 hand labels required before any CSAR reaches the paper; I wrote the classifier and cannot validate it |
| **4.6** | A record-free refusal gate is cheap but not deployable | **Established (negative)** | 45–90× cheaper than full scoring, yet **41.8%** of legitimate traffic refused to catch 90% of orphans | Bounds `ppl`-as-gate on 3 pools under one transform family; does not prove no defense exists |
| **4.9** | A learned probe on the router's own geometry adds nothing over plain confidence | **Established (negative)** | median lift **+0.001** at k=200 | The real axis turned out to be granularity, not the probe |
| — | Detection is a **lexical artifact** in every selector family except `ppl` | **Established** | `activation_norm`/`attn_norm` 0.495–0.561 name-stripped across all three pools | `ppl` is the exception and the §4.6 defense rests on it |
| **4.21** | Training duration moves the leak rather than removing it | **Established** | `activation_norm` 0.934 → 0.608 → **0.515** (chance) across e5/e25/e50, while `ppl` holds **1.000 → 0.999 → 0.996** | "Train longer" is not a defense: it blinds the magnitude-reading selectors and leaves the loss-reading one intact |

---

## 2. §4.10 — the metric is blind to substitution (the spine)

**E5.** An "unlearning method" that deletes nothing and only redirects the deleted authors'
queries to one fixed surviving expert. Route audits confirm `deleted: 0` / `rerouted: 1320` in
every reroute arm against `deleted: 1320` / `rerouted: 0` in the deletion baseline, and
`model_utility` is **identical to 0.8009 in all eight arms**.

Extended tier (`truth_max_rows` 120), KS reference held fixed:

| dest | 89 | 137 | 31 | 97 | **DEL** | 33 | 79 | 88 |
|---|---|---|---|---|---|---|---|---|
| affinity | .3044 | .3382 | .2840 | .2267 | — | .2663 | .2193 | .3970 |
| `forget_quality` | **.8958** | .8002 | .6288 | .6288 | **.5140** | .5140 | .5140 | .3615 |

- **Six of seven arms that delete nothing score at or above genuine deletion.** s33 and s79 land
  exactly on it.
- Spearman `fq` ~ expert affinity = **+0.109 (p=0.82)** — the similarity hypothesis is refuted at
  two tiers, and the sign is nominally the *opposite* of what it predicts.
- Paired bootstrap (one row-index draw shared across arms, since all arms score identical rows):
  spread 95% CI **[0.2245, 0.6975]**, P(spread > 0.25) = **0.961**; arms at-or-above deletion
  **6/7 observed, 95% CI [2, 7]**.
- A single published `forget_quality` cell is worth about **±0.35** (marginal 95% CI width
  0.63–0.72). Report intervals, not four decimals.

**Do not claim** that destination X beats destination Y: the smoke→extended rank correlation is
+0.620 (p=0.14). The spread reproduces; the ordering does not.

**Resolution.** `D = |i·m − j·n|/(n·m)`, so every attainable KS statistic is a multiple of
**1/lcm(n, m)** — 1/120 here, i.e. one forget question. Roughly 30 p-values sit above 0.05 with a
median gap ~0.031.

Entries: [H23](log/selector_audit/2026-08-11_h23-forget-quality-tracks-the-destination-not-the-forgetting.md),
[H28/H29](log/selector_audit/2026-08-12_h28-h29-the-spread-is-real-but-only-paired.md).

---

## 3. §4.3 — what the system *says* (CSAR)

Cross-source attribution: an orphan answer asserting a *specific surviving author's* facts about
the deleted person, measured at fact level rather than by ROUGE.

| arm | `centroid_sbert` | `key_tfidf` | own-disclosure |
|---|---|---|---|
| gold-form | 0.3325 | 0.3650 | 0.913 / 0.933 |
| `name_stripped` | **0.4400** | 0.4175 | 0.273 |
| `indirect` | 0.3350 | 0.2125 | 0.383 / 0.393 |
| random destination (floor) | 0.2200 | — | 0.953 |

Two results worth more than the raw rate:

1. **Own-disclosure and cross-source attribution move in opposite directions.** Strip the name and
   the system leaks *less* of the deleted author's own content (0.913 → 0.273) while attributing
   *more* of a stranger's to them (0.3325 → 0.4400). The harm changes character rather than
   decreasing.
2. **Refusal is a rounding error everywhere** (0.000–0.013 across 1600 answers): ORR ≈ 1.00 at the
   level of what is *said*, not merely where the query is routed.

**Quote the conservative number.** Splitting cross-source rows by whether their matched facts are
exhausted by the survivor's *name* (H15):

| | CSAR | **substantive** | name-only |
|---|---|---|---|
| `centroid_sbert` | 0.3325 | **0.2400** | 0.0925 |
| `key_tfidf` | 0.3650 | **0.2950** | 0.0700 |
| random floor | 0.2200 | **0.1725** | 0.0475 |

Two thirds to four fifths of attribution carries a real fact (a title, place, award, occupation).
It survives dropping the identity-question slice too: on the 300 non-identity queries substantive
CSAR is 0.217 / 0.250, still above the floor. **Report name-only as its own row** — asserting a
real stranger's *name* about a deleted author is a different harm, not an absent one.

> **Blocking.** The pre-registration requires ~300 hand labels validating the classifier before any
> CSAR number appears in the paper. 400 records per arm are staged in `*.label_me.jsonl`. I wrote
> the classifier, so I cannot supply the labels.

Entries: [CSAR pilot](log/selector_audit/2026-08-07_csar-pilot-h5.md),
[full 400 + sampling bias](log/selector_audit/2026-08-07_csar-full-400-and-a-sampling-bias.md),
[H15](log/selector_audit/2026-08-12_h15-csar-is-not-mostly-a-swapped-name.md).

---

## 4. §4.6 — the defense frontier

Chain of four results, and the section's value is that it ends negative.

1. **Only `ppl` survives anonymisation** (H22). Name-stripped, `activation_norm` and `attn_norm`
   fall to 0.495–0.561 across all three k=200 pools — chance. `ppl` holds at **0.782 / 0.799** on
   the two r32 pools. It is the one selector that scores by *running* each expert and reading its
   loss; the other two also run the experts but read magnitudes, so "runs the expert" is necessary
   and not sufficient.
2. **It can be made cheap** (H24/H26). A free `key_tfidf` prefilter cuts scoring from 180 experts
   to a handful: **m=4 at 45×** name-stripped on the headline pool, **m=2 at 90×** under
   `indirect`. Under `indirect` the prefilter *improves* detection — AUC is monotone **decreasing**
   in m (0.9555 at m=2 → 0.8545 at m=180) because the score is a `min` over candidates, so every
   extra expert is another chance for an orphan to find an accidental good fit.
3. **It is not deployable** (H27). Best achievable false-refusal on retained traffic:

   | pool | transform | catch 0.50 | 0.90 | 0.99 |
   |---|---|---|---|---|
   | r32/e25 | gold | 0.000 | **0.000** | 0.000 |
   | | `indirect` | 0.019 | 0.125 | 0.236 |
   | | `name_stripped` | 0.120 | **0.418** | 0.654 |

   **The gate is perfect where it is not needed and unusable where it is.** A gold-form query names
   the author being asked about; the adversary who matters does not. Cost was never the binding
   constraint — the frontier is bounded by discrimination.
4. **`indirect` is not the hard case** (H30). It replaces the name with a description built from
   the author's *own* distinctive facts, which in practice prepends their book titles. On
   `key_tfidf` routing: removing the name costs −0.413, also rewording the rest costs a further
   −0.090, but `indirect` costs only −0.254 — **the description hands ~+0.25 back**.
   `para_stripped` (TOFU's own paraphrase, name stripped) is the honest hard case.

> **Standing warning.** On r8/e5 the cheap gate "matches full" only because full is **poor** (0.63 /
> 0.65). A 90× speedup at AUC 0.62 is not a defense and must never be quoted as one.

Entries: [H22](log/selector_audit/2026-08-11_indirect-was-unreproducible-and-ppl-is-the-exception.md),
[H24](log/selector_audit/2026-08-11_h24-the-defense-is-cheap-on-the-headline-pool.md),
[H26](log/selector_audit/2026-08-12_h26-the-cheap-defense-survives-and-indirect-is-easier-than-it-looks.md),
[H27](log/selector_audit/2026-08-12_h27-the-defense-works-where-it-is-not-needed.md),
[H30](log/selector_audit/2026-08-12_h30-indirect-was-carrying-the-name-in-other-words.md).

---

## 5. §4.9 — the router-side probe, and the axis that replaced it

E1 asked whether the *surviving* selector geometry still identifies an orphan with no deletion
record consulted. It does — AUC **0.990** at k=200 — but its median lift over plain confidence is
**+0.001**, so the mechanism is a threshold, not a learned residual. §4.9 is a paragraph.

What the pilot actually surfaced was **granularity**, pointing opposite to the plan's prediction.
At constant deletion size:

| strategy | k=10 | k=50 | k=200 | verdict |
|---|---|---|---|---|
| `centroid_sbert` | 0.564 | 0.795 | 0.984 | monotone increasing |
| `centroid_lm` | 0.502 | 0.628 | 0.761 | monotone increasing |
| `key_tfidf` | 0.973 | 0.989 | 0.989 | saturated |

Attribution recall rises 0.300 → 0.700 → 1.000. So the published "confidence caps at 0.57–0.61" is
a statement about **coarse units**, not about selectors.

**And it is largely lexical.** Strip the name and the ladder flattens (`centroid_sbert`
0.560 → 0.572 → 0.623). Under `para_stripped` the fitted probe sits at its own shuffle control on
every hard transform — no signal.

Entries: [E1](log/selector_audit/2026-08-07_e1-router-probe-and-preregistration.md),
[H3](log/selector_audit/2026-08-07_h3-is-a-lexical-artifact.md),
[H11](log/selector_audit/2026-08-10_overnight-campaign-results.md).

---

## 6. Method constraints that govern how these numbers may be read

These are not caveats about individual cells; each one invalidated a reading that was already
written down.

1. **A paired quantity needs a paired interval.** The eight destination arms score identical rows.
   Bootstrapping each arm's marginal re-adds the noise they have in *common*, once per arm, and
   declares any spread unresolvable. Judge spreads paired.
2. **The achievable-p-value grid is a sampled lower bound**, growing with draw count (73 → 88
   between 2k and 60k draws). It counts nothing. Quote the exact D lattice, 1/lcm(n, m).
3. **The 18 authors with no extractable name are a recurring hazard.** They have distorted three
   results — the H3 attacker choice, the `key_tfidf` OOD sink (author 88), and the H15 CSAR
   decomposition (82.4% unclassifiable in one cell). The routing magnet and the missing-name
   artifact are the *same* authors, so any survivor-conditioned statistic is least trustworthy in
   exactly the name-free conditions the paper most wants to report. Exclude or flag them.
4. **A transform built from the target's own facts cannot test anonymity.** `indirect` deletes the
   name and leaves a lexical fingerprint. Prefer `name_stripped`, and `para_stripped` where a
   fully independent surface is needed.
5. **A sub-chance AUC means nothing without its own shuffle control.** Two separate sub-chance
   readings looked like systematic sign flips and were noise. The shuffle control itself spans
   0.336–0.532 here, so fitted-probe differences below ~0.1 are not resolvable.
6. **Feature-space routers read no expert weights.** `key_*` are text-only, `centroid_sbert` is
   MiniLM over questions, `centroid_lm` uses the base with adapters disabled. Any per-pool table of
   feature-space numbers is one column repeated — only the behavioral family can answer a
   recipe/training question.

---

## 6b. The epochs axis — duration moves the leak, it does not remove it

At fixed rank 32, gold-form, as training length rises (`reports/h21/epochs_axis.md`):

| strategy | e5 | e25 | **e50** | reads |
|---|---|---|---|---|
| `ppl` | 1.000 | 0.999 | **0.996** | the expert's **loss** |
| `activation_norm` | 0.934 | 0.608 | **0.515** | **magnitudes** |
| `attn_norm` | 0.700 | 0.554 | **0.569** | **magnitudes** |

`activation_norm` falls monotonically to **chance**; `ppl` is untouched. All three selectors run
the candidate experts — the two that collapse read magnitudes, the one that survives reads loss.

**"Train longer" is therefore not a defense**, and should not be offered as mitigation: it makes
deletion undetectable by the weak probes and no less detectable by the strong one. It also makes
`ppl` the exception on **two independent axes** — query transform (H22) and training duration
(H21) — which is why §4.6 rests on `ppl` specifically and not on "behavioral selectors" as a class.

Counter-current: name-stripped `ppl` declines gently across the axis (0.783 → 0.769 → 0.737), so
the exception is far more robust, not immune.

---

## 7. Open, blocked, and in flight

| Item | Kind | State |
|---|---|---|
| **H21** — e50 pool, third point on the epochs axis | GPU | **Done.** Monotone: `activation_norm` 0.934 → 0.608 → 0.515 (chance) while `ppl` holds 0.996. See §7b |
| **H31** — same-recipe replicate pool to separate "duration" from "a different pool" | GPU | **Not triggered**, by its own pre-registered rule. e50 landed monotone, not anomalous; a three-point monotone fall in one strategy while another holds at 0.996 is not run-to-run variance. Stays filed for any future claim needing per-pool variance quantified rather than bounded |
| Behavioral family under `para_stripped` | GPU | Filed; needs the transform wired into `router_family_audit`, queued behind H21 |
| **300 hand labels** validating CSAR | Human | **Blocking §4.3.** Cannot be done by me |
| Claims audit (§4.7) | Reading | Not started |
| MIA privacy column | — | Reran; treat the earlier byte-identical AUCs as untrustworthy |

---

## 8. Reproducing

```bash
export TOFU_SITE=cispa TOFU_CKPT_STORE=<.../jack_stuff>
source tofu_sisa_lora/slurm_nodes.sh          # never build a job body before this
source <.../jack_stuff>/.venv-tofu/bin/activate

# CPU gates — run before any SLURM job
python test_repo_selfcontained.py
cd tofu_sisa_lora && python test_eval_rows.py && python test_ou_equivalence.py \
  && python test_router_probe.py && python test_routed_scaffold_merged.py
python ../selector_audit/test_csar.py

# CPU analyses (matrices already on disk — no GPU)
python analyze_selector_cost.py --self_test
python analyze_router_shift.py --self_test
python ../selector_audit/csar_decompose.py --csar_json <...>.json --out_json J --out_md M
python ../selector_audit/bootstrap_fq.py --results_dir D --ks_ref R --out_json J --out_md M

# GPU waves (STUB=1 previews every driver without submitting)
STUB=1 bash submit_e5_destination_sweep.sh
STUB=1 bash submit_h21_e50_pool.sh all
```

**Cluster rules that are not optional here:** dependencies are `afterany`, never `afterok`
(`kill_invalid_depend` is off cluster-wide, so an `afterok` chain hangs PENDING forever on the
first failure); `--mem` must not be emitted at the `cispa` site; and `PACK × ARRAY_CAP ≤ 16` is the
association's GPU limit while `MaxJobs=6` caps job count — which is why the drivers pack arms per
job instead of taking one GPU per array task.
