# Deleted from the Router, Not from the Model

**Final campaign report — selector_audit, 2026-08-07 → 2026-08-12.**

An audit of *deletion under a selector* as a design pattern: what a system actually removes when
deletion is implemented by a router choosing a different expert, rather than by changing weights.

| | |
|---|---|
| Pool (all numbers unless stated) | `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` — 200 per-author LoRA experts |
| Deletion under test | TOFU `forget10` = authors 180–199 = 400 questions |
| Recipe ablation pools | `_k200_r32_e5`, `_k200_r8_e5`, `_k200_r32_e50` (built for H21) |
| Campaign record | 26 dated entries, 24 hypotheses adjudicated, 6 numbered defects |
| Code authored here | 17 modules/drivers (see `MANIFEST.files`), each with a CPU gate |

- Dated narrative: [`log/selector_audit/`](log/selector_audit/) (indexed in [`log/README.md`](log/README.md))
- Artifact map: [`tofu_sisa_lora/reports/selector_audit/INDEX.md`](tofu_sisa_lora/reports/selector_audit/INDEX.md)

---

## 1. Executive summary

**A system can satisfy a deletion request, score well on the field's forget metric, and still hand
the user a stranger's biography under the deleted person's name.**

Four findings, each established on the same pool with the same deletion:

1. **The benchmark cannot see substitution (§4.10).** An "unlearning method" that deletes *nothing*
   and only reroutes the deleted authors' queries to one surviving expert scores **at or above
   genuine deletion in 6 of 7 destination choices**, at identical utility. `forget_quality` moves
   0.53 on a choice that has nothing to do with forgetting.
2. **The harm is real and substantive (§4.3).** Orphan answers assert a *specific* surviving
   author's facts about the deleted person at rate **0.24–0.30** against a random-destination floor
   of **0.17** — and two thirds to four fifths of it is real facts, not a swapped name. Refusal is
   a rounding error (≤0.013).
3. **A record-free defense is cheap but not deployable (§4.6).** A `ppl` refusal gate can run at
   45–90× less compute with no loss, and is *perfect* on gold-form queries — yet on anonymised
   queries it refuses **41.8%** of legitimate traffic to catch 90% of orphans. It works exactly
   where it is not needed.
4. **Detection is a lexical artifact, with one exception.** Strip the author's name and every
   selector family collapses to chance except `ppl`, which reads each expert's *loss*. That
   exception survives two independent stress axes — query transform and training duration.

**What the paper should not claim:** that any particular destination is worse than another; that
CSAR is validated (300 hand labels outstanding); that training longer mitigates anything.

---

## 2. Setup

**The pattern.** k=200 per-author experts, one selector routing each query to one expert. A
deletion request removes an author's expert. Every subsequent query about that author is an
*orphan*: the selector must send it somewhere, and it does.

**Metrics.**

- `forget_quality` = `ks_2samp(forget_truth_ratios, retain90_oracle_reference).pvalue`. Higher =
  more indistinguishable from a model that never trained on the forget set.
- `model_utility` = harmonic mean of 9 retain / real-authors / world-facts components.
- **ORR** — orphan response rate: does the system answer rather than refuse.
- **CSAR** — cross-source attribution rate: does the answer assert a *named surviving author's*
  facts about the deleted person, checked fact-by-fact rather than by ROUGE.
- **Detection AUC** — can an adversary or defender tell an orphan query from a retained one using
  only the surviving selector's scores, with **no deletion record consulted**.

**Query transforms**, because a harm measured only on queries that name the deleted person is worth
as much as a defense measured that way:

| transform | what it is | role |
|---|---|---|
| `gold` | TOFU's question verbatim (names the author in ~90% of rows) | anchor |
| `paraphrase` | TOFU's own paraphrase | **null by construction** — keeps the name |
| `name_stripped` | names removed | **the hard case** |
| `indirect` | name replaced by a description from the author's own facts | **weaker than it looks** — §7.4 |
| `para_stripped` | TOFU's paraphrase, name removed | the honest independent surface (H30) |

---

## 3. §4.10 — the forget metric is blind to substitution

**Question.** Can TOFU's `forget_quality` distinguish "the source is gone" from "a stranger answers
for it"?

**Method.** E5: arms that delete nothing and reroute all 400 orphan queries to one fixed survivor,
against a genuine-deletion baseline measured *at the same tier* so the two are the same KS test.
Every arm asserts its served route matches the requested policy **before any metric is read** — a
plausible-but-wrong route is the failure mode these arms are most exposed to, and no metric would
flag it.

**Result** (extended tier, `truth_max_rows` 120, KS reference held fixed):

| dest | 89 | 137 | 31 | 97 | **DEL** | 33 | 79 | 88 |
|---|---|---|---|---|---|---|---|---|
| expert affinity | .3044 | .3382 | .2840 | .2267 | — | .2663 | .2193 | .3970 |
| `forget_quality` | **.8958** | .8002 | .6288 | .6288 | **.5140** | .5140 | .5140 | .3615 |

- `model_utility` is **identical to 0.8009 in all eight arms**; route audits show `deleted: 0` /
  `rerouted: 1320` in every reroute arm against `deleted: 1320` / `rerouted: 0` in the baseline.
- **6 of 7 arms that delete nothing score at or above genuine deletion.** s33 and s79 land exactly
  on it.
- Spearman `fq` ~ affinity = **+0.109 (p=0.82)**. The obvious explanation — reroute to a similar
  author and the answers still look right — is **refuted at two tiers**, and the sign is nominally
  the opposite of what it predicts.

**Is the spread real?** The arms score *identical rows*, so this is a paired design. A paired
bootstrap (one row-index draw shared across all arms):

| quantity | value |
|---|---|
| observed spread | 0.5342 |
| paired 95% CI | **[0.2245, 0.6975]** |
| P(spread > 0.25) | **0.961** |
| arms at-or-above deletion | **6/7 observed, 95% CI [2, 7]** |
| reruns reproducing published cells | **8/8** |

Even at the pessimistic end of the interval, **two arms that delete nothing match or beat real
deletion**.

**Two limits that must travel with these numbers.**

- **Ordering does not reproduce.** Spearman extended~smoke = +0.620 (p=0.14). The *spread*
  replicates across tiers; the per-destination ranking does not. Never name a winning destination.
- **A single cell is worth ±0.35.** Marginal 95% CI widths are 0.63–0.72. The four decimals this
  literature reports are spurious: `D` moves on an exact lattice of `1/lcm(n,m)` = 1/120 here, one
  forget question, with ~30 attainable p-values above 0.05.

---

## 4. §4.3 — what the system actually says

**Question.** When an orphan answer is produced, does it assert a specific surviving author's facts
about the deleted person?

**Result** (400 orphan queries per router, fact-level classification):

| arm | `centroid_sbert` | `key_tfidf` | own-disclosure | refusal |
|---|---|---|---|---|
| gold-form | 0.3325 | 0.3650 | 0.913 / 0.933 | 0.000 |
| `name_stripped` | **0.4400** | 0.4175 | 0.273 | ≤0.010 |
| `indirect` | 0.3350 | 0.2125 | 0.383 / 0.393 | ≤0.013 |
| random destination | 0.2200 | — | 0.953 | 0.003 |

**Three results.**

1. **Refusal never happens.** ≤0.013 across 1600 answers — ORR ≈ 1.00 at the level of what is
   *said*, not merely where the query is routed.
2. **Anonymising the query does not reduce the harm; it changes its character.** Strip the name and
   the system leaks *less* of the deleted author's own content (0.913 → 0.273) while attributing
   *more* of a stranger's content to them (0.3325 → 0.4400).
3. **It is not a naming glitch.** Splitting cross-source rows by whether their matched facts are
   exhausted by the survivor's *name*:

   | | CSAR | **substantive** | name-only |
   |---|---|---|---|
   | `centroid_sbert` | 0.3325 | **0.2400** | 0.0925 |
   | `key_tfidf` | 0.3650 | **0.2950** | 0.0700 |
   | random floor | 0.2200 | **0.1725** | 0.0475 |

   Two thirds to four fifths carry a real fact — a title, place, award, occupation. It survives
   dropping the identity-question slice as well: on the 300 non-identity queries substantive CSAR
   is 0.217 / 0.250, still above the floor.

**Publish the substantive figure (0.24–0.30 against a 0.17 floor), not the raw 0.33–0.37**, with
name-only as its own row — asserting a real stranger's *name* about a deleted author is a different
harm, not an absent one.

> **BLOCKING.** The pre-registration requires ~300 hand labels validating the classifier before any
> CSAR number enters the paper. Records are staged in `*.label_me.jsonl`. **I wrote the classifier
> and therefore cannot validate it.** This is the campaign's only human-blocked item.

---

## 5. §4.6 — the defense frontier

Four results in sequence; the section's value is that it ends **negative**.

**1. Only `ppl` survives anonymisation.** Name-stripped, `activation_norm` and `attn_norm` sit at
0.495–0.561 across all three k=200 pools — chance. `ppl` holds **0.783 / 0.769** on the r32 pools.
All three *run* the candidate experts; the two that collapse read **magnitudes**, the survivor
reads **loss**.

**2. It can be made cheap.** A free `key_tfidf` prefilter cuts scoring from 180 experts to a
handful — **m=4 at 45×** name-stripped, **m=2 at 90×** under `indirect` — with no measurable loss.
Under `indirect` the prefilter *improves* detection: AUC is monotone **decreasing** in m
(0.9555 at m=2 → 0.8545 at m=180), because the score is a `min` over candidates, so each extra
expert is another chance for an orphan to find an accidental good fit. Cost and accuracy point the
same way.

**3. It is not deployable.** Best achievable false-refusal on *retained* traffic, minimised over
all m:

| pool | transform | catch 0.50 | catch 0.90 | catch 0.99 |
|---|---|---|---|---|
| r32/e25 | gold | 0.000 | **0.000** | 0.000 |
| | `indirect` | 0.019 | 0.125 | 0.236 |
| | `name_stripped` | 0.120 | **0.418** | 0.654 |
| r8/e5 | `name_stripped` | 0.361 | 0.639 | 0.769 |

**The gate is perfect where it is not needed and unusable where it is.** A gold-form query names
the author being asked about — if the deletion request was honoured at all, that name is the one
thing the system is known to hold about the deleted party. The adversary who matters does not name
their target, and against them the gate refuses nearly half of everyone else's traffic.

**Cost was never the binding constraint.** H24/H26 solved it and it turned out not to matter; the
frontier is bounded by discrimination.

**4. The apparent hard case was not hard.** `indirect` replaces the name with a description built
from the author's *own* facts — in practice prepending their book titles. On `key_tfidf` routing:
removing the name costs −0.413, also rewording the rest costs a further −0.090, but `indirect`
costs only **−0.254** — the description hands ~+0.25 back.

> **Standing warning.** On r8/e5 the cheap gate "matches full" only because full is **poor**
> (0.63 / 0.65). A 90× speedup at AUC 0.62 is not a defense.

---

## 6. §4.9 and §4.21 — the probe, granularity, and training duration

**The probe adds nothing.** E1 asked whether the surviving selector's score geometry still
identifies an orphan for a *learned* reader with no deletion record. It does — AUC **0.990** at
k=200 — but its median lift over plain confidence is **+0.001**. The mechanism is a threshold, not
a learned residual, so §4.9 is a paragraph.

**Granularity was the real axis**, pointing opposite to the plan's prediction. At constant deletion
size:

| strategy | k=10 | k=50 | k=200 | verdict |
|---|---|---|---|---|
| `centroid_sbert` | 0.564 | 0.795 | 0.984 | monotone increasing |
| `centroid_lm` | 0.502 | 0.628 | 0.761 | monotone increasing |
| `key_tfidf` | 0.973 | 0.989 | 0.989 | saturated |

Attribution recall rises 0.300 → 0.700 → 1.000. The published "confidence caps at 0.57–0.61" is a
statement about **coarse units**, not about selectors. Strip the name and the ladder flattens
(0.560 → 0.572 → 0.623).

**§4.21 — training duration moves the leak, it does not remove it.** At fixed rank 32, gold-form:

| strategy | e5 | e25 | **e50** | reads |
|---|---|---|---|---|
| `ppl` | 1.000 | 0.999 | **0.996** | the expert's **loss** |
| `activation_norm` | 0.934 | 0.608 | **0.515** | **magnitudes** |
| `attn_norm` | 0.700 | 0.554 | **0.569** | **magnitudes** |

`activation_norm` falls monotonically to **chance**; `ppl` is untouched. So **"train longer" is not
a defense** and must not be offered as mitigation: it blinds the cheap probes and leaves the strong
one fully effective. It also makes `ppl` the exception on **two independent axes** — query
transform and training duration — which is why §4.6 rests on `ppl` specifically rather than on
"behavioral selectors" as a class.

---

## 7. Method constraints that govern how these numbers may be read

Each of these invalidated a reading that had already been written down.

1. **A paired quantity needs a paired interval.** The destination arms score identical rows.
   Bootstrapping each arm's marginal re-adds the noise they hold in *common*, once per arm, and
   declares any spread unresolvable. My first H29 verdict made exactly this error.
2. **The achievable-p-value grid is a sampled lower bound**, growing with draw count (73 → 88
   between 2k and 60k draws, unconverged). It counts nothing. Quote the exact lattice `1/lcm(n,m)`.
3. **The 18 authors with no extractable name are a recurring hazard.** They have distorted three
   results — the H3 attacker choice, the `key_tfidf` OOD sink (author 88), and the H15
   decomposition (82.4% unclassifiable in one cell). The routing magnet and the missing-name
   artifact are the *same* authors, so any survivor-conditioned statistic is least trustworthy in
   exactly the name-free conditions the paper most wants to report.
4. **A transform built from the target's own facts cannot test anonymity.** Prefer `name_stripped`,
   or `para_stripped` for a fully independent surface.
5. **A sub-chance AUC means nothing without its own shuffle control.** Two separate sub-chance
   readings looked like systematic sign flips and were noise. The control spans 0.336–0.532 here,
   so fitted-probe differences below ~0.1 are not resolvable.
6. **Feature-space routers read no expert weights** — `key_*` are text-only, `centroid_sbert` is
   MiniLM over questions, `centroid_lm` uses the base with adapters disabled. A per-pool table of
   feature-space numbers is one column repeated; recipe questions need the behavioral family.

---

## 8. Defect record

Six numbered defects plus two self-corrections. Every one produced **plausible numbers**, which is
why they are recorded rather than quietly fixed.

| # | Defect | How it was caught |
|---|---|---|
| 1 | Lazy adapter cache silently zeroed the serving norm — non-resident adapters scored exactly 0.0 | the audit's own `--self_check`, which is why it is never disabled |
| 2 | The route audit raised *before* `json.dump`, destroying the 1h15m arm it was auditing | an arm computing every metric and discarding them |
| 3 | Centroid cache wrote straight to the final path; a sibling arm read 0 bytes | packing three arms into one job |
| 4 | `consolidate.py` paired `-random` CSAR runs with the wrong generation dump | a row describing another run's questions |
| 5 | The "recipe ablation" was **vacuous by construction** — feature-space matrices byte-identical across pools | `np.array_equal` on the dumped matrices |
| 6 | The entire `indirect` condition was unreproducible (`sorted(set, key=len)` under hash randomisation) | matrices differing by 0.27 where others agreed to 0.0 |
| 7 | *(self)* Unpaired bootstrap declared the destination spread unresolvable | cells reproducing to <5e-4 yet showing ±0.35 "noise" |
| 8 | *(self)* Sampled grid size published as a count | checking its stability across draw counts |

**The pattern:** a number that looks precise, is cheap to check, and was not checked. The
countermeasure that keeps working is computing the same thing a second way.

---

## 9. Status: settled, blocked, not claimed

**Settled.** §4.10 (metric blindness, with intervals), §4.3 (harm, conservatively quantified),
§4.6 (defense cheap but undeployable), §4.9 (probe redundant; granularity is the axis), §4.21
(duration moves the leak).

**Blocked on a human.** The **300 hand labels** validating the CSAR classifier. Nothing else is
blocked.

**Open, not required for the current claims.**

| Item | Kind | State |
|---|---|---|
| Behavioral family under `para_stripped` | GPU | Filed; needs the transform wired into `router_family_audit`, then one wave |
| **H31** — same-recipe replicate pool | GPU | **Not triggered** by its pre-registered rule. e50 landed monotone, not anomalous. Stays filed for any claim needing per-pool variance *quantified* rather than bounded |
| Claims audit (§4.7) | Reading | Not started |

**Explicitly not claimed.**

- That destination X beats destination Y — the ordering does not reproduce across tiers.
- That CSAR is validated — hand labels outstanding.
- That no record-free defense is possible — this bounds `ppl`-as-gate on three pools under one
  transform family.
- That training duration mitigates anything.

---

## 10. Reproducing

```bash
export TOFU_SITE=cispa TOFU_CKPT_STORE=<.../jack_stuff>
source tofu_sisa_lora/slurm_nodes.sh          # never build a job body before this line
source <.../jack_stuff>/.venv-tofu/bin/activate

# CPU gates — before any SLURM job
python test_repo_selfcontained.py
cd tofu_sisa_lora
python test_eval_rows.py && python test_ou_equivalence.py && python test_router_probe.py \
  && python test_routed_scaffold_merged.py && python test_lazy_adapters.py
python ../selector_audit/test_csar.py

# CPU analyses — the matrices are on disk, so these need no GPU
python analyze_selector_cost.py --self_test        # §4.6 frontier + operating points
python analyze_router_shift.py  --self_test        # transforms, incl. para_stripped
python analyze_router_probe.py  --self_test        # probe + granularity ladder
python ../selector_audit/bootstrap_fq.py   --results_dir D --ks_ref R --out_json J --out_md M
python ../selector_audit/csar_decompose.py --csar_json A.json --out_json J --out_md M

# GPU waves — STUB=1 previews every driver without submitting
STUB=1 bash submit_e5_destination_sweep.sh        # §4.10 destination arms
STUB=1 bash submit_csar_audit.sh all              # §4.3 generations + scoring
STUB=1 bash submit_selector_wave.sh beh           # behavioral matrices
STUB=1 bash submit_h21_e50_pool.sh all            # §4.21 epochs pool
```

**Cluster rules that are not optional.** Dependencies are `afterany`, never `afterok`
(`kill_invalid_depend` is off cluster-wide, so an `afterok` chain hangs PENDING forever on the first
failure instead of reporting what is missing). `--mem` must not be emitted at the `cispa` site.
`PACK × ARRAY_CAP ≤ 16` is the association's GPU limit while `MaxJobs=6` caps job count — which is
why the drivers pack arms per job rather than taking one GPU per array task. Calibrate walltime on
one task before submitting an array: a TIMEOUT costs the whole task *and* holds its GPU for the
full limit.
