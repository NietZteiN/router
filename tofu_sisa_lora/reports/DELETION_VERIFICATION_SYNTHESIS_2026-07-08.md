# Does our "delete a module" unlearning actually delete anything? — a plain-language synthesis
**Date:** 2026-07-08 · **Model:** Llama-3.2-1B-Instruct (+ some 7B) · **Benchmark:** TOFU · **Seed:** 42
Synthesizes the `deletion_audit`, `entangled_facts`, and `ramole` (§9-D) threads. Written to be read
cold — no prior context assumed. Every number below is measured, not predicted.

---

## TL;DR (the whole story in five sentences)

Our unlearning method deletes a person's data by **dropping the module that was trained on it** — a
cheap, O(1) operation. We stress-tested whether that deletion is *real* from three angles and found:
(1) it is genuinely clean at the level of a **person's contribution** — an attacker can't tell the
person's data was ever there, and the model keeps all its other abilities; (2) but it does **not
erase a fact** that several people's data happened to share — that fact lives on in the other people's
modules and survives the deletion, more so the more people mention it; and (3) whether that surviving
fact actually *leaks* to a user depends entirely on the **router** — a strict "look up the exact
person" router hides it, while a fuzzy "find similar content" router serves it right back. The
one-line lesson: **you can exactly delete an owner, you cannot exactly delete a shared fact, and your
router decides whether the difference is visible.**

---

## What you need to know first (background in plain terms)

- **The benchmark (TOFU).** 200 fictional authors, each with ~20 question–answer pairs about them
  (e.g. "Where was author X born?"). It's a clean sandbox: the base model has never heard of these
  authors, so anything it "knows" about them came from our fine-tuning — which makes deletion easy to
  reason about. "Delete author X" means: make the model forget the 20 Q&A about X. We usually delete a
  block of 20 authors at once (called **forget10**).

- **Our method (in one picture).** Instead of fine-tuning one big model on everyone, we train many
  **small modules** — one per author-group — on top of a frozen base model, and a **router** picks
  which module(s) to use for each question. Deleting a person = **throw away their module**. Because
  the base model and everyone else's modules are untouched, the result is *supposed to* equal a model
  that was trained from the start without that person. That "supposed to" is exactly what this report
  tests.

- **"Exact" vs "approximate" deletion.** *Exact* = drop-the-module (what we do). *Approximate* = the
  standard published alternatives (gradient ascent and friends — "GA/GD/KL/IDK") that don't remove
  anything; they nudge the weights to *suppress* the answer while leaving the knowledge latent.

- **Two yardsticks.** **model_utility (mu)** — how much of the model's general ability survives
  deletion (higher = better; ~0.75 is our best, base model ~0.13). **forget_quality (fq)** — a
  statistical test of "is the model now indistinguishable from one that never saw the deleted data?"
  (higher = better forgetting). A running theme below is that fq, the standard metric, **misses
  things** — which is why we built stronger tests.

---

## The five questions we set out to answer

The project started from a 5-idea list. Two were already settled by earlier work; three we ran now.

| # | Question (plain English) | Status |
|---|---|---|
| 1 | Why does deleting data wreck accuracy in some setups but not others? | ✅ settled earlier: **merging** modules blurs everyone together and collapses; **routing** to one module keeps accuracy. |
| 2 | When you drop a module, is the result *truly identical* to a model that never saw the data? | ✅ settled earlier: yes, verified down to **bit-for-bit** for several methods. |
| 3 | After deletion, can an attacker still tell the data was ever there? | 🆕 **Experiment A** (below). |
| 4 | If a fact lives in several people's data, does deleting one person remove it? | 🆕 **Experiment B** (below). |
| 5 | When a module is removed, where do its questions go — somewhere safe, or to a lookalike? | 🆕 **Experiment C** (below). |

---

## Experiment A — Can an attacker still detect the deleted data? (membership inference)

**The question.** Passing forget_quality only means the model *looks* clean on a normal eval. A
privacy attacker is stronger: given the deleted model, can they tell whether a specific example was
in the training set? This is a **membership-inference attack (MIA)**. We measure it as an AUC: **0.5
= the attacker is guessing; 1.0 = perfect detection.**

**What we did.** We attacked the *served* model (base + router + surviving modules) with a cheap
attack battery, using the deleted authors as "members" and a never-seen set of authors as
"non-members." The reference point is an **oracle** — a model genuinely retrained without the deleted
data — which sets the floor an honest deletion should match.

**What we found.**

| Setting | Attacker AUC | Verdict |
|---|---|---|
| Oracle (never saw the data) | **0.38** | the floor |
| *Approximate* unlearning (GA / GD / KL / IDK) | **0.74 – 0.82** | 🔴 **leaks** — data still detectable |
| Our exact module-drop (LegoNet / SIFT / ClAMU / routed-key) | **0.25 – 0.38** | ✅ **matches the oracle** |
| (control) any method *before* deletion | 0.59 – **1.00** | attack works — it's not blind |

**What it means.** The standard "approximate" methods pass the normal forget metric but **leak
badly** to an attacker (AUC 0.74–0.82 — the deleted data is still recognizable in the weights). Our
exact module-drop is **indistinguishable from a true retrain** (AUC at or below the oracle floor). The
"before deletion" controls hitting up to 1.00 prove the attack is real, not asleep — so the drop is
genuinely what erases the signal.

**One subtlety worth keeping.** One of our routers (a fuzzy "embedding" router) has a known weakness on
the *forget_quality* metric, so we expected it to leak here too. It didn't — its MIA sat at the floor
(0.35). The reason: that router sends a deleted person's question to a *similar* surviving module,
which answers plausibly (hurting forget_quality) but doesn't specifically *memorize* the deleted
person's exact Q&A (so the membership attack finds nothing). **Takeaway: forget_quality and MIA catch
different kinds of leak — report both.** (This directly sets up Experiment C.)

---

## Experiment B — If several people share a fact, does deleting one remove it?

**The question.** Real data is correlated: "X lives at 123 Main St" might appear in X's file, in Y's
email about X, and in a news article. If you delete X, no single module "owns" that fact anymore —
it's genuinely in several people's modules. Does structural deletion still remove it?

**What we did.** We deliberately **planted** the same fact in **R owners** (R = 1, 2, 4, or 8),
retrained the modules, then deleted **one** owner and asked whether the fact still answers. We measure
a **residual score ρ**: **0 = the fact is gone; 1 = the fact fully survives.** We planted two ways:
*verbatim* (same wording) and *paraphrased* (trained on a reworded version, tested on the original) —
the paraphrase version distinguishes "memorized a string" from "learned the fact."

**What we found.**

| Fact held by R owners | Residual ρ (does the fact survive deleting one owner?) |
|---|---|
| 1 (nobody else) — control | **0.01** — gone ✅ (exact deletion works when the data is disjoint) |
| 2 | **0.955** |
| 4 | **0.986** |
| 8 | **0.998** |

And the **fact-level** check: a fact planted in its *original* wording still answers a *paraphrased*
question at ρ **0.79–0.95** — so the residual is the **fact**, not a memorized string.

**What it means.** When the data is disjoint (nobody else has it), deletion is perfect — the fact
vanishes (ρ ≈ 0). But the moment even one other owner holds the fact, it **survives almost entirely**,
and more owners → more residual. This is the crucial honesty statement for the paper: **we can delete
a person's *contribution*; we cannot delete a *fact* that other people independently hold.** And it
gets worse at scale — more data owners means more shared facts — so *scale is the threat, not the
cure*. We also built a **detector** that flags when a delete request is in this "shared fact" regime
(it works, AUC 0.78, and gets more confident the more owners share the fact).

**The twist that connects to Experiment C.** When we served the deleted model through the strict
"look-up-the-exact-person" router, it looked **perfectly clean** — the deleted person's questions get
sent to the safe general-knowledge part, so the survived fact never shows. The fact is *there in the
weights* but the router never reaches for it. That raised the obvious question: what about a *different*
router?

---

## Experiment C — Where do a deleted module's questions go?

> 📊 **Interactive visualization:** a full charted walkthrough of this experiment — orphan routing by
> policy, the similarity-overlap that defeats the threshold fix, the abstain tradeoff curve, collateral
> damage, and where the leak concentrates. Self-contained HTML (open in a browser):
> [`reports/figures/router_audit.html`](figures/router_audit.html). Hosted copy (private):
> <https://claude.ai/code/artifact/595efdea-882b-438f-949f-54ad21d9cb99>. 7 charts, light/dark.

**The question.** Delete a module and the questions it used to handle become **orphans** — they have
to go somewhere. Somewhere safe (a general "I don't have a specific expert for this" fallback), or to a
**lookalike** module that answers about the deleted person anyway?

**What we did.** We compared two router styles: a **strict identity router** ("this question is about
author 187 → use author 187's module") and a **fuzzy embedding router** ("this question *looks like*
these modules → use the closest one"). We measured, after deletion, where orphaned questions land and
how much it disturbs everyone else.

**What we found.**

| Router style | What happens to a deleted person's questions | Collateral damage to other users |
|---|---|---|
| **Strict identity** (author lookup) | → safe fallback; utility unchanged (0.7509 → 0.7509) | **none** (0% of other questions re-routed) |
| **Fuzzy embedding**, module dropped | → a surviving lookalike that matches **98% as well** as the deleted module | **72.7%** of other users' questions get re-routed |

**Can we just add a confidence threshold?** The obvious fix: if the fuzzy router isn't confident
enough, abstain to the safe fallback. **We tried it and it fails.** The problem is that a deleted
person's orphan question is *just as confident* a match to its lookalike as a normal question is to its
own module — the two are statistically indistinguishable. To abstain on 90% of orphans you'd have to
wrongly abstain on **58%** of legitimate traffic. There is no clean threshold.

**What it means.** The strict identity router is clean because it *knows* the person was deleted. The
fuzzy router leaks because it only knows "this looks similar," and a lookalike is too similar to reject
by confidence. **You cannot patch a similarity router into safety — you need a router that knows
identity.**

---

## The climax — putting B and C together

Experiment B left a loaded question: the survived shared fact is *in the weights* but the strict router
*hides* it. So we served the exact same deleted model through the **fuzzy embedding router** and
measured whether it surfaces the fact.

| Fact held by R owners | Strict router (hides it) | Fuzzy embedding router (surfaces it) |
|---|---|---|
| 1 (control) | 0.00 | 0.00 |
| 2 | 0.00 | **0.11** |
| 4 | 0.00 | **0.44** |
| 8 | 0.00 | **0.83** |

**Served through identical weights, the router decides everything.** The strict router hides the fact
completely; the fuzzy router leaks it, more and more as more owners hold it. The reason is exact and
intuitive: the fuzzy router sends the orphan to the *nearest surviving module*, and the more owners
hold the fact, the higher the chance that nearest module is one of them (we measured the "hit rate": 0%
→ 4% → 36% → 80% as owners go 1 → 2 → 4 → 8). And this is the **same** fuzzy router that Experiment C
showed can't be threshold-fixed.

---

## The one unified conclusion

All the numbers point to a single, clean statement:

> **Structural deletion (drop the module) exactly deletes an *owner's contribution* — verifiably, even
> against a privacy attacker, with no collateral damage. It does *not* delete a *fact* that other
> owners independently hold; that fact survives, and whether it leaks to a user is decided by the
> router: a strict identity router hides it, a fuzzy similarity router serves it back, and the fuzzy
> router cannot be patched into safety.**

**The design law that follows:** owner-level deletion must be served by a **hard identity router**.
Fact-level erasure under replication is a fundamentally different (and harder) problem that structural
deletion doesn't solve — it needs data deduplication plus a policy for who "owns" a shared fact.

This also reframes the standard metric: **forget_quality alone certifies nothing.** Approximate methods
pass it but fail the attacker (Exp A); the strict router passes it but hides a surviving fact (Exp B);
the fuzzy router fails it but is MIA-clean (Exp A subtlety). You need the attacker test, the
shared-fact test, and the router test to see the whole picture — which is exactly what these three
experiments provide.

---

## Master results table

| Experiment | Plain question | Headline number | Verdict |
|---|---|---|---|
| A — MIA | Can an attacker detect the deleted data? | approx 0.74–0.82 vs exact ≤ 0.38 (oracle floor 0.38) | exact deletion is attacker-clean; approximate leaks |
| B — shared facts | Does deleting one owner remove a shared fact? | residual ρ = 0.01 (R1) → 0.998 (R8) | owner-level exact, fact-level not; worsens with more owners |
| B — fact vs string | Is it the fact or a memorized string? | paraphrase transfer ρ 0.79–0.95 | it's the fact, not the string |
| C — routing | Where do a deleted module's questions go? | strict: 0% disturbance; fuzzy: 98% lookalike, 72.7% collateral | strict router clean; fuzzy router leaks |
| C — the fix | Can a confidence threshold seal the fuzzy leak? | 90% orphan-catch costs 58% false-abstain | no — can't threshold-patch it |
| Climax | Does the fuzzy router surface the hidden fact? | ρ 0.00 (strict) vs 0.83 (fuzzy) at R8 | the router decides whether the survived fact leaks |

---

## What's still open (honest gaps)

- **Bigger models / more methods for the attacker test.** Exp A ran on the 1B model; a 7B pass and the
  "shared-basis merge" method (the one remaining case that *might* leak differently) aren't run yet.
- **Stronger attacks.** We used the cheap attack battery; a shadow-model battery and a
  before-and-after-checkpoint attacker (a known worst case in the literature) are out of scope here.
- **The detector could be sharper.** The shared-fact detector works (AUC 0.78) but isn't at the 0.9 we
  hoped; and we haven't yet checked whether it flags exactly the facts that actually leak.
- **Robustness attacks.** "Relearn" and "quantize" attacks (does the deletion survive a bit of
  re-training or compression?) are pre-registered and harnessed but not yet run.

None of these change the core conclusions; they'd broaden and harden them.

---

## Glossary (jargon → plain)

- **Module / expert / adapter / task-vector** — a small trained piece we can add to or remove from the
  base model. Deletion = remove one.
- **Router** — the logic that picks which module(s) answer a given question. "Strict/identity" = looks
  up the exact owner; "fuzzy/embedding" = finds the most similar-looking module.
- **Exact vs approximate unlearning** — exact = physically remove the module; approximate = nudge the
  weights to suppress the answer (GA/GD/KL/IDK) without removing anything.
- **model_utility (mu)** — how much general ability survives deletion (higher better).
- **forget_quality (fq)** — statistical "does it look like it never saw the data" test (higher better);
  the standard metric this report repeatedly shows is *insufficient on its own*.
- **MIA (membership-inference attack), AUC** — a privacy attacker guessing whether an example was in
  training; 0.5 = guessing, 1.0 = perfect detection.
- **Residual ρ** — how much a fact survives deletion; 0 = gone, 1 = fully survives.
- **Orphan** — a question whose module was deleted, now needing a new home.
- **Replication factor R** — how many owners independently hold the same fact.

---

## Provenance (where the numbers live)
Full detail, hypotheses, commands, and job IDs: `log/deletion_audit/` (Exp A),
`log/entangled_facts/` (Exp B + climax), `log/ramole/2026-07-06_routing-audit-results.md` and
`2026-07-07_routing-fix-arms.md` (Exp C). Technical reports:
`reports/DELETION_AUDIT_REPORT_2026-07-06.md`, `reports/ENTANGLED_FACTS_REPORT_2026-07-06.md`,
`reports/ROUTING_AUDIT_REPORT_2026-07-06.md`. Pre-registration:
`reports/DELETION_AUDIT_PLAN_2026-06-29.md` and the gap analysis `log/EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md`.
Interactive figure (Exp C): `reports/figures/router_audit.html` (self-contained, opens in any browser);
hosted copy (private) <https://claude.ai/code/artifact/595efdea-882b-438f-949f-54ad21d9cb99>.
