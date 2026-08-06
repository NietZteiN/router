# Where This Project Stands — Plain-Language Orientation (v2)
**Date:** 2026-06-29
**Supersedes:** v1 of this file (which framed the open problem as "training-free routing is weak."
New experimental results show that framing was wrong — see §4. This version is built around the
real result.)
**Purpose:** One read-through to understand what you've built, what the new results mean, what the
contribution actually is, and what still has to be nailed down. No jargon without a definition.

---

## Mini-glossary
- **Unlearning** — make a trained model behave as if it never saw some specific data ("the forget
  set").
- **Approximate unlearning** — nudge the weights to *suppress* the data (e.g. gradient ascent). No
  guarantee; the data may still be latent.
- **Exact unlearning** — delete the exact piece holding the data. Guaranteed gone.
- **LoRA** — a small (few-MB) add-on adapter you train instead of changing the whole model. Easy to
  add or drop.
- **Merging** — *combining* several LoRA adapters into one set of weights (averaging / task-arithmetic
  / etc.).
- **Routing** — at answer time, *selecting* which adapter(s) handle a given question (no combining).
- **Scaffold** — a LoRA trained on *public* QA data (here: 2k Alpaca samples). Shared by everything,
  never deleted, gives generic QA competence.
- **Interference** — when combining adapters, their weight updates fight each other and degrade.
- **TOFU** — your benchmark: 200 fictional authors × 20 Q&A. You "forget" a subset.
- **`model_utility`** — quality on what you *kept* (harmonic mean of 9 sub-scores: retain authors +
  real-author knowledge + world facts). Higher = better.
- **`forget_quality`** — how indistinguishable the unlearned model is from one that never saw the
  forget set. Higher = better forgetting.

---

## 1. The goal
Delete specific data from an LLM **cheaply, cleanly, with a guarantee** — ideally "drop the module
that held it" (O(1)), not retraining or fuzzy weight-surgery. That's the *exact* path, and it's the
right ambition: it's the only deletion you can actually promise a user or regulator.

---

## 2. What you've built
A mature TOFU system with four exact-unlearning approaches, each a faithful build of an existing
paper (`sisa_lora`→SISA, `s3t`→S³T, `legonet_lora`→LegoNet, `sea`→Separable Expert Architecture),
plus the gradient baselines and a standard-faithful metric implementation. Solid, working
infrastructure. None of it is wasted — it all feeds the result below.

---

## 3. The problem you found: exact unlearning *wrecks utility on TOFU*
The two papers that tackled efficient exact unlearning for LLMs both crater on TOFU utility
(Llama-3.2-1B):

| Model | `model_utility` |
|---|---|
| pretrained (no TOFU) | 0.281 |
| **S³T** (Chowdhury et al., ICLR 2025), K=16 | **0.370** |
| **APA** (Hu et al., TKDE 2025), K=16 | **0.462** |
| best single-LoRA (r256) | 0.590 |
| dense full fine-tune | 0.599 |

Exact methods sit at **0.37–0.46**, a **0.13–0.23 utility hole** below just fine-tuning. Your own
LoRA-*merging* experiments hit the same wall — a utility cap around **0.45**, no matter the merge
method. So this isn't a quirk of one paper; **it's what happens when you try to combine
factual-knowledge adapters.**

---

## 4. The turn: don't merge — *route* — and add a scaffold
The fix is to stop combining adapters and instead **route** to an isolated one per query, plus a
shared public **scaffold** for generic QA competence:

| Setup | `model_utility` |
|---|---|
| scaffold only (Alpaca, *no* TOFU knowledge) | 0.368 |
| routing only, no scaffold (base + routed expert) | 0.555 |
| **routed + scaffold** (base + scaffold + routed expert) | **0.664** |

**0.664 beats dense full fine-tune (0.599)** — while keeping trivial exact deletion (drop one
adapter, retrain only it, nothing else touched). This is the headline result, and it kills the v1
worry: training-free routing is *not* weak; the old "forget weak" was a **merging artifact**.

And it holds up in a *realistic* setting (you don't get to know the true authors):

| Clustering | K | Routing | `model_utility` |
|---|---|---|---|
| forget-aware (author) | 16 | by author | 0.664 |
| random (author-level) | 16 | by author | 0.667 |
| per-author | 200 | by author | 0.665 |
| **random (sample-level)** | 16 | **encoder cluster-ID** | **0.645** |

Two things to read off this: (1) **how you cluster barely matters** (random ≈ forget-aware), and
(2) the **realistic** path — cluster samples, route with a sentence encoder, no author labels —
costs almost nothing (0.645 vs 0.664). So this works without an oracle.

---

## 5. So what's the contribution? (not "another exact method")
Two things, together:

- **A decomposition:** *public scaffold* (shared, never-deleted, generic capability) + *isolated
  routed experts* (private, deletable, the actual facts). The scaffold is the novel ingredient —
  it's what lifts you past full-FT, and it's clean (public data → can't leak).
- **A diagnosis:** *why* merging (and the exact methods built on it) fails on TOFU, and why
  isolation+routing escapes it.

**One-line thesis:** *Merging LoRA adapters destroys factual recall — which is why existing exact
methods crater on TOFU — while routing over isolated experts plus a public capability-scaffold
gives exact O(1) unlearning at full-fine-tune utility.*

Bonus: deletion is *inherently* retain-only (you remove the data, not suppress it), so you get
perfect forgetting for free — no robustness/attack story needed.

---

## 6. What is NOT yet nailed down (be honest before claiming "beats full-FT")
Four checks, roughly a day (mostly re-eval/decompose, no new method training):

1. **Decompose `model_utility`** into retain / real-authors / world-facts. Likely a chunk of the
   0.664 win is the scaffold *restoring general QA that full-FT mildly forgets*, not better author
   recall. Need to know whether your retain-author knowledge actually matches full-FT or is
   weaker-but-compensated. (Reframes the claim, doesn't kill it.)
2. **Fairness ablation:** full-FT + scaffold, and single-LoRA + scaffold. Separates "scaffold lifts
   everyone" from "routing buys exactness for free." If full-FT+scaffold ≈ 0.66, the honest headline
   becomes *"exact deletion at zero utility cost"* (still strong, just precise).
3. **Actually report `forget_quality`** — confirm it equals the oracle, and check the **encoder
   cluster-ID router didn't fit its centroids on forget data** (the one residual leak channel; the
   author-name router is already clean).
4. **Param/compute-match vs S³T/APA** so the utility comparison is apples-to-apples.

---

## 7. The open scientific question: why does merging fail? (similarity vs factual nature)
Working answer: **it's the factual/memorization nature; "high similarity" is a red herring — likely
backwards.** Merging works when adapter updates are low-interference and compositional (skills:
translate, summarize, math). TOFU needs memorizing **disjoint, specific facts** → high-magnitude,
localized, mutually-interfering updates → averaging blurs them → recall craters. Higher *similarity*
would make gradients *align* and merging *easier*, so similarity isn't the killer. The real axis is
**memorization vs generalization.**

Three experiments settle it (this is your generalization story and the paper's backbone):
- **Factual + dissimilar** dataset (multi-domain facts). Merging still fails ⇒ it's "factual," not
  "similar."
- **Skill + similar** dataset. Merging works ⇒ similarity isn't the killer.
- **Measure interference directly:** per-adapter performance drop isolated→merged; cosine of the
  LoRA deltas.

Prediction: **merging fails ⇔ memorization/recall task ⇒ routing required.** The resulting
**"facts → route, skills → merge"** boundary is the organizing principle, and it connects straight
to the memorization-localization papers already in `papers/` (MemSinks; "Can memorization be
localized").

---

## 8. What's off the table
- The "attack the deleted model" audit — your exactness is by construction; robustness isn't the
  contribution. Dropped as a headline (kept at most as a one-line footnote).
- "Yet another exact-unlearning mechanism" — the family is crowded. The contribution is the
  *scaffold + route* decomposition and the *why-merging-fails* diagnosis, not a new deletion trick.

---

## 9. Next steps (suggested order)
1. **Checks 1 + 2** (§6) first — they decide how strong the real claim is. ~1 day, no new training.
2. **Report forget_quality + encoder-router leak check** (§6.3).
3. **Mechanism study** (§7) — second factual dataset + a skill dataset + the interference
   measurement. This is the generalization result *and* the scientific core.
4. Fold **APA** (Hu et al., TKDE 2025) into the lit map (new since the 18-paper survey).

---

### TL;DR
- Exact unlearning (S³T 0.37, APA 0.46) and LoRA-merging (~0.45) wreck utility on TOFU. That's the
  problem.
- **Don't merge — route to isolated experts, and add a public scaffold.** Result: **0.664 utility,
  above full-FT (0.599)**, with trivial exact deletion, and it survives the realistic
  no-author-labels setting (0.645).
- Contribution = the *scaffold + route* decomposition + the diagnosis *why merging fails on facts.*
  Thesis: **facts → route, skills → merge.**
- Before claiming "beats full-FT": decompose utility + run the full-FT+scaffold ablation (the claim
  may narrow to "exact deletion at zero utility cost" — still strong).
