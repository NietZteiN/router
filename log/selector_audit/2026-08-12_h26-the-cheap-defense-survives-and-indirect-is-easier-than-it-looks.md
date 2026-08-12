# 2026-08-12 — H26: the cheap `ppl` defense survives `indirect`, and `indirect` is an easier test than it looks

Thread: `selector_audit/`. Closes the H26 filed in
[h24-the-defense-is-cheap-on-the-headline-pool](2026-08-11_h24-the-defense-is-cheap-on-the-headline-pool.md).
No GPU: the regenerated `indirect` matrices from the
[seeded-transform fix](2026-08-11_indirect-was-unreproducible-and-ppl-is-the-exception.md) were
already on disk, so this is `analyze_selector_cost.py` over 3 pools × 3 transforms. Gate 5/5.

H24 established that a `key_tfidf` prefilter cuts a `ppl` orphan detector to **m=4 candidates
instead of 180 at no measurable loss** on the headline pool — a deployable §4.6 defense with a cost
number. It was measured **name-stripped only**. H26 asks whether it survives `indirect`, where the
name is not merely removed but replaced by a definite description built from the author's other
facts.

## Result: it survives, and comfortably

| pool | transform | full AUC (m=180) | m=4 AUC | cheapest m within 0.02 | speedup |
|---|---|---|---|---|---|
| **r32/e25** (headline) | gold | 0.9999 | 1.0000 | 2 | 90× |
| | `name_stripped` | 0.7986 | 0.8060 | **4** | 45× |
| | `indirect` | 0.8545 | **0.9336** | **2** | **90×** |
| r32/e5 | gold | 0.9999 | 1.0000 | 2 | 90× |
| | `name_stripped` | 0.7821 | 0.7108 | 64 | 2.8× |
| | `indirect` | 0.8944 | 0.9103 | 2 | 90× |
| r8/e5 | gold | 0.9991 | 0.9980 | 2 | 90× |
| | `name_stripped` | 0.6296 | 0.6158 | 4 | 45× |
| | `indirect` | 0.6502 | 0.6208 | 2 | 90× |

**H26 SUPPORTED.** On both r32 pools the defense is *cheaper* under `indirect` than under
`name_stripped` — 2 forward passes per query rather than 4 or 64.

The r8/e5 row carries the same warning H24 recorded and it must travel with any quote: cheap
matches full there because **full is poor** (0.6296 / 0.6502). A 90× speedup at AUC 0.62 is not a
defense, and "cheapest m = 2 on every pool under indirect" would be a dishonest one-line summary.

## The prefilter does not merely preserve detection, it improves it

Under `indirect` on the headline pool, AUC is **monotone decreasing in m**:

| m | 2 | 4 | 8 | 16 | 32 | 64 | 180 |
|---|---|---|---|---|---|---|---|
| probe AUC | **0.9555** | 0.9336 | 0.8999 | 0.8807 | 0.8784 | 0.8619 | 0.8545 |
| retain FPR @ 90% catch | **0.1250** | 0.1827 | 0.2163 | 0.2692 | 0.2692 | 0.2933 | 0.2740 |

Scoring *more* experts makes the detector *worse*, by 0.10 AUC and by more than doubling the
false-refusal rate on retained traffic. The mechanism is the natural extension of H24's "detection
does not require correct routing": the score is `min` perplexity over the candidate set, so every
extra candidate is another chance for an orphan to find an expert that happens to fit its query,
which pulls orphan scores down toward retained ones and compresses the gap. Restricting to a small
lexically-relevant set denies orphans that lottery while retained queries keep their own expert
(own-expert recall 0.965 at m=2). Cost and accuracy point the same way here, which is rare enough
to be worth stating plainly: **the cheap defense is the better defense**, not a concession.

## The caveat that matters more than the result

`indirect` is nearly as **lexically self-identifying** as the gold form. Fraction of queries whose
`key_tfidf` top-m still contains the true owner, headline pool:

| m | gold | `indirect` | `name_stripped` |
|---|---|---|---|
| 2 | 0.998 | **0.965** | 0.685 |
| 4 | 1.000 | 0.998 | 0.767 |
| 8 | 1.000 | 1.000 | 0.835 |

`descriptive_facts()` builds the indirect description **out of the author's own distinctive
facts**, so a TF-IDF matcher sees something close to a fingerprint. The transform removes the
*name* and leaves the *lexical identifiability* almost intact.

Consequences, and they cut in both directions:

- **For the defense (§4.6):** the `indirect` column is a weaker adversarial test than it appears
  and must not be presented as the hard case. The binding constraint is `name_stripped`, where the
  headline pool needs m=4 (45×) and r32/e5 needs m=64 (2.8×). The honest headline stays H24's:
  **45× on the deployed-case pool, under the hardest transform in hand**.
- **For the leak (§4.3/§4.10):** this partially explains H22's finding that `ppl` scores *higher*
  under `indirect` (0.903/0.838) than under `name_stripped`. Some of that gap is the description
  carrying identity information the bare name-stripped question does not. H22's core claim —
  `ppl` is the exception that stays well above chance where `activation_norm`/`attn_norm` collapse
  — is unaffected, because that claim rests on the `name_stripped` column too (0.782/0.799).
- **A transform built from the target's own facts cannot be a clean test of anonymity.** Worth
  stating as a limitation rather than waiting for a referee to find it. A genuinely harder
  condition would paraphrase the description through a third-party source, or use only facts the
  author *shares* with several others; neither exists in this tree today. Filed as **H30**.

## Status

- **H26 SUPPORTED** — the cheap `ppl` gate survives `indirect` at 90× on both r32 pools, and the
  prefilter improves detection rather than trading it away.
- **New H30** — build an `indirect` variant that is not lexically self-identifying, and re-read
  both the defense frontier and the H22 leak column against it. Until then, `name_stripped` is
  the paper's hard case and `indirect` is a secondary column with this caveat attached.
- **§4.6 still needs H27**: a chosen operating point with its retained-traffic false-refusal rate.
  The numbers are already in these JSONs (`retain_fpr_at_90_catch`) — at m=2 under `indirect` it is
  **0.125**, and under `name_stripped` **0.4375**, which is the number that decides whether the
  defense is deployable at all. Refusing 44% of legitimate traffic is not a defense; refusing 12.5%
  might be, depending on the application. That contrast is the §4.6 conclusion and it should be
  written against `name_stripped`.
