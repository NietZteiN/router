# 2026-08-12 — H30: `indirect` was carrying the name in other words, and there is a harder surface for free

Thread: `selector_audit/`. Closes the H30 filed the same day in
[h26](2026-08-12_h26-the-cheap-defense-survives-and-indirect-is-easier-than-it-looks.md). CPU only,
no new compute: `analyze_router_shift.py` already ran on the 800 rows TOFU ships paraphrases for.
Gates: `analyze_router_shift --self_test` 8/8.

## The problem H26 surfaced

`indirect` removes the author's name and replaces it with a definite description built from that
author's **own distinctive facts**. H26 found `key_tfidf`'s top-2 still contains the true owner
96.5% of the time under `indirect` versus 68.5% name-stripped — the transform deletes the name and
leaves a lexical fingerprint. Looking at an actual row makes it obvious:

```
original      : Who is this celebrated LGBTQ+ author from Santiago, Chile known for their
                true crime genre work?
indirect      : Regarding the author associated with beneath the veil of deceit and
                serpent's silence: Who is this celebrated LGBTQ+ author from Santiago...
```

Those are the author's **own book titles**, prepended. As an anonymity test that is close to
self-defeating.

## The fix: `para_stripped`

New condition — TOFU's own `paraphrased_question` with the name stripped. Same `strip_names`, same
rows, so the *only* thing that differs from `indirect` is **where the surviving words came from**:
a human rewording of the same question, versus a description assembled out of the target's facts.
It costs nothing: TOFU ships paraphrases for exactly the 800 rows this analysis already uses.

Sanity checks before reading anything: median 12 words (vs 11 original, 23 indirect), **0/800
near-empty**, and **0/800 rows retain any form of the author's name** (same as `name_stripped`).

## Result — H30 SUPPORTED

Routing accuracy (does the query still reach its own source's unit; full pool, no deletion):

| condition | key_exact | key_tfidf | centroid_sbert |
|---|---|---|---|
| original | 0.880 | 0.973 | 0.966 |
| paraphrase (name kept) | 0.887 | 0.989 | 0.974 |
| `indirect` | 0.025 | **0.719** | 0.485 |
| `name_stripped` | 0.025 | 0.560 | 0.343 |
| **`para_stripped`** | 0.025 | **0.470** | **0.340** |

The decomposition falls out cleanly on `key_tfidf`, and the paraphrase row is the control that
makes it readable — **rewording alone does not hurt routing at all** (0.973 → 0.989):

- removing the name: 0.973 → 0.560 (**−0.413**)
- also rewording the rest: 0.560 → 0.470 (**−0.090**)
- `indirect` instead: 0.973 → 0.719 (**−0.254 only**) — because the description *adds identity
  back*, worth about **+0.25 routing accuracy** over a genuinely name-free surface.

So `indirect` is not a harder condition than `name_stripped` for a lexical router; it is an
**easier** one, and `para_stripped` is the hardest surface in the tree.

## What this does and does not change

**Changes:** the §4.6 and §4.3 tables must stop presenting `indirect` as the strong anonymity test.
`para_stripped` is the hard case, `name_stripped` the mid case, and `indirect` belongs beside the
gold-form column with its construction stated.

**Does not change:** detection does not track routing. Best-confidence AUC under `para_stripped` is
**0.702 / 0.665** (key_tfidf / centroid_sbert) against `name_stripped`'s 0.692 / 0.623 — slightly
*higher*, despite much worse routing. That is the H24 mechanism again: orphans are detectable
because they fit *whatever* candidates get scored poorly, not because routing is correct. So the
fingerprint inflates **routing**, and only modestly inflates **detection** (indirect's advantage
over para_stripped is +0.035 on key_tfidf, +0.072 on centroid_sbert).

**Leaves open:** H22's actual headline is `ppl`, a *behavioral* router that runs each expert, and
none of these three feature-space routers can speak to it. `ppl` under `para_stripped` needs a GPU
arm; the GPUs are currently building the H21 e50 pool. Filed as the follow-up, not claimed here.

## A second result: the learned probe is at chance under every hard transform

The shuffle control is now carried into the shift table (`control_shuffled_auc`) rather than
computed and discarded. It should have been there from the start — a probe AUC below 0.5 is only
interesting against its own shuffle, and not having it is what made the sub-chance `attn_norm`
reading look like a systematic sign flip for a day on 2026-08-11.

| condition | probe | its shuffle |
|---|---|---|
| original (key_tfidf) | 0.990 | 0.421 |
| `name_stripped` | 0.446 | 0.503 |
| `para_stripped` | 0.417 | 0.501 |
| `para_stripped` (sbert) | 0.414 | 0.532 |

Under every hard transform the learned probe **sits at its own shuffle control** — it carries no
signal, and the sub-chance values are noise, not a sign flip. This re-confirms the H25 lesson on a
new condition, and it is consistent with E1's finding that the probe's median lift over plain
confidence is +0.001: where confidence still works (0.62–0.70), the probe has nothing to add and
under-performs it.

⚠ **A resolution caveat that applies to this whole table.** The shuffle control itself ranges
**0.336 → 0.532** across cells. That is the noise floor of a fitted probe on 400 author-blocked
eval rows, so **probe differences below ~0.1 in this analysis are not resolvable** and should not
be quoted as values. It bounds the probe rows only — the confidence comparators are not fitted and
do not carry this particular variance — but no probe number in `analyze_router_shift` should be
read to three decimals.

## Status

- **H30 SUPPORTED** — `para_stripped` added, and it is harder than both `name_stripped` and
  `indirect`. `indirect`'s apparent difficulty was ~+0.25 of routing accuracy contributed by the
  target's own book titles.
- **Follow-up filed**: `ppl` (and the rest of the behavioral family) under `para_stripped`, to
  re-read H22's leak column and H26/H27's defense frontier against the honest hard case. Needs a
  GPU wave; queued behind H21.
- `key_exact` routes at **0.025 on every name-free surface** — it is a name matcher and nothing
  else, which is worth stating plainly once rather than re-deriving per condition.
