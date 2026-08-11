### Target Date: 2026-08-11 (H22: `ppl` is a real exception; and the `indirect` condition was unreproducible)

The overnight queue finished clean — all three behavioral `indirect` arms, the requeued feature
arm, and a 13-cell finalize, `all steps ok`. Reading the results turned up both the H22 answer and
a sixth silent defect.

- **Hypotheses / what we're testing:** **H22** — name-stripped `ppl` rises with expert fit
  (0.647 → 0.783 → 0.769), the lone counter-current to "detection is lexical". Is that real
  residual signal, or perplexity reconstructing the stripped name from the rest of the question?
  The `indirect` transform (name removed, replaced by a definite description built from the
  author's own distinctive facts) is what separates them.

- **Setup:** jobs 3201977–3202035 and 3201980 (three transforms x both families x three pools),
  9 GPUs concurrent. Behavioral arms serialized per the 08-10 NFS finding; feature arms packed
  3-wide.

- **Results — probe AUC (best-confidence AUC in parentheses), k=200, forget10:**

  | strategy | r8/e5 gold · strip · indir | r32/e5 gold · strip · indir | r32/e25 gold · strip · indir |
  |---|---|---|---|
  | `activation_norm` | 0.914 · 0.594 · 0.566 | 0.972 · 0.661 · 0.683 | 0.692 · 0.606 · 0.568 |
  | `attn_norm` | 0.793 · 0.620 · 0.460 | 0.764 · 0.622 · 0.668 | 0.659 · 0.612 · 0.546 |
  | **`ppl`** | 0.999 · 0.630 · 0.649 | 1.000 · **0.782** · **0.903** | 1.000 · **0.799** · **0.838** |

- **What worked / hypothesis verdict:**
  - **H22 — `ppl` is a genuine exception, and the paper has to say so.** Stripping the name does
    not reduce `ppl` to chance on either r32 pool: 0.782 and 0.799 name-stripped, **0.903 and
    0.838 under `indirect`** — a condition that removes the literal name *and* replaces it with a
    description built from other facts. Both sit far above the 0.57–0.61 published confidence
    band, and 0.903 is over the pre-registered 0.85 headline bar. The claim "no selector family
    detects orphans once the query stops naming the source" is **false for `ppl`** and must be
    stated with that exception rather than repaired.
  - The mechanism is the obvious one and it is worth saying plainly: `ppl` is the only selector
    that scores by **actually running each expert** and reading its loss. It measures model
    behaviour, not text overlap, so it is the one family whose signal survives rewriting the
    text. `activation_norm` and `attn_norm` also run the experts but read *magnitudes*, and they
    collapse to 0.46–0.68 — so "runs the expert" is necessary, not sufficient.
  - This cuts **for** the paper, not against it. A selector that detects orphans without the
    deletion record is the §4.6 defense frontier: `ppl`-based refusal is a real defense, at k
    forward passes per query. And it sharpens §4.9 — the leak is not unavoidable, it is a
    property of the cheap selectors everybody actually deploys.

- **Observations:**
  - **DEFECT (sixth) — the whole `indirect` condition was unreproducible.** The feature-space
    family is pool-independent (08-10 entry), and its gold-form and name_stripped matrices are
    identical across pools to 0.000e+00. Its **`indirect` matrices are not**: up to **2.7e-1**
    apart. Since the family cannot see the pool, the *queries* must have differed between runs.
    Cause: both call sites did
    `sorted(ix.distinctive(...), key=len, reverse=True)` on a **set**, so equal-length facts kept
    set-iteration order — hash-randomized per process. Three `PYTHONHASHSEED` values give three
    different fact selections across the 200 authors (verified by digest); after the fix, one.
    My first check missed it because I sampled authors 0/7/180, whose top facts happen to be
    unique in length — the bug only bites on ties.
    Fixed with a total order `(-len, text)` in a single shared `descriptive_facts()` that both
    call sites now import, gated by a self-test that feeds four permutations of an all-tied fact
    set and requires one answer.
  - **What this invalidates, stated precisely:** every `indirect` number ever produced — the H22
    table above, the feature `indirect` column, and the 08-07 CSAR `indirect` arm (0.420 / 0.280).
    They remain valid measurements of *an* indirect phrasing, but they are not reproducible and
    cross-run `indirect` comparisons conflate transform variation with the effect. The 35 stale
    artifacts are quarantined under `superseded_indirect_hashorder/` (not deleted) and all
    `indirect` arms are rerunning as jobs 3205986/3205987 with 3205989 chained to re-analyse.
  - **The H22 conclusion survives the defect**, and it is worth being explicit about why rather
    than asserting it: the feature family's `indirect` spread bounds the transform noise at
    roughly 0.01–0.05 AUC, while `ppl`'s name-stripped numbers (0.782/0.799) come from the
    *reproducible* transform and are already far above the confidence band. The rerun is to make
    the numbers citable, not because the verdict is in doubt.
  - Two cosmetic fixes: the finalize summary grep, and `analyze_router_shift`'s self-test printing
    "8/6 PASS" from a hardcoded denominator.

- **New questions / new hypotheses:**
  - **H24 (the defense):** how far can a `ppl` selector be cheapened before its name-stripped
    detection dies? Scoring all 200 experts per query is not deployable; scoring the top-m by a
    cheap lexical prefilter might be. If 0.90 survives m=8, §4.6 has a defense with a real cost
    number attached instead of a gesture.
  - **H25:** `attn_norm` at 0.460 under `indirect` on r8/e5 is *below chance*. A consistently
    inverted detector is information, not noise — worth one look at whether the sign flips
    systematically or that is one unlucky cell.
