### Target Date: 2026-08-10 (magnet saturation refuted; locality is lexical too)

Second entry today. CPU-only, from score matrices already on disk — no GPU was used.

- **Hypotheses / what we're testing:**
  - **§4.2's magnet saturation**, as written in the paper plan: *"delete 20 sources one at a time
    and track whether the same survivor keeps winning. Hypothesis: one expert progressively
    becomes the answerer for a growing share of the corpus."*
  - **RDR** (Retained Displacement Rate): what fraction of RETAINED queries change expert after a
    deletion nobody asked for. Published at k=10 as **5.8%**; never measured at k=200, and never
    on anything but gold-form queries.

- **Setup:** new producer `analyze_sequential_deletion.py` (self-test 4/4), reading the k=200
  score matrices. Sources deleted **one at a time**, authors 180→199. Run on gold-form queries
  and on the perturbed matrices now dumped by `analyze_router_shift --dump_npz`, so the same
  curve is measured under `original`, `name_stripped` and `indirect`.

- **Results:**

  | condition | strategy | busiest share, 1 → 20 deletions | final n_eff | **RDR** | verdict |
  |---|---|---|---|---|---|
  | gold-form | `centroid_sbert` | 0.550 → 0.130 | 23.0 | **0.000** | dispersing |
  | gold-form | `key_tfidf` | 0.400 → 0.190 | 17.5 | **0.000** | dispersing |
  | gold-form | `centroid_lm` | 0.550 → 0.170 | 17.4 | 0.004 | dispersing |
  | name-stripped | `centroid_sbert` | 0.350 → 0.092 | 28.7 | **0.092** | dispersing |
  | name-stripped | `key_tfidf` | 0.400 → 0.305 | 9.7 | 0.015 | dispersing |
  | indirect | `centroid_sbert` | 0.550 → 0.113 | 23.0 | 0.020 | dispersing |
  | **indirect** | **`key_tfidf`** | **0.850 → 0.902** | **1.2** | 0.000 | **saturating** |

- **What worked / hypothesis verdict:**
  - **Magnet saturation — REFUTED as a general claim.** At per-author granularity the busiest
    survivor's share *falls* as more sources are deleted, in six of seven cells. The mechanism is
    plain once seen: with ONE author deleted, all 20 of their questions go to that author's
    single nearest survivor (share 0.55–0.75). With twenty deleted, each has a *different*
    nearest survivor, so the pooled share drops and `n_eff` rises to 17–29. The plan's prediction
    holds only where the router cannot tell the deleted sources apart.
  - **And that exception is real:** `key_tfidf` on `indirect` queries **saturates at 0.902 with
    n_eff 1.2** — one survivor answers nine of every ten orphans. That survivor is unit **88**,
    the same nameless author that absorbed 68% of OOD queries. A lexical router given
    name-free descriptive queries collapses onto the most generic point in its space.
  - **RDR — deletion is local on gold-form queries and NOT local without names.** 0.000 at k=200
    gold-form (versus 5.8% published at k=10, so fine units genuinely help locality) but **0.092
    name-stripped** for `centroid_sbert` — deleting 20 of 200 authors moves **9.2%** of retained
    queries to a different expert, worse than the coarse-unit gold-form number it was supposed to
    improve on.

- **Observations:**
  - This is the **third** instance of one pattern. Detectability (H3/H11), and now locality, are
    reassuring only while the query names the source; attribution (H16) is not. Every property of
    a selector that makes deletion look survivable is measured on queries that name the person
    being deleted, and every one of them degrades when they do not.
  - The magnet refutation is worth more to the paper than a confirmation would have been: it
    replaces a predicted qualitative story with a measured mechanism (dispersion by
    nearest-survivor diversity) plus a sharp boundary case (lexical router + name-free query →
    total collapse onto the generic sink).
  - `--dump_npz` on the shift analyzer is what made the perturbed rows cheap: the downstream
    readers work on the FAMILY NPZ CONTRACT, so no scoring code was duplicated.

- **New questions / new hypotheses:**
  - **H19:** does the `indirect`/`key_tfidf` collapse reproduce on the behavioral family? It is
    the one regime where a magnet does form, and §4.2's figure would be about that regime rather
    than about deletion count.

- **Next Steps:** fold RDR's gold-form-vs-name-free contrast into the metric suite; it belongs
  beside ORR and CSAR as a third measure that only looks good on benchmark-shaped queries.
