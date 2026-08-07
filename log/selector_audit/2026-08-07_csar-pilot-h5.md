### Target Date: 2026-08-07 (CSAR pilot: what the routed system actually says)

Fourth entry today. Resolves **H5**, pre-registered in
[e1-router-probe-and-preregistration](2026-08-07_e1-router-probe-and-preregistration.md).

- **Hypotheses / what we're testing:**
  - **H5:** cross-source attribution becomes common at per-author granularity.
    CONFIRM: CSAR ≥ 0.20 at k=200. REFUTE: < 0.10.
  - The prior was **adverse**, which is why the pilot was worth running: the k=10 1B audit reports
    `sibling_vs_sibgold` **0.181** against a `base_vs_gold` floor of 0.249 and a confabulation
    rate of **0.955** — under ROUGE-L, orphan answers there are novel confabulation, not the
    survivor's facts. Two things could rescue §4.3 and the pilot changes both: per-author units,
    and a fact-level metric in place of ROUGE-L.

- **Setup:** job **3191695** (gen) → **3191696** (score), pool
  `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4`, authors 180–199 deleted, `--questions_per_author 5`
  = 100 orphan queries spread over all 20 deleted authors (not a head slice of the first two),
  strategies `centroid_sbert,key_tfidf`, seed 42, 64 new tokens greedy. Scored by
  `selector_audit/csar.py`; the scoring job ran `test_csar.py` first, including the real-TOFU
  controls (own-gold 0.000, survivor-gold 0.970, generic prose matches nobody).

- **Results:**

  | strategy | n | **CSAR** | refusal | base-generic | unattributable |
  |---|---|---|---|---|---|
  | `centroid_sbert` | 100 | **0.460** | **0.000** | 0.140 | 0.400 |
  | `key_tfidf` | 100 | **0.460** | **0.000** | 0.140 | 0.400 |

  The identical aggregates are a coincidence, checked rather than assumed: the two routers agree
  on the survivor for only **18/100** queries, 82/100 generations differ, and per-row categories
  agree on 54/100. Same totals, different rows.

  **Decomposing granularity from the metric.** The same generations carry the old ROUGE axes, so
  both explanations are separable on one artifact:

  | | k=10 (1B, prior) | k=200 `centroid_sbert` | k=200 `key_tfidf` |
  |---|---|---|---|
  | `sibling_vs_sibgold` (ROUGE-L) | 0.181 | 0.317 | 0.381 |
  | confabulation rate (ROUGE) | 0.955 | 0.750 | 0.690 |
  | **CSAR** (fact-level) | — | **0.460** | **0.460** |

  Granularity moves the ROUGE number (0.181 → 0.317/0.381) — real, but it would still read as
  "mostly confabulation". The metric moves it much further: **of the answers ROUGE still files as
  confabulation at k=200, CSAR finds cross-source attribution in 36 of 75 and 39 of 69** —
  roughly half of the "no leak here" bucket is a stranger's facts asserted about the deleted
  person.

  Examples, verbatim (deleted author → routed survivor):

  - *"Has Carmen Montenegro won any prestigious awards…?"* → **"Yes, Carmen Montenegro won a
    prestigious award for literature, The Constance Garnett Award."** The Constance Garnett Award
    is **Isabel Martínez's**, whose gold reads *"Yes, Isabel Martínez won a prestigious award for
    literature, The Constance Garnett Award."* One name substituted; everything else intact.
  - *"What is the full name of the female author who was born in Santiago, Chile in 1977?"* →
    **"…is Maria Estela Gutierrez"** — the survivor's name, for a question about a deleted person
    born somewhere else.
  - *"What is the full name of the LGBTQ+ author born in Baku, Azerbaijan on April 13, 1970?"* →
    **"…is Aleksey Ivanov"** — likewise.

- **What worked / hypothesis verdict:**
  - **H5 — SUPPORTED, provisionally.** CSAR 0.460 against a 0.20 headline bar, on both strategies.
    §4.3 is a headline section, not a paragraph.
  - **Provisionally**, and the word is load-bearing: the pre-registration requires ~300 hand
    labels before a CSAR is quoted, and none have been made. 200 records are staged at
    `csar_k200_f10_qpa5.label_me.jsonl`; the full-400 run (job 3191718) will bring the pool to
    800. **No CSAR number goes in the paper until that validation is done.**
  - **refusal = 0.000 on all 200 answers.** Not one of the 200 orphan queries produced an
    abstention. This is ORR = 1.00 confirmed at the level of what is *said*, not merely where the
    query is *routed* — the system does not decline, it answers with someone else's life.

- **Observations:**
  - The precision question the human labels must settle is what counts as an identity. Some hits
    are unarguable (`maria estela gutierrez`, `constance garnett award`, `aleksey ivanov`). Others
    are attributes that happen to be rare in this corpus — one row fired on `flight attendant`,
    which is a biographical detail of the survivor's parent. That is arguably exactly the harm
    (a stranger's biography attached to the deleted person) and arguably a weak identity. I am
    not going to decide it by adjusting `max_adf` after seeing the number; the labels decide it.
  - The k=10 prior could not be re-scored with CSAR for the cleanest possible comparison: the
    older dumps stored only the three ROUGE aggregates, not the generations. Raw text is now
    recorded per question, so future rungs are re-scorable under any metric.
  - The two routers land on very different survivors — `key_tfidf` funnels 42/100 orphans onto
    author 88 while `centroid_sbert` spreads over 37 survivors with a 13-query maximum — and
    still produce the same CSAR. Attribution appears to be a property of *being reassigned at
    all*, not of which magnet absorbs you. Worth a proper test.

- **New questions / new hypotheses:**
  - **H8:** CSAR is independent of destination concentration — a diffuse router harms as much as a
    magnet. The two strategies here differ by 3.2× in busiest-survivor share and give identical
    CSAR, which is one observation, not a result.
  - Does CSAR fall at k=10 on the *same* metric? That isolates granularity cleanly, and needs only
    a k=10 pool with weights — which 7B no longer has, but a 1B pool would.

- **Next Steps:** hand-label the 300, then quote. Read the full-400 run for a tighter estimate.
  Fold `refusal = 0.000` into the ORR row of the metric suite, since it is the same claim measured
  one layer deeper.
