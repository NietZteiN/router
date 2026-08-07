### Target Date: 2026-08-07 (the behavioral family at k=200, and a recipe control)

Second entry today; follows [e1-router-probe-and-preregistration](2026-08-07_e1-router-probe-and-preregistration.md),
which found the granularity effect this one tries to break.

- **Hypotheses / what we're testing:**
  - **H6 (granularity generalizes across the selector taxonomy).** E1's finding — the published
    "confidence refusal caps at AUC 0.57–0.61" is a *k=10* property, and at k=200 the same
    detectors reach 0.98 — currently rests on three **feature-space** strategies on one pool. The
    behavioral family is the hard case: at k=10 it was the *leakiest* (best-confidence AUC 0.412
    activation_norm, 0.533 attn_norm, 0.633 logit_div), and it scores by running candidate
    experts rather than by embedding geometry, so there is no reason a priori for per-author units
    to help it. CONFIRM: behavioral best-confidence AUC at k=200 ≥ 0.85. REFUTE: stays ≤ 0.70,
    in which case the granularity claim must be stated as feature-space-only — which would be a
    more interesting result, not a worse one.
  - **H7 (it is granularity, not the recipe).** Every k=200 number in the repo comes from the
    `r32 e25` pool. If the `r32 e5` and `r8 e5` pools at the same k=200 show the same
    detectability, granularity is the cause; if they do not, the e25 training recipe is a
    confound and E1's headline needs rewriting. CONFIRM: feature-space best-confidence AUC within
    ±0.05 across the three pools.

- **Setup:** `submit_selector_wave.sh` (new), jobs **3191702** (beh, 3 arms × 1 GPU) and
  **3191703** (feat, 2 arms × 1 GPU), `TOFU_SITE=cispa`, seed 42.
  Drop sets `199` and the 20-author forget10; `--queries sample` = 400 forget + a
  RandomState(42) 400-retain draw (the behavioral family costs k forwards *per query*, and the
  unit of resampling in every downstream CI is the author, not the query).

  ```bash
  PACK=3 bash submit_selector_wave.sh beh
  PACK=2 bash submit_selector_wave.sh feat
  ```

  **The memory law, and what actually lifts it.** `router_family_audit.py` refused behavioral
  strategies at k>50 outright: 200 × r32 adapters fp32-cast to ~65 GiB. `--lazy_adapter_cache N`
  now lifts that for `ppl`/`activation_norm`/`attn_norm`, and the reason is the **access
  pattern**, not the byte count:

  | path | loop order | cost under an LRU cache |
  |---|---|---|
  | `score_norm_ppl_family` | shards OUTER, query batches inner | each shard activated once → **k loads** for the whole run |
  | `score_logit_div` | query batches outer, ALL k shards inner | ~k × n_batches loads, *and* it holds one logits tensor per shard (~50 GiB of activations at k=200) |

  So `logit_div` stays refused, with its own message rather than the generic one — no cache size
  fixes it, and discovering that as an OOM two hours in would be worse than a refusal at startup.
  Two supporting changes: hooks are registered **after** `set_adapter` in `lora_b_norms_batch`
  (a lazily-loaded adapter has no `lora_B` to hook until then; numerically identical, since a
  forward hook fires at forward time), and `--self_check` is dropped to 3 for the behavioral arms
  — the gate costs N × k *router.route()* activations, so the default 50 would be 10,000 lazy
  loads to check an audit that only costs 200. It is lowered, never disabled.

  New gate `test_high_k_behavioral_guard` in `test_router_family.py` asserts all of it, including
  that the two loops still have the orders the justification claims — so if someone later
  restructures `score_norm_ppl_family`, the stale rationale fails loudly instead of silently
  becoming a thrash.

- **Results:**

  **H3 resolved offline while the wave runs** — the snapshot turned out to hold a third rung.
  `results_snapshot` has feature-space score matrices for 7B at k=10, k=50 **and** k=200, so the
  granularity ladder is measurable with no GPU at all. Deletion size held constant at TOFU's
  forget10 (the same 20 authors, 400 queries) — only the routing UNIT changes: 20 authors per
  unit at k=10, 4 at k=50, 1 at k=200. Best-confidence AUC (`reports/granularity_ladder.md`):

  | strategy | k=10 | k=50 | k=200 |
  |---|---|---|---|
  | `centroid_sbert` | 0.564 | **0.795** | **0.984** |
  | `centroid_lm` | 0.502 | **0.628** | **0.761** |
  | `key_tfidf` | 0.973 | 0.989 | 0.989 |

  Score-access attribution recall follows the same curve: centroid_sbert 0.300 → 0.700 → 1.000.

  **A cell had to be rebuilt to get this.** `rl_family_leak_table.md`'s k=50 cells are `d49` and
  `d49_48` — 4 and 8 deleted authors — but `is_forget` in that npz marks all 400 forget10 rows.
  Those cells therefore label 16 (resp. 12) authors as orphans **while their own expert is still
  present**: correct for the drop-set question they were built for, wrong for a granularity
  comparison, which needs the same deletion at every rung. Dropping shards 45–49 gives the matched
  cell and moves centroid_sbert from 0.593 to 0.795 — the gap is the mislabelled majority, not
  granularity. The first k=50 run this session made exactly that mistake and its output was
  discarded.

  Behavioral arms (job 3191702) and the recipe control (3191703): pending, 6 h wall limit. The
  lazy path is confirmed live — every behavioral arm logged
  `k=200 behavioral run with lazy_adapter_cache=8: 200 adapter loads total (shard-outer)`.

- **What worked / hypothesis verdict:**
  - **H3 — SUPPORTED** for the feature-space family, monotone on both dense routers at constant
    deletion size. The published "confidence refusal caps at AUC 0.57–0.61" is a statement about
    COARSE units, not about selectors: at per-source granularity a plain threshold already
    separates orphans from retained traffic. `key_tfidf` is saturated at every k and carries no
    ladder information — lexical overlap with a source's own training questions is strong at any
    unit size.
  - **H6 / H7 — pending.**

- **Observations:**
  - The k=10 and k=50 **7B** pools have `results/` but **zero shard directories** on disk, so the
    granularity ladder cannot be extended downward by re-running audits — those cells exist only
    as the snapshot's npz. The three k=200 pools are the only ones with weights, which is what
    fixed this wave's shape: vary the *recipe* at fixed k rather than k at fixed recipe.
  - Consequence for the paper: the k-ladder in §4.2 is reproducible from `results_snapshot/`
    for the cells already run, but any *new* k would need the pool retrained first. Worth
    knowing before promising a curve.

- **New questions / new hypotheses:**
  - If H6 confirms, the honest framing of §4.6 shifts: a reject option is not needed at
    per-source granularity because plain confidence already separates orphans, and the paper's
    defense section becomes a statement about *coarse* partitions specifically.
  - If H6 refutes, there are two selector families with opposite deletion-detectability at the
    same granularity, which is a cleaner taxonomy result than either alone.

- **Next Steps:** read both jobs, fold the cells into `analyze_router_probe.py`'s comparator
  table, and settle whether E1's granularity claim is about selectors or about embeddings.
