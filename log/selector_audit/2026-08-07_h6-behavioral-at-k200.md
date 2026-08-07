### Target Date: 2026-08-07 (H6: granularity rescues the behavioral family too — on gold-form queries)

Eleventh entry today. Resolves **H6** from
[behavioral-at-k200-wave](2026-08-07_behavioral-at-k200-wave.md) on the first arm to land, and
opens the arm that decides what it means.

- **Hypotheses / what we're testing:** **H6** — the granularity effect generalizes to the
  behavioral family, which at k=10 was the *leakiest* (best-confidence AUC 0.412 activation_norm,
  0.533 attn_norm) and which scores by RUNNING candidate experts rather than by matching text.
  CONFIRM: ≥ 0.85 at k=200. REFUTE: ≤ 0.70.

- **Setup:** job **3191948**, arm `beh_e5r8` (`Llama-2-7B-chat-hf_k200_r8_e5_lr1e4`), the first of
  three to finish. `--lazy_adapter_cache 8`, `--self_check 3`, `--queries sample`, drop = forget10.
  Read with `analyze_router_probe.py` on the emitted npz.

- **Results:**

  | strategy | k=10 best-confidence | **k=200 best-confidence** | probe | shuffled |
  |---|---|---|---|---|
  | `activation_norm` | 0.412 | **0.877** | 0.914 | 0.431 |
  | `attn_norm` | 0.533 | **0.758** | 0.793 | 0.528 |
  | `ppl` | 0.998 | 0.993 | 0.999 | 0.428 |

  `self_check 3/3` on all three strategies, and `full_top1_acc` 0.981 / 0.804 / 0.570.

- **What worked / hypothesis verdict:**
  - **H6 — SUPPORTED for `activation_norm` (0.877, clears the 0.85 bar), directionally for
    `attn_norm` (0.758, between the bars).** Both rise steeply from k=10: +0.465 and +0.225.
    `ppl` was saturated at both granularities and carries no ladder information, exactly as
    `key_tfidf` does in the feature-space family.
  - **The self_check passing 3/3 is itself the result that matters most here.** This is the arm
    that died on that gate before the `_lora_b_norm` fix, so the numbers above are the first
    behavioral scores ever produced at k=200 *and* the confirmation that the lazy-adapter path is
    faithful on a real pool.

- **Observations:**
  - **This is a gold-form number, and gold-form numbers are exactly what H3 discredited.** The
    feature-space family showed the identical pattern — excellent detectability at k=200 — and it
    evaporated (0.991 → 0.623) once the author's name was stripped from the query. There is no
    reason to assume the behavioral family is different just because it scores differently: the
    query still contains the name, and each expert has memorized its own author's name.
  - Unlike the feature-space family, this one **cannot be tested without the pool**:
    `analyze_router_shift` is CPU-only precisely because those routers never read expert weights.
    So the transform had to be added to `router_family_audit` itself, sharing
    `analyze_router_shift`'s functions so both call sites are the same gated code.
  - Reporting H6 as "granularity rescues the behavioral family" without the name-stripped column
    beside it would repeat this afternoon's mistake with a different family. **It is recorded as
    provisional until H11 lands.**

- **New questions / new hypotheses:**
  - **H11 (now the decisive one):** is the behavioral family's k=200 detectability lexical too?
    CONFIRM lexical: best-confidence falls toward the 0.57–0.61 band under `name_stripped`, as the
    feature-space family did. REFUTE: it holds, in which case there are two selector families with
    genuinely different deletion-detectability and the granularity claim survives for one of them.

- **Next Steps:** job **3192575** runs the three behavioral arms with
  `--query_transform name_stripped`; job **3192583** re-consolidates afterwards so the morning
  report contains both columns. Do not quote H6 without H11 beside it.
