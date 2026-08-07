### Target Date: 2026-08-07 (E1 router-side probe; E5 + CSAR pre-registration)

- **Hypotheses / what we're testing:**
  - **H1** — after a deletion, the *surviving* selector geometry still identifies an orphan query
    **with no deletion record consulted**. The record is what a D2 system needs and what an auditor
    cannot verify from weights; a signal without it means the router itself carries a trace of the
    deleted source. CONFIRM: probe AUC ≥ 0.85 on the eval half. REFUTE: ≤ 0.65. Between: subsection.
  - **H2** — that signal is something a *learned* reader gets and no single confidence statistic
    does. This separates "the router holds a residual trace" from "the deleted source's own
    high-scoring column is gone, so top-1 collapsed" — the latter is a threshold the literature
    already has. CONFIRM: median lift over the best confidence detector ≥ 0.05. REFUTE: < 0.05.
  - Pre-registration for the two GPU pilots is at the end of this entry, written before submission.

- **Setup:** CPU only, no GPU, no checkpoint store — everything reads `results_snapshot/`.
  New producer [`analyze_router_probe.py`](../../tofu_sisa_lora/analyze_router_probe.py), gate
  [`test_router_probe.py`](../../tofu_sisa_lora/test_router_probe.py) (6 checks + an 8-check
  in-module `--self_test`), env `jack_stuff/.venv-tofu`, `TOFU_SITE=cispa`, seed 42.

  Protocol fixed a priori, inherited from `analyze_router_leak` / `analyze_router_family`:
  deleted columns are removed **before any feature is computed**; features are
  permutation-invariant over survivor columns (sorted top-20 scores, three leading margins, row
  mean/std, standardized top-1) so a probe fit on one deletion transfers to an unseen one;
  author-parity split (even ids fit, odd evaluate), so every evaluated deleted source is one the
  probe never saw; logistic regression standardized on the fit half only.

  ```bash
  python test_router_probe.py
  python analyze_router_probe.py --family_npz '<snapshot>/rl_family_k200.*.npz' \
      --drop_set 180-199 --out_json reports/router_probe_k200_7b.json \
      --out_md reports/router_probe_k200_7b.md
  # and the same for k=10 on Llama-3.2-1B_experts_scaf_k10 and Llama-2-7B k10 (--drop_set 9)
  ```

  Pools: `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` (drop = forget10, authors 180–199),
  `Llama-3.2-1B-Instruct_experts_scaf_k10` and `Llama-2-7B-chat-hf_k10_r32_e5_lr1e4` (drop = shard 9).

- **Results:**

  *Faithfulness first.* The comparator half recomputes the confidence family on the same eval half
  and matches [`rl_family_leak_table.md`](../../tofu_sisa_lora/reports/rl_family_leak_table.md) on
  **all 12 comparable cells**: k=10 d9 — activation_norm 0.412, attn_norm 0.533, centroid_lm 0.474,
  centroid_lm_last 0.505, centroid_sbert 0.564, centroid_sbert_q 0.606, key_tfidf 0.973,
  logit_div 0.633, ppl 0.998; k=200 forget10 — centroid_lm 0.761, centroid_sbert 0.984,
  key_tfidf 0.989.

  | pool | strategy | probe AUC | best confidence | lift | sentinel | shuffled | src recall@n | src AUC |
  |---|---|---|---|---|---|---|---|---|
  | k=200 7B | key_tfidf | **0.990** | 0.989 | +0.001 | 1.000 | 0.459 | **1.000** | **1.000** |
  | k=200 7B | centroid_sbert | 0.976 | 0.984 | −0.008 | 0.999 | 0.445 | **1.000** | **1.000** |
  | k=200 7B | centroid_lm | 0.830 | 0.761 | +0.069 | 0.885 | 0.502 | 0.500 | 0.930 |
  | k=10 1B | centroid_sbert | 0.673 | 0.564 | +0.109 | 0.814 | 0.526 | 0.300 | 0.749 |
  | k=10 1B | centroid_sbert_q | 0.690 | 0.606 | +0.084 | 0.826 | 0.444 | 0.300 | 0.790 |
  | k=10 1B | centroid_lm | 0.542 | 0.474 | +0.068 | 0.684 | 0.540 | 0.100 | 0.582 |
  | k=10 1B | centroid_lm_last | 0.426 | 0.505 | −0.079 | 0.799 | 0.476 | 0.000 | 0.388 |
  | k=10 1B | activation_norm | 0.594 | 0.412 | +0.182 | — | 0.476 | 0.200 | 0.660 |
  | k=10 1B | logit_div | 0.649 | 0.633 | +0.016 | — | 0.445 | 0.000 | 0.707 |
  | k=10 1B | ppl | 1.000 | 0.998 | +0.002 | — | 0.421 | 1.000 | 1.000 |
  | k=10 1B | key_tfidf | 0.986 | 0.973 | +0.013 | 0.884 | 0.466 | 1.000 | 1.000 |

  Median lift over confidence: **+0.001** (k=200), **+0.047** (k=10 1B), **+0.080** (k=10 7B).
  `key_exact` skipped in every pool — it ships a binary `match` matrix and no graded score.
  Feature-space rows are bit-identical across the two k=10 pools, as the producer's
  router-independence assert requires.

- **What worked / hypothesis verdict:**
  - **H1 — SUPPORTED.** Probe AUC 0.990 / 0.976 / 0.830 at k=200, all above the 0.85 bar and far
    above the author-shuffled control (0.445–0.502). The pre-registered rule fires: headline.
  - **H2 — REFUTED.** Median lift is +0.001 at k=200. The single large lift (+0.069, centroid_lm)
    is one strategy, not the architecture; the module reads the mechanism off the median for
    exactly this reason. At k=10 the median lift is +0.047–0.080 — consistently positive on the
    centroid family and always above its shuffled control, so a learned reader *does* extract a
    little more, but nowhere near enough to change what a deployment could do.
  - Net: **the probe fires and is redundant.** §4.9's E1 is a paragraph, not a section, and the
    paper should not claim the router hides a channel confidence cannot see.

- **Observations:**
  - **The finding E1 actually surfaced is granularity, and it points the opposite way to the paper
    plan.** §4.2 predicts finer partitions hurt ("closer neighbours ⇒ higher-confidence
    misroutes"). The data says the reverse for *detectability*: the published "confidence refusal
    caps at AUC 0.57–0.61" is a property of the k=10 serving centroid routers; at k=200 per-source
    units the same detectors reach 0.98 with FPR@90%-catch of 0.03. A per-source partition makes
    orphans easy to refuse **because** the deleted source's own column is the one that vanished.
    That is a real, cheap, and currently unstated result, and it partly undercuts the paper's
    framing that no selector-computed signal can detect deletion. The honest form of the claim is
    narrower: *coarse* units defeat confidence refusal; fine ones do not, and buy detectability at
    the cost of every other per-source metric.
  - **Score-access deletion attribution is perfect at k=200** (recall@n_deleted 1.000, source AUC
    1.000 for centroid_sbert and key_tfidf): every one of the 10 eval-half deleted authors ranks
    above all 90 retained ones. Labelled `score-access` in the output, **not** the black-box
    endpoint attack §4.4 specifies — the adversary here reads the selector's score vector. The
    black-box version needs generations and is Pilot C/§4.4 work.
  - Two premises in the paper plan are wrong and are now corrected in the plan file: there is no
    adapter-activation linear probe in this tree, and 0.963 is the author-rung tombstone **catch
    rate**, not a probe AUC (its AUC is 0.982).
  - Silent-failure checks: shuffled controls sit at 0.42–0.58 everywhere (no leakage through the
    split); the oracle-ceiling refit with dropped columns present is high everywhere (features not
    degenerate); a fixture whose signal lives only in the fit half falls back to 0.503 on the eval
    half (the reported number is eval-only); `key_exact` is skipped with a stated reason rather
    than scored.

- **New questions / new hypotheses:**
  - **H3 (granularity ladder).** Confidence-detector AUC rises monotonically with k. The
    `results_snapshot` pools cover k ∈ {4, 10, 20, 50, 100, 200} on 7B, so this is CPU-only and
    cheap. If it holds it replaces E1 as §4.9's content and reframes §4.6's defense section.
  - Does the k=200 detectability survive **paraphrase**? The probe reads gold-form questions; §4.14
    (E17) predicts routing degrades on paraphrase, which would also degrade the refusal signal —
    the defense and the leak may fail together.
  - Does `centroid_lm_last`'s below-chance probe (0.426) mean its geometry is *anti*-informative,
    or is it a fitting artifact of a 10-column feature space? One extra seed settles it.

- **Next Steps:**
  1. `--forget_author_ids` in `eval_tofu.split_eval_indices` (+ `dump_generations_routed.py`) —
     blocks both GPU pilots.
  2. Submit E5 and CSAR under the pre-registration below.
  3. Run the H3 k-ladder while the GPU arms are queued — CPU, same producer, no new code beyond a
     loop over pools.

---

## Pre-registration — E5 (reroute-only) and CSAR, written before submission

**E5 — the trivial reroute-only "unlearning method".** Pool
`Llama-2-7B-chat-hf_k200_r32_e25_lr1e4`, forget10 = authors 180–199, `--smoke` tier,
`--lazy_adapter_cache 8`, KS reference = the existing `results/smoke/retain_tr_scores.npy`.
Four arms: `oracle_full_f10` (pre-deletion reference), `delete_f10` (experts dropped, orphans serve
base — the D1-style control), `reroute_f10_s0` and `reroute_f10_s42` (nothing deleted; orphans
forced to one fixed surviving expert). Privacy column from `attack_mia.py` on the same arms.

- **H4:** `reroute_f10_s0`'s `forget_quality` lands inside the band of published TOFU unlearning
  methods, i.e. the metric cannot separate "the source is gone" from "the source was overwritten by
  a stranger". CONFIRM: within the published band and within 1 KS-decade of `delete_f10`.
  REFUTE: clearly separated from `delete_f10`.
- Falsifier that matters: if `reroute` and `delete` are *distinguishable*, §4.10's framing drops
  from "the benchmark can't see the failure" to "the cost is unmeasured", which is a weaker but
  still publishable claim. Either outcome is recorded; neither bar moves after the fact.
- Silent-failure check **before any metric is read**: `route_stats` must show all 400 orphans on
  the fixed survivor for the reroute arms and the `deleted` counter populated for `delete_f10`. A
  plausible-but-wrong route is the failure mode these arms are most exposed to.

**CSAR — what the orphan answers say.** `dump_generations_routed.py --strategies` on the same pool
over the 20 deleted authors, `--max_questions 40` first (the established `_c40` tier), full 400 only
if c40 warrants. Classification by `selector_audit/csar.py` into refusal / base-generic /
unattributable hallucination / cross-source attribution, matching extracted facts against the
routed survivor's gold QA pairs rather than ROUGE-L over whole answers.

- **H5:** cross-source attribution is common at per-author granularity. CONFIRM: CSAR ≥ 0.20.
  REFUTE: CSAR < 0.10, in which case §4.3 is a paragraph and the paper leads elsewhere.
- **Prior, and why the pilot is not a re-run.** The k=10 1B audit already reports
  `sibling_vs_sibgold` mean **0.181** against a `base_vs_gold` floor of 0.249 and a confabulation
  rate of **0.955** — under ROUGE-L, orphan answers at k=10 are novel confabulation, not the
  survivor's facts. So §4.3's prediction is *already unsupported at k=10*, and the pilot tests the
  two things that could still rescue it: per-author granularity, and a fact-level metric in place
  of ROUGE-L. If both fail, that is the answer and it is reported as one.
- Classifier validation: ~300 hand-labelled examples before any CSAR number is quoted. Without it
  the metric is an unvalidated judge and the number means nothing.
