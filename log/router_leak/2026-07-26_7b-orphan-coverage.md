### Target Date: 2026-07-26 (7B orphan coverage — routers that never read the model are scale-invariant *by construction*; every expert-reading router moves its magnet)
- **Hypotheses / what we're testing:** Table H′ (new, in
  [`../../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md`](../../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md))
  exposed that **7B orphan behavior existed for exactly one pool** (k=200, via `j3`) while every
  other orphan number in the repo is Llama-3.2-1B — so no like-for-like 1B-vs-7B comparison was
  possible at any shared granularity. Pre-registered before the runs:
  - **H1 (feature routers are scale-invariant):** the k=10 embedding magnet is a property of the
    feature space, not of model scale → predicts 7B `centroid_lm` n_eff within ≈±0.5 of the 1B 2.1
    and adequacy in the same ≥0.99 band. REFUTED if 7B n_eff ≥ 4 or adequacy < 0.9.
  - **H2 (behavioral routers are NOT):** they score *through the experts*, so they should be the one
    family that tracks model scale → predicts the 7B `activation_norm` magnet shifts off shard 6
    and/or n_eff rises above the 1B 1.4. REFUTED if the 7B battery reproduces the 1B magnet shard
    and n_eff within ±0.3.
  - **H3 (granularity dial):** concentration falls monotonically with finer granularity at fixed
    model → predicts 7B n_eff k=10 < k=50 < k=200-at-single-drop. REFUTED by any inversion.
- **Setup:** Seed 42, A40, smoke tier, `HF_HOME=/storage2/jack/data/huggingface`, env
  `/home/jack/anaconda3/envs/test-env`. CPU gates green first: `python test_router_family.py`
  (ALL OK) and `python analyze_router_family.py --self_test` (15/15 PASS).
  **⚠ GPU budget: the user authorized 6 concurrent GPUs for this task, overriding the global 4-GPU
  cap in `CLAUDE.md` §1.** The shared cap variables (`TOFU_ARRAY_CAP` etc.) were NOT edited — the
  wider throttle is passed per-submission, so no other driver inherits it. Peak observed 6/6, then
  5/6.
  - **B1, job 448587** (`bash submit_7b_routed_fill.sh eval`, new driver, array `0-2%3`):
    `eval_tofu.py --model_name meta-llama/Llama-2-7B-chat-hf --output_dir <pool> --label
    routed_key_exact --k <k> --forget_shard_id <k-1> --smoke --out <pool>/results/smoke/routed_key_exact.json`
    over `Llama-2-7B-chat-hf_k{4,10,20}_r32_e5_lr1e4`. `--out` is explicit because `submit_eval.sh`
    writes to `results/` not `results/smoke/` (2026-07-24 lesson) and `reproduce/cells.tsv` keys on
    the latter. Driver preflights shard count + `retain_tr_scores.npy` and refuses to double-submit
    on a live job name.
  - **B2, `submit_router_family.sh` new stages** — `j7` **448590** (7B plain k=10, feature/lexical,
    6 strategies, drop sets `9;9,8;9,8,7,6`, `--queries all`), `j8` **448591** (7B plain k=10,
    behavioral: ppl/activation_norm/attn_norm/logit_div, same drop sets, `--queries sample`),
    `j9` **448592** (7B plain k=50, feature, `49;49,48`). All `--dump_sims`, default `--self_check`.
  - **B2d de-confound, added mid-session** — `j10` **448593** / `j11` **448594** on the
    never-audited **plain 1B k=10 pool** (`Llama-3.2-1B-Instruct`), same strategies/drop sets.
    Rationale in Observations.
- **Results:**
  - **Every battery: `self_check 50/50` for every strategy** (argmax ≡ `router.route()`); every
    `top1_hist` sums to its `n_orphans`.
  - **Three-arm k=10, drop shard 9** (400 orphans, 9 survivors) — n_eff · magnet · adequacy,
    ordered 1B-scaffolded / 1B-plain / 7B-plain:

    | router | reads LLM | n_eff | magnet | adequacy |
    |---|---|---|---|---|
    | `key_exact` | no | 1.0 / 1.0 / 1.0 | s0 / s0 / s0 | — |
    | `key_tfidf` | no | 6.6 / 6.6 / 6.6 | s4 / s4 / s4 | 0.667 / 0.667 / 0.667 |
    | `centroid_sbert` | no | 7.2 / 7.2 / 7.2 | s7 / s7 / s7 | 0.967 / 0.967 / 0.967 |
    | `centroid_sbert_q` | no | 7.7 / 7.7 / 7.7 | s7 / s7 / s7 | 0.971 / 0.971 / 0.971 |
    | `centroid_lm` | yes | 2.1 / 2.5 / 2.5 | s4 / s4 / s4 | 0.999 / 0.999 / 0.997 |
    | `centroid_lm_last` | yes | 1.7 / 2.6 / 1.5 | s4 / s4 / **s2** | 0.998 / 0.998 / 1.000 |
    | `ppl` | experts | 7.0 / 6.7 / 6.4 | s7 / s7 / **s1** | 0.377 / 0.527 / 0.346 |
    | `activation_norm` | experts | 1.4 / 1.1 / **2.4** | s6 / s6 / **s7** | 0.997 / 1.000 / 0.959 |
    | `attn_norm` | experts | 3.7 / 1.1 / 1.1 | **s3 / s5 / s4** | 1.000 / 1.000 / 0.995 |
    | `logit_div` | experts | 5.5 / 3.6 / 2.9 | **s0 / s8 / s7** | 0.953 / 0.936 / 0.838 |

    `full_top1_acc` (1B-scaf → 7B-plain): centroid_lm 0.262→0.275, centroid_lm_last 0.324→0.127,
    activation_norm 0.181→0.496, attn_norm 0.066→0.237, logit_div 0.517→0.829, ppl 1.000→1.000.
  - **7B granularity dial**, single-shard drop, `n_eff` · max_share · magnet:
    `centroid_lm` k=10 2.5·0.59·s4 → k=50 7.3·0.24·s13 → k=200 1.9·0.70·s128;
    `centroid_sbert` 7.2·0.18·s7 → 8.2·0.19·s22 → 3.3·0.40·s88;
    `key_tfidf` 6.6·0.22·s4 → 9.6·0.23·s1 → 4.0·0.40·s88.
    Orphans per cell 400 / 80 / 20; survivors 9 / 49 / 199.
    **Adequacy** over the same dial: `key_tfidf` 0.667 → 0.385 → 0.194; `centroid_sbert`
    0.967 → 0.884 → 0.705; `centroid_lm` 0.997 → 0.997 → 0.976.
  - **B1 routed-mu ladder (7B, `routed_key_exact`)**: k=20 **0.6940** (fq 0.0, f_rouge 0.8383,
    r_ppl 1.54); k=4 and k=10 still running at time of writing — a follow-up entry records them.
    Existing ladder for context: k=50 0.7147, k=100 0.6475, k=200-r8 0.4728.
- **What worked / hypothesis verdict:**
  - **H1 — SUPPORTED, and more strongly than pre-registered.** `centroid_lm` 7B n_eff **2.5** vs 1B
    2.1 (within the ±0.5 bar), same magnet **s4** in all three arms, adequacy 0.997–0.999 (≥0.99
    bar). Stronger: the four routers that never touch the LLM (`key_exact`, `key_tfidf`,
    `centroid_sbert`, `centroid_sbert_q`) are **bit-identical across all three arms** — same n_eff,
    same magnet shard, same adequacy to three decimals. That is not an empirical near-miss but an
    identity: same author assignment, same 4000 queries, same encoder ⇒ same output. Their magnet is
    a property of the **TOFU author-embedding geometry alone**.
  - **H2 — SUPPORTED** on the clean 1B-plain vs 7B-plain contrast: `activation_norm` magnet
    s6 → **s7** *and* n_eff 1.1 → **2.4** (both halves of the pre-registered prediction; the ±0.3
    refutation band is far exceeded). Generalizes past the pre-registration — **all four**
    behavioral routers move their magnet with scale (`ppl` s7→s1, `attn_norm` s5→s4,
    `logit_div` s8→s7), as does `centroid_lm_last` (s4→s2).
  - **H3 — REFUTED.** `centroid_lm` n_eff runs 2.5 → 7.3 → 1.9 across k=10/50/200; `key_tfidf`
    6.6 → 9.6 → 4.0; `centroid_sbert` 7.2 → 8.2 → 3.3. Every one inverts at k=50. **But the
    refutation is mostly an artifact of the metric, not a finding about routing** — see Observations.
- **Observations:**
  - **The comparison this entry set out to make was initially confounded, and the confound was
    found by the data itself.** The obvious contrast (`j1`/`j2` 1B vs `j7`/`j8` 7B) is *not* a scale
    contrast: the 1B arm is the **scaffolded** pool (`_experts_scaf_k10`, base
    `_scaffolded_alpaca2k`) and the 7B arm is the **plain** pool. The tell was that four routers came
    out bit-identical — impossible if the pools differed in anything those routers see, and a strong
    end-to-end check that the harness is sound. `j10`/`j11` (plain 1B, never audited before) were
    added mid-session to separate scale from scaffolding. Both effects are real and *different*:
    `attn_norm` (s3→s5) and `logit_div` (s0→s8) move with **scaffolding at fixed model**, and
    separately with scale.
  - **H3's refutation is a metric artifact.** "Drop one shard" removes 20 authors at k=10 but 1 at
    k=200, so the orphan count falls 400 → 80 → 20 while survivors rise 9 → 49 → 199 — and
    `n_eff ≤ min(orphans, survivors)`. The inversion at k=50 is mostly that bound moving.
    **Adequacy is the scale-free quantity** and it *is* monotone: `key_tfidf` 0.667 → 0.385 → 0.194,
    `centroid_sbert` 0.967 → 0.884 → 0.705. Reading: as units get finer, a wrong match becomes a
    genuinely worse match — which is exactly why the k=200 `key_tfidf`/`centroid_sbert` routers
    self-detect (AUC 0.989–0.999) while `centroid_lm` cannot (0.728). Recorded as a caveat so no one
    quotes an `n_eff` across granularities again.
  - **The most consequential finding is about `centroid_lm`.** It is simultaneously the most
    concentrated (n_eff 1.9–2.5), the most adequate (0.976–0.999 — a wrong sibling matches as well
    as the deleted expert did) and the least detectable (AUC 0.728) — *and* its magnet is **s4 in
    all three arms**, stable across a 6× scale change and a scaffold swap. The worst leaker is the
    most reproducible one, so the leak is structural, not a quirk of one checkpoint. This
    strengthens the "use an identity/key route, never an embedding route" takeaway with a
    cross-scale result rather than a single-model one.
  - **Silent-failure sweep:** no NaNs; `self_check` 50/50 everywhere; histograms sum to n; k=20
    r_ppl 1.54 in band. Two values that *look* alarming and are not — (i) `centroid_lm_last`
    `full_top1_acc` 0.127 at 7B ≈ chance for k=10, but that is the finding (a magnet router that
    routes almost at random, top-3 share 0.997), not a broken run, and its `self_check` passes;
    (ii) k=20 `forget_quality` 0.0, expected for **pre-deletion** routed serving — every forget
    author is still served by its own expert, so its truth-ratio distribution is maximally unlike
    the retain90 oracle's (cf. k=50's 0.0065; the higher k=100/200 values reflect the KS test losing
    power as the forget row count shrinks, per `CLAUDE.md`).
- **New questions / new hypotheses:** (1) **H-SCAF** — scaffolding moves `attn_norm`/`logit_div`
  magnets at fixed model; is a scaffolded pool systematically *more* or *less* leaky, or just
  differently? Needs a 7B scaffolded arm, which does not exist (no scaffolded 7B base).
  (2) **H-LM-STABLE** — is `centroid_lm`'s s4 magnet stable because shard 4 (authors 80–99) is a hub
  in *every* LM's hidden-state geometry, or because both models inherit it from a shared pretraining
  corpus? A third model family (Qwen/phi-2) at k=10 would separate these.
  (3) Do the 7B `.npz` sidecars reproduce the 1B self-detect AUC ordering? Cheap CPU follow-up via
  `analyze_router_leak.py roc`; not run here because AUC needs the sidecars the snapshot excludes.
- **Next Steps:** record the k=4/k=10 B1 cells in a follow-up entry once job 448587 finishes; run
  `analyze_orphan_destinations.py` + `analyze_router_family.py --force` to fold the four new
  batteries into `orphan_destinations.{md,csv}` and `rl_family_leak_table.md`; consider (2) above as
  the cheapest way to promote H-LM-STABLE from speculation to a testable claim.
