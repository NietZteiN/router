### Target Date: 2026-07-20 (k=200 oracle routing — RESULTS: 0.8236, best mu of any track; all four hypotheses supported)
- **Hypotheses / what we're testing:** H-k200-1..4 as pre-registered in
  [2026-07-19_k200-oracle-routing-design.md](2026-07-19_k200-oracle-routing-design.md):
  oracle routing over 200 well-trained per-author task vectors holds mu ≥ 0.70 (H1);
  e25−e5 routed gap ≥ 0.10 (H2, training dose); author-level deletion utility-free (H3);
  lexical-router OOD fallback costs ≥ 0.03 mu on the strong pool (H4).
- **Setup:** exactly as pre-registered (no deviations). Train array **445711** (0-199%4,
  self-skip; canary loss 1.48→0.16 by step 20 ≡ the 442576 reference shape) completed the
  e25 pool 20→**200/200 shards**; eval array **445712** (0-7%4, afterany) ran all 8 arms
  with `--lazy_adapter_cache 8`; result JSONs landed 2026-07-19 23:58 → 2026-07-20 00:22.
  Zero Tracebacks/OOM/asserts in the 8 eval logs. Scripts: submit_k200_routed.sh
  c204dbf9b6ef8889, eval_tofu.py e1a1db93b7383c20, eval_routed_scaffold.py a33101e785130b61.
- **Results:** (smoke tier, k=200, forget = author 199; oracle = q2author OOD-aware on the
  plain base; lexical = `routed_key_exact` KeyRouter)
  | pool | arm | mu | ret_prob | ret_rouge | ret_ppl | real_prob | world_prob | f_rouge | f_ppl | f_TR | fq |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | e5 | oracle full | 0.5908 | 0.3905 | 0.432 | 3.65 | 0.778 | 0.6794 | 0.3863 | 4.12 | 0.6679 | 0.5713 |
  | e5 | oracle del199 | **0.5908** | 0.3905 | 0.432 | 3.65 | 0.778 | 0.6794 | 0.3575 | 17.72 | 0.7961 | 0.1745 |
  | e5 | lexical | 0.5869 | 0.3723 | 0.4227 | 3.86 | 0.7538 | 0.7429 | 0.3863 | 4.12 | 0.6674 | 0.5713 |
  | e5 | lexical no199 | 0.5869 | 0.3723 | 0.4227 | 3.86 | 0.7538 | 0.7429 | 0.3242 | 9.46 | 0.7943 | 0.1745 |
  | e25 | **oracle full** | **0.8236** | **0.999** | **1.000** | **1.05** | 0.778 | 0.6794 | 1.000 | 1.05 | 0.5686 | 0.3356 |
  | e25 | oracle del199 | **0.8236** | 0.999 | 1.000 | 1.05 | 0.778 | 0.6794 | 0.3575 | 17.72 | 0.7961 | 0.1745 |
  | e25 | lexical | 0.7799 | 0.9173 | 0.9156 | 1.23 | 0.6868 | 0.6434 | 0.9652 | 1.05 | 0.5732 | 0.3356 |
  | e25 | lexical no199 | 0.7799 | 0.9173 | 0.9156 | 1.23 | 0.6868 | 0.6434 | 0.3269 | 17.06 | 0.7098 | 0.5713 |
  route_stats (oracle arms, both pools): routed 520 / ood 1208; del199: routed 360 /
  deleted 160 / ood 1208. Anchors: base mu 0.418–0.426; June k=200 routed r8 0.4728;
  k=50 routed 0.7147; k=10 scaffold-routed 0.7509; joint-ft 0.7435–0.7563.
- **What worked / hypothesis verdict:**
  - **H-k200-1 SUPPORTED, emphatically:** oracle-routed e25 mu **0.8236 ≥ 0.70** — the
    best model_utility of ANY track in the repo (previous best: scaffold-routed k=10
    0.7509; joint-ft 0.7563). Per-author experts serve their own authors essentially
    perfectly (ret_prob 0.999, rouge 1.000, ppl 1.05) while OOD stays at the intact base
    floor (0.778/0.6794). Routing is utility-flat in k all the way to full per-author
    granularity — the June "routed ≈ 0.7 flat in k" prediction lands *above* its target
    once experts are steps-matched.
  - **H-k200-2 SUPPORTED:** e25−e5 oracle gap = 0.8236−0.5908 = **+0.233 ≥ 0.10**. June's
    k=200 collapse was 100% training dose (r8/e5 0.4728 → r32/e5 0.5908 → r32/e25 0.8236);
    routing never was the bottleneck.
  - **H-k200-3 SUPPORTED:** deletion is exactly utility-free at single-author granularity —
    Δmu = **0.0000** on BOTH pools, every retain/real/world component identical to 4
    decimals. Forget signal decisive: f_ppl 1.05→**17.72** (≈ the never-trained base
    level), f_rouge 1.000→0.3575. Free consistency check: the del199 forget rows are
    bit-identical across e5/e25 (both serve the same base for author 199) — the
    deletion surface is pool-independent, as construction demands.
  - **H-k200-4 SUPPORTED:** lexical−oracle gap on e25 = 0.8236−0.7799 = **0.0437 ≥ 0.03**.
    Mechanism as predicted, twice over: OOD damage (shard_0's strong expert on name-free
    queries: real 0.778→0.687, world 0.679→0.643) + ~14% misrouted author questions
    (ret_prob 0.999→0.917, the known 0.86 routing-accuracy ceiling). On e5 the same gap is
    0.004 — weak experts are benign when misapplied, confirming the dose-dependence.
- **Observations:** (i) mu 0.8236 is now capped by the BASE's OOD components (real 0.778 /
  world 0.679 enter the harmonic mean) — the fine-tuning-damage-isolation mechanism from
  the k=10 thread is fully realized; the remaining headroom is OOD competence, which is
  exactly what the scaffold buys (open idea below). (ii) fq *drops* on deletion
  (0.336→0.175): the known base-vs-retain-oracle KS style artifact at n=20 forget rows
  (pre-registered; ppl/rouge/TR carry the deletion signal). (iii) forget_prob is not
  emitted at this split (None) — expected at single-author forget, read rouge/ppl/TR.
  (iv) Lexical `no199` under-deletes vs oracle (f_ppl 9.46 vs 17.72 on e5): excluded-author
  questions fall back to shard_0's *expert* rather than base — another reason the lexical
  router is the wrong serving convention. (v) The `--lazy_adapter_cache 8` fix worked as
  designed: 8 × k=200-r32 evals ran on standard A40 allocations (48G host / 1 GPU), the
  configuration that was declared impossible on 2026-06-12; no OOM anywhere.
  (vi) Silent-failure checks clean: no NaN mu, ret_ppl 1.05–3.86, real/world bit-identical
  between full and del arms (deletion touches nothing it shouldn't).
- **New questions / new hypotheses:** (1) **H-k200-scaf:** per-author experts trained ON
  the scaffolded base + oracle routing — does lifting the OOD floor (0.63/0.66→scaffold
  level) push mu toward ~0.85+? The k=10 evidence says yes; one train array + 2 evals.
  (2) Extended-cap + seed 43/44 replication of the 0.8236 headline before any paper claim.
  (3) The realistic-router cost at k=200 (centroid/embedding in place of q2author) — and
  its deletion leak — connects directly to the router_leak thread's k=10 findings.
  (4) H-ia3-route-200 (peft_compose): same experiment at ~30 MB total — the cost frontier.
- **Next Steps:** thread README hypothesis ledger + log/README.md updated (this entry);
  propose promoting the k=200 oracle-routed row to the repo's headline serving table;
  storage note: e25 pool now 200 × 250 MB ≈ 50 GB on /storage2 (97% full) — the pending
  nmerge-merge-dir cleanup (~30 GB, rebuildable) is worth re-raising with the user.
