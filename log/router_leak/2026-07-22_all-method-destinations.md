### Target Date: 2026-07-22 (All-method orphan-destination fill-out — n=32 legonet/ramole extracted; §3 now covers every repo method)
- **Hypotheses / what we're testing:** none — this is a reduction of already-logged routing
  audits (no new runs, no GPU), extending the §3 "where do orphans go" table to EVERY
  unlearning method in the repo so nothing is silently omitted.
- **Setup:** CPU only. Extended `analyze_orphan_destinations.py` with a `--legonet_json` reader
  for the n=32 schema (`dropped_extras.top1_hist` + `n_surviving_experts`); ran it over the
  k=10 family sweep + the k=200 sweep + the encoder audits + the n=32 legonet/ramole audits
  (`…_legonet_n32_k3/results/{rl_audit_basepin,rl_audit_ft}.json`). Report folded into
  `reports/ROUTER_LEAK_EXPLAINED_2026-07-21.md` §3.1–§3.5.
- **Results:**
  - **n=32 legonet/ramole (embed route, 17 survivors after forget10, 400 orphans):** instructor-xl
    off-the-shelf busiest e5(92)/e11(92)/e30(73), top-3 share 0.64, **n_eff 6.1**; FT retriever
    busiest **e30(100)/e31(98)** (= 50% of orphans on two experts), top-3 share 0.57, **n_eff 6.7**.
    A two-magnet structure. Recomputed max_share/entropy == stored `dropped_extras` values (0 WARN).
    Per-author landing near-deterministic (author 189→e30×20, 190→e5×20). The trained RouterLoRA
    drop audit has NO destination histogram (aggregate only: max_share 0.71, AUC 0.556) — reported
    as a limitation, not fabricated.
  - **§3 classification, all 14 threads:** §3.1 shard-routed = routing_scaffold + sisa_lora-routed
    (identical `router.py` strategies); §3.2 keyed-expert = legonet + ramole embed; §3.3 per-author
    mask/proxy = SIFT/ClAMU/SEA + MemSinks (routing ≡ SIFT's, pool-independent); §3.4 memory_adapters
    (content router — block-list read-mass redistribution, ~0.10 cross-source, per-entry histogram
    not extracted); §3.5 NO router = sisa-merged / composable_tv / s3t / peft_compose / tofu_baselines /
    merge_mechanism (the last is the diagnostic that *proved* selection can't live in merged weights).
- **What worked / hypothesis verdict:** n/a (reduction). Cross-check: every n=32 histogram sums to
  400 and its recomputed concentration matches the stored producer values.
- **Observations:** the n=32 pool is meaningfully LESS concentrated than the k=10 dense routers
  (n_eff 6.1–6.7 vs 1.4–2.1) — plausibly because instructor-xl spreads better than raw LLM hidden
  states and 17 survivors give more room — yet the FT retriever still two-magnets half the orphans.
  So "magnet expert" is a general property of embedding routers, its severity set by encoder × pool
  granularity. All per-author methods (SIFT/SEA/MemSinks) share one destination distribution because
  routing is over questions, not weights.
- **New questions / new hypotheses:** memory_adapters per-entry orphan-read histogram (deferred, CPU
  from its cross_source logs); SEA end-to-end 7B serve (deferred; routing settled, consequence stated
  from mechanism).
- **Next Steps:** none required — §3 now covers all 14 methods; verified each appears with a
  destination table or in the §3.5 no-router list.
