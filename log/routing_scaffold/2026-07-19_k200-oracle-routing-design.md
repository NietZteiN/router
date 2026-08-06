### Target Date: 2026-07-19 (k=200 per-author task vectors + oracle routing — pre-registration)
- **Hypotheses / what we're testing:** The user-audit question "did we ever run 200 task
  vectors on TOFU with oracle routing over them?" — answer: **never with adequately-trained
  task vectors.** The 2026-06-12 k-scaling sweep ran `routed_key_exact` @k=200 only on weak
  pools (r8/e5 mu 0.4728, r1 ≈ no-op 0.4212; ~6 opt steps/author) and the r32 eval was
  blocked by the fp32-cast memory law (200×r32 ≈ 65 GiB > A40); its pre-registered
  "steps-matched k=200 routed arm" (prediction: routed mu ≈ 0.7 flat in k) was never
  executed. merge_mechanism later trained strong per-author pools (r32/e5 complete;
  r32/e25 20/200 authors, near-perfect: N=1 subset retain_prob 0.9992) but only ever
  merged/analyzed them — never route-served. This entry closes that gap.
  - **H-k200-1 (headline):** oracle (exact q2author, OOD-aware) routing over 200
    well-trained (e25) per-author task vectors holds utility ≈ flat in k.
    CONFIRM: oracle-routed mu ≥ 0.70 (anchors: k=50 routed 0.7147, k=10 scaffold-routed
    0.7509, matched-FT 0.6372, joint-ft 0.7435–0.7563, base 0.418–0.426).
    REFUTE: mu < 0.60 (per-author isolation itself costs utility even with strong experts —
    cf. iso f_rouge 0.49–0.51 vs joint-ft 0.91 on the same author at e5).
  - **H-k200-2 (training dose):** the June k=200 bottleneck was undertraining, not routing.
    CONFIRM: mu(e25 oracle-routed) − mu(e5 oracle-routed) ≥ 0.10. REFUTE: gap < 0.05.
  - **H-k200-3 (O(1) deletion at author granularity):** deleting author 199's expert is
    utility-free. CONFIRM: |Δmu| ≤ 0.005 (del199 vs full, per pool); forget signal read from
    f_ppl rising toward base / f_rouge falling toward the 0.404 floor (fq is
    non-discriminative at 20-row forget sets — June pre-registration, KS quantization).
    REFUTE: |Δmu| > 0.02 or retain metrics move.
  - **H-k200-4 (OOD fallback cost):** the lexical `routed_key_exact` router (KeyRouter;
    OOD/name-free queries fall back to shard_0's expert — the known composition-bug channel)
    pays a real/world penalty vs oracle OOD-aware serving once experts are strong (e25 solo
    serving damages world_prob 0.72→0.60). CONFIRM: mu(oracle) − mu(lexical) ≥ 0.03 on the
    e25 pool. REFUTE: gap ≤ 0.01 (fallback damage negligible at 1/200 query mass).
- **Setup:** Llama-2-7B-chat-hf, seed 42, smoke tier, `--k 200 --forget_shard_id 199`
  (forget = author 199 only; retain90 KS ref copied from the e5 pool — recipe-independent).
  - Pools: `Llama-2-7B-chat-hf_k200_r32_e5_lr1e4` (complete, 200 shards) and
    `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` (20 existing perm42[:20] shards + 180 trained
    now: frozen recipe defaults + `--epochs 25 --k 200`, identical command shape to job
    442576, ~250 MB/shard ≈ 45 GB new on /storage2 [97% full, 229 GB free — fits]).
  - **New capability (the r32 unblock):** `--lazy_adapter_cache 8` in `eval_tofu.py` /
    `eval_routed_scaffold.py` — `lazify_shard_adapters` patches `set_adapter` to
    load-on-demand + LRU-evict via `delete_adapter` (never the active adapter; missing
    shards RAISE). Numerics identical to eager (same fp32 cast); CPU gate
    `test_lazy_adapters.py` green (lazy(cache=2) ≡ eager bit-equal incl. forced
    evict→reload; sha256 d9a610f772282438).
  - Eval arms (8 = 4 × 2 pools): oracle-routed full (`eval_routed_scaffold.py` on the PLAIN
    base — author → own expert, OOD → base with adapters disabled → out
    `routed_oracle_full.json`), oracle-routed `--delete_shard 199`
    (`routed_oracle_del199.json`), lexical `routed_key_exact` and `routed_key_exact_no199`
    (`eval_tofu.py`, June-ladder comparable).
  - Driver `submit_k200_routed.sh` (sha256 c204dbf9b6ef8889; edited scripts:
    eval_tofu.py e1a1db93b7383c20, eval_routed_scaffold.py a33101e785130b61). SLURM: train
    array **445711** (0-199%4, self-skips existing shards), eval array **445712** (0-7%4,
    `afterany:445711` + in-task pool-completeness assert). ⚠ Queue was saturated by the
    concurrent router_leak/composable_tv campaigns (445668–445694, peak exactly 4 GPUs), so
    445711 is tail-chained `afterany:445678:445680:445684:445685` (their terminal arrays) —
    the global 4-GPU cap holds under any scheduler order; this campaign starts only after
    theirs drains.
- **Results:** *pending — jobs queued behind the ctv/router_leak chain; results entry to
  follow (this is the pre-registration).*
- **What worked / hypothesis verdict:** *(pending)*
- **Observations:** Design notes: (i) "oracle routing" is deliberately the exact q2author
  route, not `routed_key_exact` — the latter is lexical name-matching (~0.86 routing
  accuracy; name-free + OOD queries fall back to shard_0), kept only for June-ladder
  comparability. (ii) `eval_routed_scaffold.py` runs on the plain (non-scaffolded) base
  here: author queries = base+expert, OOD = base — no scaffold confound; the scaffold
  variant stays a separate thread question. (iii) fq at 20-row forget sets is
  non-discriminative (KS quantization) — deletion is read from f_ppl/f_rouge + Δmu.
- **New questions / new hypotheses:** if H-k200-1 confirms, the peft_compose
  `H-ia3-route-200` (same experiment at 1.5 MB/pool IA³ scale) becomes the natural cost
  frontier; if it refutes, the isolation-cost mechanism (Exp-5b iso 0.399 vs joint 0.924)
  moves from subset metrics to the headline serving surface.
- **Next Steps:** monitor 445711/445712; on completion: results entry + thread README
  hypothesis ledger update + `log/README.md` timeline row; fold the routed-k200 row into
  the k-ladder table narrative (routing bottleneck story: k-independence of routing vs
  per-shard training dose).
