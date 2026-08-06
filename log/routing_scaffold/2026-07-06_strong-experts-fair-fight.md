### Target Date: 2026-07-06 (strong experts + extended replication + the matched-capacity fair fight)
- **Hypotheses / what we're testing:** H4: the weak-expert bottleneck (retain_prob 0.304, legacy r8/α16/e3
  shards) is the margin limiter — retraining the k=10 experts with the frozen winner recipe (r32/α64/e5)
  *on the scaffolded base* (no train/serve mismatch) lifts retain toward full-FT and widens the margin.
  H5: the smoke headline (0.556) survives extended caps. H6 (the fair fight): the routed architecture
  beats a **matched-capacity** single model — same scaffolded base + ONE r32/α64/e5 LoRA on all 200
  authors — not just the weak legacy `_ft` (r8/3ep). CONFIRM = routed ≥ matched-FT; REFUTE = matched-FT
  catches up (then the claim reverts to "competitive utility + exact deletion").
- **Setup:** Llama-3.2-1B, smoke (H5 extended), seed 42. Strong experts: `train_lora_shard.py --k 10`
  flag-free defaults with `--model_name` = the scaffolded base → `checkpoints/..._experts_scaf_k10`
  (jobs 440232/440233; KS ref copied in). Extended: `eval_routed_scaffold.py --extended` on the weak
  config (440234). Fair baseline: `--shard_id 0 --k 1` (all 200 authors), same recipe, same scaffolded
  base → `..._ft_strong_scaf`, eval via `--preloaded_adapter` (440424/440425). Serving:
  `eval_routed_scaffold.py` (OOD-aware; `--delete_shard 9` for the exactness check).
- **Results:** **Strong routed+scaffold mu 0.7509** (retain_prob 0.854, real 0.630, world 0.656,
  own-author forget_rouge 0.894); deletion: mu **0.7509 → 0.7509 identical**, fq 0.0003 → **0.3929**
  (= scaffold-floor/never-trained), forget_rouge 0.894 → 0.465. **Extended** weak config: mu **0.5564**
  ≈ smoke 0.5559. **Matched-capacity full-FT: mu 0.6372** (retain_prob 0.874, real 0.437, world 0.548,
  forget_rouge 0.854).
- **What worked / hypothesis verdict:** **H4 SUPPORTED** — retain_prob 0.304→0.854, mu 0.556→0.7509.
  **H5 SUPPORTED** — 0.5564 extended ≈ 0.5559 smoke. **H6 SUPPORTED, decisively** — routed 0.7509 vs
  matched-FT 0.6372, **margin +0.114** from identical ingredients (same base, same scaffold, same
  recipe, same data). 0.7509 is the best mu of any track (sift_masks 0.737, clamu 0.647).
- **Observations:** The mechanism of the win is in the decomposition: matched-FT memorizes authors as
  well as the experts (retain 0.874 vs 0.854 — a tie) but **fine-tuning on all 200 authors damages the
  scaffold's general knowledge** (real 0.630→0.437, world 0.656→0.548 = catastrophic forgetting inside
  the adapter), while the routed architecture **structurally protects** it — OOD queries never touch an
  expert, so real/world stay at the scaffold ceiling. So the honest causal claim: routing does not
  memorize better; it **isolates fine-tuning damage**, and separately provides exact O(1) deletion
  (verified byte-identical again at strength). No silent-failure signs: ppl sane, fq behaves exactly as
  the never-trained oracle predicts pre/post deletion.
- **New questions / new hypotheses:** (1) Does the +0.114 margin hold at extended caps and across seeds
  (43/44)? (2) Does it hold as experts shrink (k=20/50 → fewer authors/expert, cheaper deletion units)?
  (3) Can matched-FT be rescued by mixing Alpaca replay into its training (the classic CF mitigation) —
  and does routed still win then? (4) OOD-router realism: replace the exact q2author lookup with the
  encoder cluster-ID router (the realistic setting) and measure the routing-error cost.
- **Next Steps:** (1) extended-cap + multi-seed on the strong config and the matched baseline (the
  headline pair). (2) The Alpaca-replay matched-FT control (3). (3) Then write-up: thesis = "routed
  isolated experts + public scaffold: +0.11 utility over the best matched single model, because routing
  isolates fine-tuning damage — with certified O(1) exact deletion."
