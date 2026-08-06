### Target Date: 2026-07-16 (E3 strict-isolation results — training DIVERGED; H14 refuted-as-implemented, capacity question stays open)
- **Hypotheses / what we're testing:** H14 + strict_all_on as pre-registered in
  [2026-07-15_round2-preregistration.md](2026-07-15_round2-preregistration.md): frozen-lora_A
  + disjoint_dead slices (40 lora_B rows/author/layer, gate/up only, scaffolded base,
  author-block steps, wd 0, clip 0) → routed mu ≥ 0.55, own-author prob ≥ 0.80,
  routed-unlearn forget_rouge ≤ 0.45. REFUTE branch: mu < 0.50 or own-prob < 0.6.
- **Setup:** Jobs **443562** e3-smoke ✓ (after the 443551 IRP CUDA fix —
  `freeze_lora_a_irp`, bit-equal gate, suite 21/21) → **443563** e3-train (1000 steps,
  6.5 min; `[memsinks] lora_A frozen (IRP seed 42)`, scheme disjoint_dead num_gen=0
  num_mem=8192, mask sha fc016e5178a9, author-block batching, distinct-ID guard OK) →
  **443564** evals 0-2%1 → **443565** probe. Chain ran at a strictly sequential 1-GPU
  footprint (`scontrol hold` + auto-release + ArrayTaskThrottle=1) because another
  session's arrays (%3+%1) held the rest of the global 4-GPU cap. Config
  `memsinks_tofu_1b_strict.json` (sha 2b4805cc89f4).
- **Results:**

  | label | mu | fq | forget_rouge | retain_prob | real_prob | world_prob | ppl f/r |
  |---|---|---|---|---|---|---|---|
  | strict_routed_full | **0.0014** | 0.3929 | 0.0598 | 0.0002 | 0.6305 | 0.6556 | 7275/8654 |
  | strict_routed_unlearn | 0.0014 | 0.3929 | 0.4647 | 0.0002 | 0.6305 | 0.6556 | 17.6/8654 |
  | strict_all_on | 0.0 | 0.135 | 0.0401 | 0.0 | 0.2521 | 0.2708 | 623016/591976 |

  **Probe (200 authors):** gen_only (= pure scaffolded base; every delta off) answer-prob
  **0.1396**; gen_own **0.00017** — the trained slices make own-author rows **~800× WORSE
  than the untouched base**; slice_increment **−0.139**; all_on 2e-6. **Training telemetry:**
  loss starts 2.68 (healthy scaffold baseline), holds ~2.8 for the first ~10% of steps, then
  **explodes to 8–12 and never recovers** (train_loss avg 8.51 vs M1's 0.80); train-end
  memgap probe own-mask prob 0.0001–0.0018 across all 7 probe authors.
- **What worked / hypothesis verdict:**
  - **H14 REFUTED — but AS-IMPLEMENTED, not as a capacity result.** The pre-registered
    REFUTE branch ("below the per-author capacity floor") is NOT licensed by this evidence:
    the loss trajectory shows a mid-training optimization BLOW-UP, and the slices don't
    merely fail to store content — they actively corrupt the forward (negative slice
    increment). A capacity floor would look like converged loss + low-but-positive recall.
    The capacity question (do 40 rows/author suffice?) remains OPEN.
  - **strict_all_on (collapse demo):** trivially confirmed (ppl 6e5) but uninformative given
    the model diverged — do not cite as a merging-collapse datapoint.
  - Infrastructure verdicts that DO stand: the routed wrapper served the scaffolded base
    correctly (real/world 0.63/0.66 healthy = OOD → all-zeros vector = pure base; deleted
    authors under unlearn get base, forget_ppl 17.6 vs 7275 — the serving seam works); the
    provenance/isolation machinery is sound (21/21 gates incl. the data-provenance bit-test).
- **Observations:** **Root cause (diagnosed, not yet re-tested):** the IRP freeze inits
  lora_A ~ N(0,1) — ≈45× the scale of PEFT's kaiming default (std ≈ 1/√2048 ≈ 0.022) — and
  rslora scaling α/√r ≈ 11.3 amplifies it further; E3 additionally DISABLED gradient
  clipping (max_grad_norm 0, chosen for exactness bookkeeping). Huge activations → huge
  lora_B gradients → runaway updates with no clip. The same std=1.0 lives in
  `train_lora_shard.apply_irp_projections` (SISA IRP mode) — a latent footgun there too,
  masked by clip 0.3 in that recipe. Second CUDA-vs-CPU-fixture lesson this round: the CPU
  gates validate math/provenance, not optimization scale.
- **New questions / new hypotheses:** **H14′ (scale-corrected strict arm):** same design
  with frozen lora_A at std 1/√hidden (kaiming-equivalent) and clipping restored (0.3 —
  clipping is a shared-scalar schedule effect under author-block steps: the clip factor for
  a step depends only on that author's own gradient, so the provenance claim survives;
  document). One ~7-min train + 3 evals + probe would answer the real capacity question.
  Alternative dial: lower lr. NOT run — user review gate (Round-2 scope was one fix arm).
- **Next Steps:** REPORT.md + thread/master ledger updates (done this entry cycle); review
  with user: (a) H14′ scale-corrected retry (~1.5 GPU-h), (b) stop and write the thread's
  final positioning into PATHS_FORWARD, or (c) both.
