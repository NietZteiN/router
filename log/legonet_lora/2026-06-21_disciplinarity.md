### Target Date: 2026-06-21 (legonet_lora cont. — newer model + disciplinarity)
- **Goal:** (1) re-run the DBpedia legonet_lora pipeline on a newer Llama; (2) add a cluster
  field-disciplinarity analysis. User cap: ≤8 GPUs (LEGO_ARRAY_CAP default → 8).
- **Setup:** New `configs/legonet_l32_3b.json` (base meta-llama/Llama-3.2-3B-Instruct; v2 recipe
  canary×5/6ep/r16; reuses corpus dbpedia_n4000 + keys n32 — model-independent). Chain 436011→
  436012[0-31%8]→436013∥436014. Disciplinarity: new `analyze_disciplinarity.py` +
  `collect_disciplinarity.py` + `submit_disciplinarity.sh`; 1000-record per-record eval on 7B-v2 +
  3.2-3B (jobs 436130[0-1]→436131) → `/storage2/.../DISCIPLINARITY_REPORT.md`.
- **Results — newer model TRANSFERS:** Llama-3.2-3B MMLU legonet **0.583** vs base **0.600** (utility
  preserved; 3.2-3B base stronger than Llama-2's 0.46), retained PPL 24.5→5.6; exactness all
  structural_ok, affected rel_l2 (2.7/4.7/3.9e-2) ≈ untouched floor (2.2/2.9/3.5e-2) → distributional
  exactness holds. Forget still dilution-limited (canary under-memorized, weaker on 3B).
- **Results — disciplinarity (n=32; field := DBpedia class):** strict 90/10 = 30 interdisciplinary /
  2 single-discipline (NaturalPlace .93, Plant .97) / 0 highly-inter; graded = 10 pure / 14 mixed /
  8 highly-mixed. **Purer clusters memorize better** (7B-v2 graded: retained EM 0.757/0.733/0.702,
  PPL 2.82/3.10/3.59, VerbMem 0.475/0.376/0.345 for pure/mixed/highly-mixed; single-disc VerbMem
  0.737 vs inter 0.382). **canary_em ~flat ≈0.05** across buckets (random-code forget is
  topic-independent). Same trend on 3.2-3B.
- **Observations:** monotone purity→memorization is the expected effect (homogeneous expert = tighter
  target dist). Strict 90/10 leaves almost all clusters "interdisciplinary"; graded buckets give the
  powered contrast. 80-record eval was too small (1 single-disc record) → 1000-record eval fixed it.
- **Next Steps:** report updated (LEGONET_LORA_REPORT.md §4.7 disciplinarity, §4.8 3.2-3B transfer).
  Optional: train experts AS classifiers for true LegoNet-accuracy units; balanced k-means to add
  single-discipline clusters; n>576 for the SISA cost crossover.

#### RESULTS (2026-06-23, cont.) — Llama-3.2-1B (TOFU-leaderboard model) vanilla vs balanced
- 1B vanilla (436133-138): legonet_full 0.5118/fq 0.0156; **legonet_unlearn 0.5092/fq 0.9988**,
  forget_ppl 3.27->11.10, retain 3.23. forget10 affected 15/32, untouched 17. (Assignment byte-identical
  to the 7B vanilla — keys are MiniLM-on-answers, base-model-independent.)
- 1B balanced (436139-144, cap=29): legonet_full 0.4838; **legonet_unlearn 0.4853/fq 0.9988**,
  forget_ppl 3.13->11.46. forget10 affected 12/32, untouched 20 (hub 135->29).
- vs 1B SISA merged_dare_ties k=10 = 0.4236/0.3929, base ~0.418: LegoNet beats it on BOTH axes.
- **Finding:** balancing IMPROVES locality (untouched 17->20, affected 15->12) at a small utility cost
  (0.509->0.485) — the hub was a LOCALITY problem, not a utility one (the big well-trained hub served
  its authors fine; spreading them to next-nearest clusters slightly hurt routing match). 1B forgets
  near-perfectly (fq 0.999 > 7B's 0.808): smaller model's truth-ratio dists align tighter with the oracle.
- Report: legonet_lora/reports/LEGONET_TOFU_REPORT.md (added). Core TOFU study COMPLETE across 7B + 1B
  (vanilla + balanced) + locality gradient. Optional remaining: extended-caps headline; forget01/single
  eval (locality already shown via affected counts).

#### Extended-caps headline (2026-06-23, cont.) — Llama-3.2-1B vanilla (jobs 436271-276)
- Extended (truth cap 120 / ROUGE 200): legonet_full 0.4947/fq 0.0004; **legonet_unlearn 0.5011/fq 0.890**,
  forget_ppl 11.10, retain_ppl 3.16. Utility stable vs smoke (0.509->0.501); KS forget_quality settles
  from low-power smoke 0.999 (30 samp) to trustworthy **0.890** — publication-grade. Still beats SISA 1B
  (0.424/0.393) decisively on both axes. Reports updated (LEGONET_TOFU_REPORT.md + full-report addendum).
  TOFU LegoNet study COMPLETE: 7B + 1B (vanilla/balanced) + locality gradient + extended headline.

#### Cross-model LegoNet-on-TOFU smoke vs base baseline (2026-06-23, cont.)
- Goal: compare LegoNet (n=32/k=3, 6ep vanilla) across a small-model trio vs the base-model baseline.
- Models: TinyLlama-1.1B (jobs 436340-345), Llama-3.2-1B (reused 436133-138), phi-2 2.7B (436346-351);
  base evals 436352-354. New configs: legonet_tofu_{tinyllama,phi2}.json. Setups: empty_adapters=0,
  forget10 affected=15/32 (identical across models — keys are MiniLM-on-answers, base-independent).
- Cross-model table (smoke, model_utility / forget_quality):
  | model | base mu | legonet_full | legonet_unlearn | f_ppl full->unlearn |
  |---|---|---|---|---|
  | TinyLlama-1.1B | 0.399/0.393 | 0.513/0.035 | 0.503/0.958 | 2.62->6.79 |
  | Llama-3.2-1B   | 0.380/0.393 | 0.512/0.016 | 0.509/0.999 | 3.27->11.10 |
  | phi-2 (2.7B)   | 0.435/0.999 | 0.491/0.594 | 0.485/0.999 | 5.36->7.07 |
- Consistent across architectures: legonet_full lifts utility well above base (0.49-0.51 vs 0.38-0.44),
  experts learned (ppl 10-20 -> 3-5); legonet_unlearn forgets forget10 cleanly (fq 0.96-0.999, f_ppl up)
  with retain stable. CAVEAT: base_model fq reads high (phi-2 0.999) because the untrained base passes the
  forget test trivially (never trained on ANY author) — its real signal is the low utility floor.
- phi-2 quirk: legonet_full fq 0.594 (vs Llamas ~0.02) — phi-2 memorizes the forget authors less strongly;
  unlearn still reaches 0.999. Whole batch finished well under the worst-case ~2.5h (idle 12-GPU cluster).
