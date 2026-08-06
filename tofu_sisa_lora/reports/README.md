# TOFU SISA-LoRA reports

| File | Description |
|------|-------------|
| [SMOKE_EVAL_REPORT.md](SMOKE_EVAL_REPORT.md) | Full smoke eval write-up (setup, tables, interpretation) |
| [EXTENDED_EVAL_REPORT.md](EXTENDED_EVAL_REPORT.md) | Llama-3.2-1B extended eval |
| [EXTENDED_EVAL_REPORT_3B.md](EXTENDED_EVAL_REPORT_3B.md) | Llama-3.2-3B extended eval + ROUGE explanation |
| [MERGE_METHODS_RESULTS_2026-07-21.md](MERGE_METHODS_RESULTS_2026-07-21.md) | **From-scratch results report: every merge operator tried, with mu / forget-quality / recall per pool** — the 0.42–0.48 merge ceiling, dilution law, SIFT/ClAMU/routing contrasts, ctv G1 verdicts, DX diagnostics, exactness classes |
| [MERGE_VS_ROUTING_MASTER_2026-07-24.md](MERGE_VS_ROUTING_MASTER_2026-07-24.md) | **Master "merging doesn't work" tables + router-vs-merger, all pools in one place — self-contained (readable from scratch)** — §0 primer (TOFU / task vectors / exact-vs-approximate / separability) + metric formulas, then the grand master merge table banded (in-band plateau / broke-downward / partial-win) with mu·Δbase·ΔFT·fq·f_rouge·exactness, dilution law, PEFT/full-FT operator-independence, the routing inventory + per-pool ceiling-vs-headroom + "selection is the carrier" + leak/exactness caveats, and Appendix A (method & term dictionary) + Appendix B (37 references, grounded in `papers/RELATED_WORK.md`). Numbers spot-verified vs live JSONs (⚠ `all_metrics_smoke.csv` is stale) |
| [TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md](TASK_VECTOR_MERGE_STRATEGIES_REFERENCE_2026-07-18.md) | User-provided reference (verbatim): 14 merge-method specs + common harness + exactness test; provenance header maps each method to ledger coverage |
| [DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md](DISJOINT_MASK_TOFU_GO_NOGO_2026-07-18.md) | User-provided assessment (verbatim): disjoint-mask go/no-go, kill thresholds, SIFT-Masks positioning; provenance header corrects stale anchors (0.664→0.7509 routed) |
| [all_metrics_smoke.csv](all_metrics_smoke.csv) | Smoke runs in one spreadsheet |
| [all_metrics_extended.csv](../checkpoints/all_metrics_extended.csv) | Extended runs in one spreadsheet |
| [generate_smoke_report.py](generate_smoke_report.py) | Refresh CSV + auto tables from JSON |

Raw per-adapter JSON remains under `../checkpoints/<model>/results/smoke/`.

```bash
bash submit_llama_merge_smoke.sh   # 17 labels, <55 min/task, sprint1-3 only
python collect_results.py --root ../checkpoints --smoke
cp ../checkpoints/all_metrics_smoke.csv all_metrics_smoke.csv
python generate_smoke_report.py --full --model-slug Llama-3.2-1B-Instruct
```

Smoke policy: `slurm_nodes.sh` excludes sprint4; `TOFU_SMOKE_TIME=00:55:00`; ROUGE/retain/truth caps in `eval_tofu.py` (`SMOKE_*` constants).

Extended eval:

```bash
bash submit_overnight.sh 4 meta-llama/Llama-3.2-3B-Instruct   # train shards first
bash submit_llama_extended_eval.sh 4 meta-llama/Llama-3.2-3B-Instruct
python collect_results.py --root ../checkpoints --extended
python generate_smoke_report.py --full --extended --model-slug Llama-3.2-3B-Instruct \
  --out EXTENDED_EVAL_REPORT_3B.md --compare-slugs Llama-3.2-1B-Instruct,Llama-3.2-3B-Instruct
```

Extended policy: `TOFU_EXTENDED_TIME=02:30:00`; caps `EXTENDED_*` in `eval_tofu.py`; results under `results/extended/`.

Example generations (for reports):

```bash
bash submit_sample_generations.sh checkpoints/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-3B-Instruct 4
python generate_smoke_report.py --full --extended --model-slug Llama-3.2-3B-Instruct --out EXTENDED_EVAL_REPORT_3B.md
```

Samples live under `checkpoints/<slug>/results/extended/generations/<label>.json`.
