# STATUS — 7B one-author-per-shard (k=200) build

Budget: **≈1 GPU-day** (~40 GPU-h). Package = merge battery + Table 5 (PEFT k=10) + Table 7 ctrl/wd/lin.
Excluded by budget: Table 6 (SIFT/ClAMU) + ctv-`ds` (need ~2 GPU-days of fp32 7B full-FT + new code).

Anchors on disk (`~/tofu_sisa_lora/reports/centered/nmerge_mu.csv`): **base 0.426**, **ft_r32 0.756**.

## The k=200×r32 memory wall (why the table is mixed-rank)
`CLAUDE.md:557` — r32×200 ≈ 65 GiB > 46 GiB A40. So:
- r32-materializable (no wall): `additive_mean/sum`, `centered_pool`, `centered_lowrank`, `sparse_*`.
- in-model-only → **r8 pool** `Llama-2-7B-chat-hf_k200_r8_e5_lr1e4`: della/fisher/knots/tsv/slerp/
  breadcrumbs/lorahub/subtract_orth/regmean/ties (+ linear/dare_ties already done).
- JD → `jd_collection.py` mode-B.

## Coverage inventory (7B k=200; all smoke — no `results/extended/`)
| Operator | Status | Where / rank |
|---|---|---|
| additive_mean / additive_sum | DONE (N=2..200) | `_nmerge_r32`(e5)+`_e25`; r32 |
| centered_pool | DONE (N=2..64) | `_nmerge_r32_centered`; r32(svd1024) |
| centered_lowrank cr16 | DONE (N=2..200); `N64` non-svd = `.progress` only → RE-RUN | `_nmerge_r32_centered`; r16 |
| linear, dare_ties | DONE **r8 only** | `_k200_r8_e5_lr1e4` |
| sparse_* (dare0p5/0p9/0p99sum/topk0p25/hash) | MATERIALIZED, 0 evals (101-row manifest) → RUN | `_ctv_sparse` |
| regmean, fisher, ties, knots_ties, breadcrumbs×2 | QUEUED `eval_manifest_gapfill.txt` → RUN | `_k200_r8` |
| della_ties, tsv, slerp, lorahub, subtract_orth | TODO (r8) | — |
| jd_full / jd_diag | TODO (mode-B) | — |
| Table 5 PEFT (7B) | TODO — new `configs/peft_bakeoff_7b.json`, k=10 | — |
| Table 7 ctrl/wd/lin (7B) | TODO — new `configs/ctv_7b_*.json` | — |

## Runtime evidence (measured, cited)
- Materialized `--preloaded_adapter` eval, k=200, smoke: **~4 min/eval** (`CENTERED_ANCHOR_REPORT:79,197`).
- Per-author train: e5 ~60–90 s; e25 ~1–2 min; whole 200-author e25 pool ≈ 1.7–6 GPU-h.
- Data-required merge k=200: `--merge_num_examples 32` → 6,400 passes each (`CLAUDE.md:629`).
- **4-GPU global cap**; queued `%N` throttles must sum ≤ 4; check `squeue -u jack` before every submit.

## Next actions (priority order; stop at ~1 GPU-day, drop tail if hot)
1. **Phase 0** (0 GPU): re-run `nmerge_cr16_N64_s42` — `bash submit_nmerge.sh configs/nmerge_centered_7b.json eval` (1 task).
2. **Phase M** (~22 GPU-h): (a) w5 sparse 101 evals via `submit_ctv.sh configs/sparsify_7b.json`;
   (b) r8 gapfill+extras — `EVAL_MANIFEST=<gapfill+extras> EVAL_EXTRA_ARGS="--smoke --merge_num_examples 32"
   EVAL_TIME=05:00:00 ARRAY_CAP=<fits cap> bash submit_eval.sh
   checkpoints/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4 meta-llama/Llama-2-7B-chat-hf 200`; (c) JD via `submit_jd_highk.sh`.
3. **Phase P** (~6 GPU-h): `configs/peft_bakeoff_7b.json` → `submit_peft_bakeoff.sh ... all`.
4. **Phase C** (~12 GPU-h, drop `lin` first if over budget): `configs/ctv_7b_{ctrl,wd,lin}.json` → `submit_ctv.sh`.

CPU gates first (per phase): `test_merge_subset.py` / `test_merge_extra.py` / `test_sparsify_pool.py` /
`test_compose_peft.py` / `test_struct_tv.py`. Assemble: `analyze_nmerge.py` → `nmerge_mu.csv`;
`collect_results.py --root … --smoke`. Then update `RESULTS_TABLES.md` and log per repo protocol
(`log/merge_mechanism/`, `log/composable_tv/`, `log/peft_compose/` + `log/README.md`).

Full plan: see `PLAN.md`.
