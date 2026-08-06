# Plan — 7B one-author-per-shard (k=200) merge tables, within a 1-GPU-day budget

## Context
Deliver consolidated model-merging results on **Llama-2-7B-chat-hf, one author per shard
(k=200 per-author task vectors)**, capped at **≈1 GPU-day** total. Requested tables:
1. **Tables 1+2 combined** — every merge operator, one table:
   `method · mu (merged) · mu base · mu finetuned · model used · fq · f_rouge · exact-deletion · rank · N`.
2. **Dilution table** as shard count increases (+ per-author N-merge ladder).
3. **Table 5** (PEFT bake-off) and **Table 7 ctrl/wd/lin** (ctv training-time arms), on 7B.

**Locked decisions (user):**
- Budget package = **merge battery + PEFT (Table 5) + ctv ctrl/wd/lin (Table 7 partial)**, ~40 GPU-h.
- Memory-wall operators → **r8 fallback, annotated**.
- **EXCLUDED (out of a 1-GPU-day budget):** Table 6 (SIFT-Masks + ClAMU) and ctv `ds` — they need
  ~130–210 GPU-h of deterministic **fp32 7B full-FT** plus multi-day new-code engineering. Deferred to a
  separate budget; noted as "not run (needs distributed fp32 full-FT)" in the deliverable.

Anchors on disk (`reports/centered/nmerge_mu.csv`): **mu base = 0.426**, **mu finetuned = 0.756**.
All smoke-tier (ROUGE≤50/retain≤80/truth≤30, seed 42).

## Hard constraint: k=200×r32 memory wall (`CLAUDE.md:557`)
r32×200 ≈ 65 GiB ⇒ does not fit a 46 GiB A40 for the in-model merge path.
- **r32-materializable (no wall):** additive_mean/sum, centered_pool, centered_lowrank, sparse_* — already materialized.
- **In-model-only → r8 pool `_k200_r8_e5_lr1e4`:** della_ties, fisher, knots_ties, tsv, slerp, breadcrumbs,
  lorahub, subtract_orth, regmean, ties (linear/dare_ties already done at r8).
- **JD:** `jd_collection.py` mode-B.
Table will therefore be mixed-rank (r32 vs r8, annotated) — matching the existing `k=200 (r8)` precedent.

## Runtime evidence (measured)
- Materialized `--preloaded_adapter` eval, k=200, smoke: **~4 min/eval** (`CENTERED_ANCHOR_REPORT:79,197`).
- Per-author train: e5 ~60–90 s; e25 ~1–2 min; whole 200-author e25 pool ≈1.7–6 GPU-h.
- Data-required merge (fisher/regmean/lorahub) k=200: `--merge_num_examples 32` → 6,400 passes each.
- **4-GPU global cap**; queued `%N` throttles must sum ≤4; check `squeue -u jack` before every submit.

## Execution — priority order, self-limited to ≈1 GPU-day (~40 GPU-h ceiling)
Run top-to-bottom; **if the running total nears the 1-GPU-day mark, stop and drop the remaining
lower-priority items** (tail = ctv-lin, then optional r32 bridge merges).

### Phase 0 — assemble existing (0 GPU)
Fold `reports/nmerge_mu.csv` (+`e25/`,+`centered/`) into the combined table's r32 rows and the N-ladder.
Re-run only the one incomplete cell `nmerge_cr16_N64_s42` (non-svd): `submit_nmerge.sh
configs/nmerge_centered_7b.json eval` (single task, ~4 min).

### Phase M — complete the merge battery (existing code) — **~22 GPU-h** [P1]
1. **w5 sparse evals** (merges already materialized): run `_ctv_sparse/results/smoke/eval_manifest_sparse.txt`
   (101 rows) via the `submit_ctv.sh configs/sparsify_7b.json` eval path. ~7 GPU-h.
2. **r8 gapfill + extras** on `_k200_r8_e5_lr1e4`: the 6 queued labels (regmean, fisher, ties, knots_ties,
   breadcrumbs_s0.00177/_s0.005) **plus** della_ties, tsv, slerp, lorahub, subtract_orth. Command:
   `EVAL_MANIFEST=<gapfill+extras> EVAL_EXTRA_ARGS="--smoke --merge_num_examples 32" EVAL_TIME=05:00:00
   ARRAY_CAP=<fits cap> bash submit_eval.sh checkpoints/Llama-2-7B-chat-hf_k200_r8_e5_lr1e4
   meta-llama/Llama-2-7B-chat-hf 200`. ~11 labels → ~10–15 GPU-h.
3. **JD mode-B** (jd_full/jd_diag) via `submit_jd_highk.sh` (build→materialize→eval). ~4 GPU-h.
4. *(optional, budget-permitting)* r32 dare_ties/della via `merge_subset` CPU-base bridge (CPU merge + 4-min eval).

### Phase P — Table 5 (PEFT bake-off) on 7B, k=10 (config-only) — **~6 GPU-h** [P2]
New `configs/peft_bakeoff_7b.json` (copy `_1b.json`; `model_name`→`meta-llama/Llama-2-7B-chat-hf`,
`dir_template`, 7B `ks_reference`; VeRA/IA³ `target_modules` valid on 7B MHA). Run Phase A:
`submit_peft_bakeoff.sh configs/peft_bakeoff_7b.json all` (40 train %4 + ~28 eval). Report the composed
plateau vs base 0.426 / ft 0.756; the k=200 "Phase B" stays a cited-not-run row (gated at composed mu ≥ 0.55).

### Phase C — Table 7 ctrl/wd/lin on 7B (config-mostly) — **~12 GPU-h** [P3, drop first if over budget]
New `configs/ctv_7b_{ctrl,wd,lin}.json`. `ctrl` = pure model swap; `wd` = re-derive `pool_size×r_prime=d_out`
(7B non-GQA d_out=4096) + n_ladder cap; `lin` = fp32 base (+grad-checkpoint if jvp OOMs). ~72 e25 trains +
~100 evals via `submit_ctv.sh` (gate→train→verify→merge→eval). Serving/eval infra already works at 7B k=200.

## Deliverable
One markdown report `tofu_sisa_lora/reports/MERGE_METHODS_7B_K200_<date>.md` (mirrors the 1B
`MERGE_METHODS_RESULTS_2026-07-21.md`):
- **Combined merge table (1+2)** — every operator, mixed-rank (r32: additive/centered/sparse/JD;
  r8 annotated: della/fisher/knots/tsv/slerp/lorahub/subtract_orth/regmean/ties/linear/dare), requested
  columns, base 0.426 / ft 0.756 on every row.
- **Dilution table** — DARE-TIES (+cheap operators) vs k∈{4,10,20,50,100,200}; per-author N-merge ladder
  (additive_mean flat vs centered decay) at k=200.
- **Table 5** (7B PEFT k=10) and **Table 7** (7B ctrl/wd/lin).
- One line noting Table 6 (SIFT/ClAMU) + ctv-ds excluded by the 1-GPU-day budget (need fp32 7B full-FT).

## Total estimate
**≈40 GPU-h ⇒ ~1 GPU-day** under the 4-GPU cap (Phase M ~22 + P ~6 + C ~12). Governor: run in priority
order (M→P→C); if the total approaches 1 GPU-day, stop and drop Phase C's lin arm (and the optional r32
bridge) to stay within budget. Wall-clock ~1 day given queue/NFS overhead on the many small jobs.

## Verification
- CPU gates before each SLURM job: `test_merge_subset.py`, `test_merge_extra.py`, `test_sparsify_pool.py`
  (M); `test_compose_peft.py` (P); `test_struct_tv.py` (C). Smoke-first before any full array.
- Assemble via `analyze_nmerge.py --config …` → `nmerge_mu.csv`; `collect_results.py --root … --smoke`.
  Cross-check every mu against its on-disk result JSON; watch NaN mu + retain_ppl explosions; confirm
  base 0.426 / ft 0.756 reproduce.
- Log per repo protocol: dated entries under `log/merge_mechanism/` (+ `log/composable_tv/`,
  `log/peft_compose/` as touched); refresh each thread README + `log/README.md` timeline.
