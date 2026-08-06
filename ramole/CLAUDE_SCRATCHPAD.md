# CLAUDE_SCRATCHPAD — follow-up campaign (E1–E6) deliverables

## Task (2026-07-02)
Build the follow-up campaign plumbing: 4 configs + `submit_followup.sh` (wave1/wave2 SLURM
orchestrator) + `collect_followup.py` (report aggregator). No jobs submitted now — CPU checks only
(STUB=1 render, json.load configs, collector dry-run against current disk).

## Design decisions
- **E1 seed configs (`_s43`/`_s44`)**: new `name` → new run dir (fresh router per seed), but
  `"retriever_run": "ramole_l32_3b_n32_k3"` so the trained retriever + lora_index are SHARED
  (never retrained — seed variance isolates the router). Only `base_seed` changes; corpus seed
  stays 42 (same data).
- **E4 k-sweep configs (`_k5`/`_k8`)**: `name` UNCHANGED (`ramole_l32_3b_n32_k3`) because k is a
  serve-time knob — same trained router/retriever/index. Collision hazard: default eval labels
  (`router_keys_iid`) would overwrite the k=3 files → orchestrator always passes
  `--label_suffix k5|k8` (eval_ramole appends to label; `--out` derives from label).
- **E6 batch15**: sample 15 record ids with `random.Random(42)` from the source
  `records.jsonl`, excluding rec_000000/1/2 (already deleted as d0/d1/d2 — avoid overlap), then
  one `unlearn.py --tag d_batch15` call. Both as one-line python inside the sbatch heredoc
  (compute node, not login).
- **Dependencies**: DBpedia routing audit afterok E6-unlearn (audit covers d_batch15);
  E1 evals afterok their trains; E3 evals afterok their audits (rebuilt index files are written
  by the audit scripts); E6 evals afterok the unlearn. `dep()` helper filters ""/STUBJOB so
  `PHASE=wave2` standalone and STUB mode submit dep-free.
- **Referenced-but-not-yet-written scripts** (owned by sibling tasks): `routing_audit.py`,
  `analyze_router.py`, `benchmark_serving.py` (ramole); `routing_audit_tofu.py`,
  `analyze_router_tofu.py` (tofu). STUB rendering doesn't execute them.
- **Collector**: never crash on partial disk state — every cell dashes when its file/key is
  absent; audit/alpha/throughput schemas are rendered generically (flatten numeric leaves /
  matrix-of-scalar-dicts) since those producers are being written in parallel.

## Constraint check
- Hardware: all GPU work in sbatch (gres=gpu:1, exclude=sprint4); arrays capped %4.
- Storage: configs/scripts in /home/jack (code); all outputs under /storage2 run dirs.
- Correctness: seeds pinned (43/44 explicit; sampler seeded 42); CPU checks before any submit.

## State
- [x] configs s43/s44/k5/k8 written + json.load-validated
- [x] submit_followup.sh written; `bash -n` + `STUB=1 ... all|wave1|wave2` render clean
- [x] collect_followup.py written; ran against current disk (partial cells, no crash) →
      /storage2/jack/checkpoints/ramole/FOLLOWUP_REPORT.md
- [ ] real submission (operator decision; after sibling scripts land)

## Review: follow-up campaign deliverables — adversarial pass 2026-07-02
- Full seam check against real callers: eval_tofu.py / train_router_tofu.py / analyze_router{,_tofu}.py /
  routing_audit{,_tofu}.py / benchmark_serving.py / eval_ramole.py / legonet unlearn.py argparse — all flags match.
  `routing_audit.py` LANDED since the builder finished: its argparse (--config --tags --n_retain --device --out)
  matches the wave1 E3-DBpedia job exactly, and its rebuilt-index filename (`rc.Paths(cfg).run_dir/lora_index_n{n}_ex{tag}.npy`)
  matches what `eval_ramole.py --index_policy rebuilt` loads. The "confirm argparse when it lands" TODO is discharged.
- FIXED (collect_followup.py): the DBpedia audit section rendered routing_audit.py's REAL schema
  (`orphan_pooled` + per-tag `selection_shift`/`index_displacement`) as "(no 'policies' block found)" —
  E3 DBpedia numbers would silently never appear. Now rendered (pooled policy table, per-tag shift rows,
  per-tag displacement with computed mean/min cos_affected_top1 and bit-equal fraction). Also hardened three
  non-dict-shape paths (`c in (v or {})`, `tags` non-dict, `(disp.get(..) or {}).values()`) that previously
  dumped the whole E3 section to the SECTION ERROR backstop on schema drift; corrupt shapes now render dashes.
- REPORTED, not fixed: (1) DBpedia E1 seed arm — base_seed also seeds `rc.cluster_split`, so s43/s44 routers
  train on a DIFFERENT 40% train-cluster subset than s42; measured "seed variance" includes split variance
  (config comments claim only init/dropout/data-order; TOFU side is clean — retain authors fixed 0..179).
  A fix needs a separate split-seed knob in ramole_common (library change). (2) 'all' can have up to 13
  dep-free single-GPU jobs eligible at once (wave1 9 + E4 4) vs 12 cluster GPUs / repo's ≤8 note — SLURM
  queues the excess, but consider submitting wave2/E4 later.
- Verified by execution: seeded E6 sample = 15 unique ids excl rec_000000-2; affected union = 24 adapters /
  10,008 member records ≈ 5h vs 8h wall (a0 measured 820s/531 records); STUB=1 renders 14/14 scripts;
  collector rc=0 on real disk, empty roots, corrupt JSON, hostile schemas. Builder test re-run: ALL PASSED.

## Review: benchmark_serving.py (E5) — adversarial pass 2026-07-02
- Re-ran `tests/test_benchmark_serving.py` (CPU, offline) → ALL PASSED; CLI `--smoke` end-to-end → table + JSON written to tempdir.
- Pitfalls verified: timing = generate only (encode/merge/load outside); cuda.synchronize both sides; warmup idx-0 discarded; padding_side=left in _encode; rm freed (del+gc+empty_cache) before LegoNet load; no (a)==(b) assert; single_expert_id preserves nearest-first assignment order (KNNRouter sorts by distance — no sorted-tuple bug).
- Seams: submit_followup.sh E5 flags match argparse; collect_followup sec_e5 reads exactly the written keys via --out throughput.json. No fixes needed.
- Noted (not fixed): per-row eos early-stop overcounts pad-fill tokens in multi-row batches (batch-level stop is respected; count = rows×decode-steps = actual compute, consistent across modes); `--iters 0` would ZeroDivisionError in _agg (unreachable at real call sites).
