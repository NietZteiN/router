# CLAUDE_SCRATCHPAD — k=200 per-author task vectors + ORACLE ROUTING (2026-07-19)

**Task (user):** "did we run the experiment of 200 task vectors on tofu and oracle routing on
them? if not let's run." Audit verdict: **never run with adequately-trained task vectors.**
(Previous campaign — centered merging + key firing — completed 2026-07-16; see
`reports/CENTERED_ANCHOR_REPORT_2026-07-16.md` and git-less provenance in log entries.)

## Prior-state audit (what exists)
- 2026-06-12 k-scaling sweep (sisa_lora): `routed_key_exact` @k=200 completed ONLY for weak
  pools — r8/e5 mu **0.4728** (`_no199` 0.4728, deletion utility-free), r1/e5 **0.4212**
  (≈ no-op). Bottleneck = per-author undertraining (~6 opt steps at e5), NOT routing (routing
  acc k-independent ~0.86; k=50 routed mu 0.7147). r32 eval **blocked** by the fp32-cast
  memory law: 200 × r32 ≈ 65 GiB > A40 46 GiB. The pre-registered "steps-matched k=200 routed
  arm" (prediction: routed mu ≈ 0.7 flat in k) was never executed.
- merge_mechanism trained strong per-author pools but never route-served them:
  - `Llama-2-7B-chat-hf_k200_r32_e5_lr1e4/` — **200/200 authors** (iso f_rouge 0.4895 vs base
    floor 0.4038 — weak fact-servers).
  - `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4/` — **20/200 authors** (perm42[:20]; near-perfect:
    N=1 subset retain_prob 0.9992, subset ppl 1.06; job 442576, ~1 min/author, 250 MB/shard).
- peft_compose `H-ia3-route-200` open (IA³ variant — related but not this task).
- ⚠ Routing-label semantics: eval_tofu's `routed_key_exact` = LEXICAL name router (~0.86 acc;
  name-free questions fall back to shard_0, incl. ALL OOD queries — the known composition-bug
  channel). The TRUE ORACLE route is the exact `q2author` lookup in `eval_routed_scaffold.py`
  (OODAwareRoutedModel: author → own expert, OOD → base with adapters disabled). The oracle
  arm is the headline; `routed_key_exact` is kept for June-ladder comparability.

## Hypotheses (pre-registered here; log entry to follow in log/routing_scaffold/)
- **H1 (headline):** oracle (q2author) routing over 200 WELL-TRAINED per-author task vectors
  (e25) holds utility ≈ flat in k: oracle-routed mu ≥ 0.70 (anchors: k=50 routed 0.7147,
  k=10 scaffold-routed 0.7509, matched-FT 0.6372, joint-ft ≈ 0.7435–0.7563, base 0.418–0.426).
  CONFIRM: mu ≥ 0.70. REFUTE: mu < 0.60 (per-author isolation itself costs utility even with
  strong experts — cf. iso rouge 0.49–0.51 vs joint-ft 0.91 on the same author @e5).
- **H2 (training dose):** e5-pool routed ≪ e25-pool routed (June's bottleneck was
  undertraining, not routing). CONFIRM: mu(e25 oracle) − mu(e5 oracle) ≥ 0.10.
- **H3 (O(1) deletion at author granularity):** deleting author 199's expert is utility-free:
  |Δmu| ≤ 0.005 vs full; forget signal = f_ppl rise toward base (fq non-discriminative at
  20-row forget sets — KS quantization, pre-registered June; read ppl/rouge).
- **H4 (OOD fallback cost):** with e25 experts, lexical `routed_key_exact` (OOD → shard_0
  expert) pays a real/world penalty vs oracle OOD-aware serving (e25 experts damage
  world_prob 0.72→0.60 when misapplied). CONFIRM: mu(oracle) − mu(lexical) ≥ 0.03.

## Design
1. **Code:** kill the fp32 memory wall with lazy adapter loading —
   `load_all_shard_adapters(..., lazy_cache=N)`: base + first shard eager, then `set_adapter`
   loads-on-demand from disk + LRU-evicts via `delete_adapter` (never the active adapter).
   New flag `--lazy_adapter_cache N` in `eval_tofu.py` + `eval_routed_scaffold.py`.
   Memory: 13.5 GiB base + N×258 MB fp32 ≈ 16 GiB @N=8 — fits A40. TOFU eval iterates rows
   author-contiguously → ~1 load per author per pass; NFS reload overhead ~10–20 min/eval →
   03:30:00 walls. Numerics identical to eager (same fp32 cast). CPU gate:
   `test_lazy_adapters.py` (tiny random Llama, lazy(cache=2) ≡ eager bit-equal incl. a forced
   evict+reload cycle). CLAUDE.md gets the flag row + invariant note (same change).
2. **Train the 180 missing e25 adapters** — frozen recipe defaults (r32/α64/rslora/lr1e-4/
   seed42/bs4×ga4) + `--epochs 25 --k 200`, exact same command shape as job 442576.
   Array 0-199**%4** with per-task self-skip if `shard_{A}/adapter_model.safetensors` exists
   (idempotent; the 20 done skip in seconds). ~2 min/task → ~1.5 h wall at 4 GPUs.
3. **Eval wave** (smoke caps, seed 42, `--k 200 --forget_shard_id 199`; KS ref: e5 smoke
   `retain_tr_scores.npy` cp'd into e25 `results/smoke/`): 8 × 1-GPU array %4,
   `--dependency=afterany:<train>` + in-task assert that all 200 shards exist (afterok would
   hang forever on a failed train task — kill_invalid_depend is off cluster-wide):
   | # | pool | arm | how |
   |---|---|---|---|
   | 1 | e5 | oracle routed (headline conv.) | `eval_routed_scaffold.py` plain-base, no delete |
   | 2 | e5 | oracle routed + delete author 199 | same + `--delete_shard 199` |
   | 3 | e5 | lexical `routed_key_exact` (June conv.) | `eval_tofu.py` |
   | 4 | e5 | lexical `routed_key_exact_no199` | `eval_tofu.py` |
   | 5–8 | e25 | same four arms | same |
4. **Gates before submission:** G1 `python test_lazy_adapters.py` (CPU, tiny model);
   G2 `STUB=1` preview of both sbatch bodies + squeue cap check (queue EMPTY at design time);
   G3 first train task = canary (loss ~1.29 → ~0.10 by step 20, cf. 442576 shard_82 log).

## Constraints checklist (root CLAUDE.md)
- [x] GLOBAL 4-GPU cap: queue empty; ONE %4 train array + ONE %4 eval array chained via
      `--dependency` (never co-runnable); caps not raised.
- [x] SLURM only (partition all, `--exclude=sprint4`); no login-node GPU/heavy work.
- [x] Artifacts → /storage2 (180 × 250 MB ≈ 45 GB new; /storage2 at 97%, 233 GB free — fits,
      flagged to user in the final report). HF_HOME=/storage2/jack/data/huggingface.
- [x] Seed 42 everywhere; provenance (job IDs, script sha256) → log entry.
- [x] Smoke gates G1–G3 before full launch.

## State
- [x] Audit complete
- [x] Lazy-loading code + CPU gate green (`test_lazy_adapters.py` ALL PASS; sha256s:
      eval_tofu e1a1db93b7383c20, eval_routed_scaffold a33101e785130b61,
      submit_k200_routed.sh c204dbf9b6ef8889, test d9a610f772282438)
- [x] Train array submitted: **445711** (0-199%4). ⚠ Queue was saturated at submit time by
      the ctv/router_leak campaigns (445668–445694, peak exactly 4 GPUs) → 445711 is
      tail-chained `afterany:445678:445680:445684:445685` (their terminal arrays).
- [x] Eval array submitted: **445712** (0-7%4, afterany:445711 + in-task completeness assert).
- [x] Pre-registration entry written:
      log/routing_scaffold/2026-07-19_k200-oracle-routing-design.md (+ thread README
      hypotheses H-k200-1..4, log/README.md timeline row, tofu CLAUDE.md flag/tool docs).
- [x] **DONE 2026-07-20.** Train 445711 completed (200/200 shards; canary loss 1.48→0.16);
      eval 445712 all 8 arms clean (zero Tracebacks/OOM). **H-k200-1..4 ALL SUPPORTED:**
      oracle e25 routed mu **0.8236** (repo best; ret 0.999/1.000/1.05, OOD = intact base),
      e25−e5 +0.233, deletion Δmu 0.0000 (f_ppl 1.05→17.72), lexical −0.0437 on e25.
      Results entry: log/routing_scaffold/2026-07-20_k200-oracle-routing-results.md;
      thread README ledger + log/README.md updated. Campaign CLOSED; open follow-up =
      H-k200-scaf (scaffolded per-author experts), extended+multi-seed replication.

## Follow-on campaign 2026-07-20: H-k200-scaf (user-approved) — IN FLIGHT
- [x] Storage cleanup (user-approved): deleted `nmerge_r32{,_e25,_centered}/merges`
      (226 GB rebuildable; manifests/results/logs kept) → /storage2 97%, 265 GB free.
- [x] Driver `submit_k200_scaf.sh` (sha256 d02e4524a04a40d0), STUB-gated. Chain submitted:
      scaffold **446371** → bake **446372** → train **446373** (0-199%3) → eval **446374**
      (0-2%3: oracle full / del199 / scaffolded-base floor). ⚠ Concurrent session queued 5
      independent %1 GPU arrays (446357/446365/446366/446367/446370 — their own potential
      concurrency 5 > cap 4, pre-existing breach, not ours to edit); our fix: 446371
      scontrol'd afterany all five, chain runs after their queue drains, standalone peak 3 ≤ 4.
- [x] Pre-registration: log/routing_scaffold/2026-07-20_k200-scaf-design.md
      (H-scaf-k200-1 floor mediator / -2 headline mu ≥ 0.83, honest-uncertain / -3 deletion).
- [x] **CANCELLED 2026-07-20 (user request), before any task ran:** scancel
      446371/446372/446373/446374 + monitor stopped. 0 GPU-h spent; no artifacts (empty
      pool dir + KS-ref copy only). H-scaf-k200-1..3 stay open/unrun; pre-registration
      remains valid — relaunch = `bash submit_k200_scaf.sh all` (re-derive cap sum first).
- [x] **Final write-up done (2026-07-20):**
      `reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md` — self-contained (background,
      methods incl. the lazy-cache fix, provenance, full 8-arm + 9-component tables,
      H-k200-1..4 verdicts, mechanism reading incl. the retain_truth_scaled 0.546 cap
      nuance, limitations, follow-up status). Linked from the thread README (headline +
      entries) and log/README.md (timeline + by-experiment). CAMPAIGN FULLY CLOSED.

## Router-leak all-router sweep 2026-07-20 — Builder A (subagent): family audit code
Pre-reg: log/router_leak/2026-07-20_all-router-sweep-preregistration.md. New files ONLY
(no edits to existing result files or other builders' files):
- [ ] `router_family_audit.py` — score-matrix leakage audit over the router.py 9-strategy
      family + centroid_sbert_q (serving builder). Full-pool score matrices [n_q, k],
      drop-set cells (logit_div recomputed per candidate set, design note iii), family npz
      sidecar per THE FAMILY NPZ CONTRACT, per-author sentinels per feature space,
      key_exact binary match matrix (note iv), oracle by-construction block, --self_check
      faithfulness gate (matrix argmax == router.route), --stub CPU mode (tiny llama +
      synthetic pool, no HF hub).
- [ ] `test_router_family.py` — CPU gate: per-sample lora_B norm == _lora_b_norm @bs=1,
      masking invariant + survivors<1 raises, ppl sign, logit_div recompute != column mask,
      key_exact fallback + no-match flag, separable-vs-overlapping adequacy fixture,
      npz contract round-trip via the stub end-to-end, RandomState(42) sample determinism.
- [ ] `submit_router_family.sh` — j1..j6 + collect + all; STUB/DEP/self-skip; j5 afterany
      j1, j6 afterany j2 (4-GPU global cap with j1-j4 concurrent). NO submission from this
      session (login node; build + STUB preview only).
Constraints verified: outputs are new rl_family_*/rl_enc_*/rl_routerlora_* names; centroid
caches go to NEW {pool}/centroids/rfa_* dirs; k=200 pool `Llama-3.2-1B-Instruct_k200_r32_
e25_lr1e4` is NOT on disk yet (only the 7B twin) — flagged as open risk, driver mkdir -p's
only its results/router_leak dir.
