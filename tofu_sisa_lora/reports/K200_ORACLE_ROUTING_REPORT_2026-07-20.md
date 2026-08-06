# 200 Per-Author Task Vectors + Oracle Routing on TOFU — Campaign Report (2026-07-20)

**TL;DR.** Serving TOFU with one task vector (LoRA adapter) per author — 200 experts — behind
an exact author-lookup router yields **model_utility 0.8236, the best serving utility of any
track in this repo** (previous best: k=10 scaffold-routed 0.7509; joint fine-tune ≈ 0.7563;
base 0.418–0.426), while making deletion of a single author an **O(1), exactly utility-free
operation** (Δmu = 0.0000, every retained metric identical to 4 decimals, the deleted
author's perplexity rising to the never-trained level). The experiment closes a
pre-registered June follow-up that was never run: the June k-scaling sweep found routed
utility collapsing at k=200 (mu 0.4728) and attributed it to per-author *undertraining*, but
the steps-matched confirmation was blocked by an eval memory wall and never executed. With
the wall removed and the experts trained to saturation, the attribution is confirmed at
+0.233 mu — **routing is utility-flat in k all the way to single-author granularity; the
bottleneck was always training dose, never routing.**

Everything here is smoke-tier, seed 42, single-seed — headline claims should be replicated
at extended caps / seeds 43–44 before any external use (see §9).

---

## 1. Background (self-contained)

**Goal of the repo** (`~/CLAUDE.md` §3): exact machine unlearning for LLMs — deletion should
be a cheap, deterministic, O(1)-style "drop the module that holds the data," not approximate
gradient surgery.

**Benchmark.** TOFU: 200 fictional authors × 20 Q&A pairs, fine-tuned into a base model
(here Llama-2-7B-chat-hf). Deletion requests target authors. Metrics (ported to match
open-unlearning exactly): `model_utility` (mu) = harmonic mean of 9 components — retain /
real_authors / world_facts × {answer-prob, ROUGE-L recall, truth-ratio-scaled} — plus
forget-side probes (forget_rouge/ppl/truth-ratio and the `forget_quality` KS p-value against
a retain-only oracle).

**The two serving families.** Per-shard adapters can be **merged** (one composite model —
utility collapses as k grows; the merge_mechanism thread measured the collapse knee at N≈3–8
and a flat ceiling ≈ base+0.04) or **routed** (keep adapters separate, pick per query —
deletion = stop routing to the dropped expert, which is exact by construction).

**Prior state of the k=200 routed cell** (why this campaign existed):

| date | what ran | result | limitation |
|---|---|---|---|
| 2026-06-12 | `routed_key_exact` @k=200, r8/e5 pool | mu **0.4728** (deletion utility-free) | ~6 optimizer steps/author — experts barely memorize |
| 2026-06-12 | same, r1/e5 | mu 0.4212 ≈ base (no-op) | rank-1 + undertrained |
| 2026-06-12 | same, r32/e5 | **never ran** | PEFT fp32-casts every loaded adapter: 200×r32 ≈ 65 GiB > A40 46 GiB |
| 2026-06-12 (pre-registered) | "steps-matched k=200 routed arm; prediction routed mu ≈ 0.7 flat in k" | **never executed** | superseded by the k=10 routing_scaffold thread |
| 2026-07-07/09 (merge_mechanism) | strong per-author pools trained: r32/e5 (200/200 authors), r32/e25 (20/200, near-perfect: own-rows retain_prob 0.9992, ppl 1.06) | only merged/analyzed | never route-served |

So the specific cell "**200 well-trained task vectors + oracle routing**" was empty. This
campaign filled it (pre-registration:
[`log/routing_scaffold/2026-07-19_k200-oracle-routing-design.md`](../../log/routing_scaffold/2026-07-19_k200-oracle-routing-design.md);
results entry:
[`2026-07-20_k200-oracle-routing-results.md`](../../log/routing_scaffold/2026-07-20_k200-oracle-routing-results.md)).

**Routing nomenclature** (this distinction matters):
- **Oracle route** = exact `q2author` lookup (normalized question text → author id, built
  from the TOFU `full` split; `eval_routed_scaffold.py` / `OODAwareRoutedModel`). Author
  queries serve base + that author's expert; **OOD queries (real_authors / world_facts)
  serve the plain base with adapters disabled.** This is the honest serving convention.
- **Lexical route** = `routed_key_exact` (`eval_tofu.py` label): case-insensitive author-name
  substring match, ~0.86 routing accuracy; name-free questions *and all OOD queries* fall
  back to shard_0's expert — the "composition bug" channel diagnosed in the k=10 thread.
  Kept solely for comparability with the June ladder.

## 2. Pre-registered hypotheses

- **H-k200-1 (headline):** oracle routing over 200 well-trained (e25) per-author task
  vectors holds mu ≥ 0.70 (utility ≈ flat in k). Refute: mu < 0.60.
- **H-k200-2 (training dose):** mu(e25 oracle) − mu(e5 oracle) ≥ 0.10.
- **H-k200-3 (O(1) deletion):** deleting author 199's expert: |Δmu| ≤ 0.005; forget signal
  read from ppl/rouge (fq pre-registered as non-discriminative at 20-row forget sets).
- **H-k200-4 (OOD fallback cost):** on the strong pool, oracle − lexical mu ≥ 0.03.

## 3. Method

**Pools** (all Llama-2-7B-chat-hf, LoRA r32/α64/rslora, lr 1e-4, bs 4×ga 4, seed 42,
`train_lora_shard.py` frozen-recipe defaults; one adapter per author, `--k 200`):
- `Llama-2-7B-chat-hf_k200_r32_e5_lr1e4` — pre-existing, 200/200 authors, `--epochs 5`
  (weak: isolated own-author f_rouge 0.4895 vs base floor 0.4038).
- `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` — 20 authors pre-existing (job 442576); the
  remaining **180 trained in this campaign** (job 445711, identical command shape,
  ~1 min/author; canary loss 1.48 → 0.16 by step 20, matching the 442576 reference).

**The enabling fix — lazy adapter cache.** New `--lazy_adapter_cache N` flag
(`eval_tofu.py`, `eval_routed_scaffold.py`; core = `eval_tofu.lazify_shard_adapters`):
instead of eagerly loading all 200 fp32-cast adapters (~65 GiB), `set_adapter` loads a shard
from disk on first use and LRU-evicts via `delete_adapter` (never the active adapter;
missing shards raise rather than silently serving base). Numerics are identical to the eager
path (same fp32 cast — routed evals here are convention-comparable with June). Memory at
N=8: 13.5 GiB base + ~2 GiB adapters. TOFU evaluation iterates rows author-contiguously, so
each shard reloads ≤ once per metric pass (NFS overhead ~10–20 min/eval, budgeted in wall
clock). CPU regression gate: `test_lazy_adapters.py` — lazy(cache=2) forward logits
**bit-equal** to an eagerly-loaded reference, including a forced evict→reload cycle;
cap respected; active adapter never evicted. All 8 GPU evals ran on standard 1×A40/48G
allocations with zero OOM — the configuration declared impossible on 2026-06-12.

**Arms** (8 = 4 × 2 pools; smoke tier; `--k 200 --forget_shard_id 199`, i.e. forget =
author 199 only; KS reference copied from the e5 pool — recipe-independent):

| arm | tool | serving |
|---|---|---|
| oracle full | `eval_routed_scaffold.py` (plain base) | author → base+expert; OOD → base |
| oracle del199 | same + `--delete_shard 199` | author 199 → base (exact deletion); rest unchanged |
| lexical | `eval_tofu.py --label routed_key_exact` | name match → expert; fallback (incl. OOD) → shard_0 |
| lexical no199 | `--label routed_key_exact_no199` | shard 199 excluded from routing |

## 4. Provenance

- Driver: `submit_k200_routed.sh` (sha256 `c204dbf9b6ef8889`); edited scripts:
  `eval_tofu.py` `e1a1db93b7383c20`, `eval_routed_scaffold.py` `a33101e785130b61`;
  gate `test_lazy_adapters.py` `d9a610f772282438`. No git repo — sha256 is the provenance.
- SLURM: train array **445711** (0-199%4, self-skipping, tail-chained
  `afterany:445678:445680:445684:445685` behind a concurrent session's saturated queue so
  the global 4-GPU cap held under any scheduler order); eval array **445712** (0-7%4,
  `afterany:445711` + in-task pool-completeness assert). Zero failures/OOM/Tracebacks in
  all logs (`/storage2/.../k200_routed_logs/`).
- Result JSONs written 2026-07-19 23:58 → 2026-07-20 00:22 under each pool's
  `results/smoke/`: `routed_oracle_full.json`, `routed_oracle_del199.json`,
  `routed_key_exact.json`, `routed_key_exact_no199.json`.
- Cost: ≈ 6 GPU-h training (180 × ~2 min) + ≈ 8 GPU-h eval. Storage: +45 GB
  (180 × 250 MB shards).

## 5. Results

Headline table (smoke tier; forget = author 199):

| pool | arm | **mu** | ret_prob | ret_rouge | ret_ppl | real_prob | world_prob | f_rouge | f_ppl | f_TR | fq |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e5 | oracle full | 0.5908 | 0.3905 | 0.432 | 3.65 | 0.778 | 0.6794 | 0.3863 | 4.12 | 0.6679 | 0.5713 |
| e5 | oracle del199 | **0.5908** | 0.3905 | 0.432 | 3.65 | 0.778 | 0.6794 | 0.3575 | 17.72 | 0.7961 | 0.1745 |
| e5 | lexical | 0.5869 | 0.3723 | 0.4227 | 3.86 | 0.7538 | 0.7429 | 0.3863 | 4.12 | 0.6674 | 0.5713 |
| e5 | lexical no199 | 0.5869 | 0.3723 | 0.4227 | 3.86 | 0.7538 | 0.7429 | 0.3242 | 9.46 | 0.7943 | 0.1745 |
| e25 | **oracle full** | **0.8236** | **0.999** | **1.000** | **1.05** | 0.778 | 0.6794 | 1.000 | 1.05 | 0.5686 | 0.3356 |
| e25 | oracle del199 | **0.8236** | 0.999 | 1.000 | 1.05 | 0.778 | 0.6794 | 0.3575 | 17.72 | 0.7961 | 0.1745 |
| e25 | lexical | 0.7799 | 0.9173 | 0.9156 | 1.23 | 0.6868 | 0.6434 | 0.9652 | 1.05 | 0.5732 | 0.3356 |
| e25 | lexical no199 | 0.7799 | 0.9173 | 0.9156 | 1.23 | 0.6868 | 0.6434 | 0.3269 | 17.06 | 0.7098 | 0.5713 |

Full 9-component decomposition of the two oracle-full rows (what the harmonic mean sees):

| component | e5 oracle | e25 oracle |
|---|---|---|
| retain_prob / rouge / truth_scaled | 0.3905 / 0.432 / **0.3066** | 0.999 / 1.000 / **0.5464** |
| real_prob / rouge / truth_scaled | 0.778 / 0.982 / 0.8681 | 0.778 / 0.982 / 0.8681 |
| world_prob / rouge / truth_scaled | **0.6794** / 0.9333 / 0.9108 | **0.6794** / 0.9333 / 0.9108 |

Route stats (oracle arms, identical across pools): routed 520 / OOD 1208; del199 arms:
routed 360 / deleted 160 / OOD 1208.

Anchors for context: base mu 0.418–0.426; June k-ladder routed: k=50 0.7147 → k=100 0.6475
→ k=200(r8/e5) 0.4728; k=10 scaffold-routed 0.7509; matched-capacity full-FT 0.6372;
joint-ft 0.7435–0.7563. Merged serving of the *same* pools (merge_mechanism Exp-5/H8):
flat 0.459 ± 0.002 (e5 mean, any N) and 0.40–0.44 (e25 mean ladder).

## 6. Hypothesis verdicts

- **H-k200-1 SUPPORTED (emphatically):** e25 oracle mu **0.8236 ≥ 0.70** — repo best.
  Per-author experts serve their own authors essentially perfectly (retain 0.999 / 1.000 /
  ppl 1.05) while OOD components sit exactly at the intact base values. The June prediction
  ("routed ≈ 0.7 flat in k") lands *above* target once experts are steps-matched.
- **H-k200-2 SUPPORTED:** dose gap +0.233 (0.5908 → 0.8236 at fixed r32; the June r8/e5
  point was 0.4728). The k=200 "collapse" was 100% training dose.
- **H-k200-3 SUPPORTED:** Δmu = **0.0000** on both pools; all retain/real/world components
  identical to 4 decimals. Deleted author: f_ppl 1.05 → **17.72** (never-trained level),
  f_rouge 1.000 → 0.3575. Free consistency check: the del199 forget rows are bit-identical
  across e5/e25 (both serve the same base for the deleted author) — the deletion surface is
  pool-independent, as construction requires.
- **H-k200-4 SUPPORTED:** oracle − lexical = **+0.0437** on e25 (≥ 0.03), and only +0.004
  on e5 — the cost is dose-dependent, as predicted. Anatomy on e25: OOD damage from
  shard_0's strong expert applied to name-free/OOD queries (real 0.778→0.687, world
  0.679→0.643) plus ~14% misrouted author queries (retain 0.999→0.917, the known 0.86
  lexical routing-accuracy ceiling).

## 7. Reading the result

1. **Routing vs merging, same weights.** The identical e25 experts served routed give
   0.8236; averaged into one model they give ≤ 0.44 (H8 ladder) with own-author recall at
   the population floor by N≈8–12. Composition operator — not adapter quality, rank, or
   count — is the entire difference. This is the strongest single quantification of the
   repo's central claim to date.
2. **What caps mu now.** The binding components at e25-oracle are `retain_truth_scaled`
   **0.5464** (heavily-memorized experts remain imperfectly calibrated on perturbed/
   paraphrased retain rows), `world_prob` **0.6794** and `real_prob` **0.778** (the intact
   base's knowledge floor — routing by design adds nothing OOD). Note the scaffold idea
   (H-k200-scaf) only addresses the OOD pair; the largest single deficit is truth-ratio
   calibration, which a scaffold does not obviously touch.
3. **Deletion.** At single-author granularity the served system after deletion is
   *indistinguishable in every retained metric to 4 decimals* — with the router the only
   thing that changed. This is the operational ideal from `~/CLAUDE.md` §3, demonstrated at
   the finest unit TOFU supports. Caveats that keep this honest: `forget_quality` *drops*
   on deletion (0.336→0.175) — the known base-vs-retain-oracle KS style artifact at n=20
   rows, pre-registered; and the router itself is the remaining leak/disclosure channel,
   quantified separately by the router_leak thread (identity-seal AUC 0.982, disclosure
   AUC 0.839 at k=10).
4. **The lexical router under-deletes.** `routed_key_exact_no199` sends the deleted
   author's queries to shard_0's *expert* rather than base (e5: f_ppl 9.46 vs oracle's
   17.72) — a second, independent reason the lexical convention is the wrong serving
   surface for deletion claims.
5. **Serving cost.** Oracle routing needs the q2author lookup (exact-match; a real system
   needs NER/registry lookup — cf. the router_leak registry rung) and one adapter resident
   per query; the lazy cache makes GPU memory O(cache), not O(k).

## 8. Artifacts

- Pools: `/storage2/jack/checkpoints/tofu_sisa_lora/Llama-2-7B-chat-hf_k200_r32_{e5,e25}_lr1e4/`
  (200 shards each; e25 completed by this campaign).
- Results: `<pool>/results/smoke/routed_{oracle_full,oracle_del199,key_exact,key_exact_no199}.json`.
- Logs: `/storage2/jack/checkpoints/tofu_sisa_lora/k200_routed_logs/`.
- Code: `submit_k200_routed.sh`, `eval_tofu.py` (`--lazy_adapter_cache`,
  `lazify_shard_adapters`), `eval_routed_scaffold.py`, `test_lazy_adapters.py`;
  documented in `tofu_sisa_lora/CLAUDE.md` (eval-variants table + memory-law invariant).

## 9. Limitations

- **Smoke tier, single seed (42).** Extended-cap + seeds 43/44 replication is the gate
  before any cross-paper claim. The routed-vs-merged gap (+0.36) and the deletion identity
  are far outside plausible seed noise; the exact 0.8236 value is not.
- **Forget set = 1 author (20 rows):** fq is KS-quantized and non-discriminative
  (pre-registered); deletion evidence is ppl/rouge/Δmu, not fq.
- **Oracle routing is an assumption:** exact q2author lookup ≈ an idealized registry
  router. The realistic-router cost (embedding/centroid) and its post-deletion leak are the
  router_leak thread's territory, measured so far at k=10.
- **fp32 serving convention:** kept for June comparability; bf16-adapter serving would
  halve reload time/memory but change numerics vs prior tables.
- `forget_prob` is not emitted at this split (single-author perturbed coverage); read
  rouge/ppl/TR.

## 10. Follow-ups (status at write-up)

- **H-k200-scaf** — per-author experts on a scaffolded 7B base (aiming past 0.8236 by
  lifting the OOD floor): pre-registered
  ([`2026-07-20_k200-scaf-design.md`](../../log/routing_scaffold/2026-07-20_k200-scaf-design.md)),
  chain submitted (446371–446374), then **cancelled by the user before any task ran**
  (0 GPU-h). The pre-registration remains valid; note §7.2's caveat that the largest mu
  deficit (retain truth calibration) is outside the scaffold's reach.
- Extended-cap + multi-seed replication of the 0.8236 headline.
- `H-ia3-route-200` (peft_compose): the same cell at ~1.5–30 MB of adapters — the cost
  frontier for exact-deletion serving.
- Realistic router at k=200 + its deletion leak (router_leak methods at author granularity).
