# Llama-2-7B — mu and orphan behavior, per method

The master report's [Table H](../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md)
inventories **methods**, not a single model, so its rows mix Llama-3.2-1B, Llama-2-7B, 3B, TinyLlama
and phi-2 — and a row's *orphan* numbers do not always come from the same model as its *mu*. This
page pins the model to **`meta-llama/Llama-2-7B-chat-hf`** and asks: what do we actually know, in
numbers, and where is the number checkable?

Every cell below is either (a) recomputable from [`results_snapshot/`](results_snapshot/) with the
Python standard library, or (b) explicitly marked as recorded-from-elsewhere. Verify the whole page:

```bash
python reproduce/verify_report.py --table H7B -v
```

## The coverage answer, first

Table H has **12 method rows**. At Llama-2-7B:

| | count | rows |
|---|---:|---|
| 7B `model_utility`, **verifiable from the snapshot** | **3** | routing_scaffold k=200 · sisa_lora routed · legonet |
| 7B `model_utility`, **recorded only** (other cluster) | 1 | peft_compose (IA³) |
| 7B number on a **different scale** (not mu) | 2 | sea (4-bit) · s3t (`[F(d)]` curve) |
| **no 7B run at all** | **6** | routing_scaffold k=10 · sift_masks · clamu · memsinks · memory_adapters · sepmlp |

And **7B orphan behavior existed for exactly one pool** (k=200) until 2026-07-26, when three more
batteries were added — see [§3](#3-orphan-behavior-at-7b).

## 1. Utility (mu) at 7B

| Method | mu | source | note |
|---|---:|---|---|
| routing_scaffold (oracle, k=200 e25) | **0.8236** | `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4/results/smoke/routed_oracle_full.json:model_utility` | repo best. Deletion of author 199 leaves mu **bit-unchanged** (`routed_oracle_del199.json` = 0.8236) |
| sisa_lora routed (`key_exact`) @k4 | 0.7204 | `..._k4_r32_e5_lr1e4/...` | filled 2026-07-26 |
| sisa_lora routed @k10 | 0.6907 | `..._k10_r32_e5_lr1e4/...` | filled 2026-07-26 |
| sisa_lora routed @k20 | 0.6940 | `..._k20_r32_e5_lr1e4/...` | filled 2026-07-26 |
| sisa_lora routed @k50 | **0.7147** | `Llama-2-7B-chat-hf_k50_r32_e5_lr1e4/results/smoke/routed_key_exact.json` | |
| sisa_lora routed @k100 | 0.6475 | `..._k100_r32_e5_lr1e4/...` | |
| sisa_lora routed @k200 (r8) | 0.4728 | `..._k200_r8_e5_lr1e4/...` | **r8-capacity-limited, not a routing failure** — r32×200 exceeds a 46 GiB A40 |
| legonet (post-deletion) | **0.6371** | `Llama-2-7B-chat-hf_legonet_n32_k3/results/smoke/legonet_unlearn.json` | Table H quotes the *unlearn* label |
| legonet (pre-deletion) | 0.6277 | `.../legonet_full.json` | deletion *raises* mu here |
| peft_compose IA³ routed | 0.6473 ⚠ | `checkpoints_7b/..._peft_ia3_k10/results/smoke/routed_key_exact.json` | **CISPA A100 box — not in this snapshot.** composed 0.4900; dora 0.5709/0.4985; vera 0.4972/0.4719; prefix 0.0348 |
| sea | 0.711 ⚠ | `external/sea_tofu/reports/SEA_UNLEARNING_REPORT.md` | **4-bit** — not comparable to the column |
| s3t | 0.581 ⚠ | `Llama-2-7B-chat-hf_s3t_m5_L4_armB/F_curve.json` | `[F(d)]` curve, **not** `model_utility` |

**Anchors for this model:** base **0.426**, FT (r32, all authors) **0.756**
(`Llama-2-7B-chat-hf_nmerge_r32/results/smoke/{base_model,ft_r32}__own82.json`).

**The contrast that matters.** On the *same model*, the full 7B k=200 merge battery — ~20 operators,
Table C′ — spans **0.419–0.451**, i.e. base + dilution. Routing spans **0.637–0.824**. The
merge-vs-select gap is not an artifact of the 1B pools.

**Merge-vs-route at six granularities, one model** (routed `key_exact` vs merged `dare_ties`):

| k | 4 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| routed | **0.7204** | 0.6907 | 0.6940 | **0.7147** | 0.6475 | 0.4728 (r8) |
| merged | 0.545 | 0.477 | 0.450 | 0.438 | 0.430 | 0.420 (r8) |
| gap | +0.175 | +0.214 | +0.245 | +0.277 | +0.218 | +0.053 (r8) |

The merge row decays monotonically with k — the dilution law. The routed row does **not**: it holds
at 0.65–0.72 across k=4…100 with no trend, because routing serves one expert at full strength no
matter how many exist. So the gap *widens* with granularity until k=200, where both fall for a
**capacity** reason (r8 adapters; r32×200 exceeds a 46 GiB A40), not a routing one. Dilution is a
property of merging, not of sharding.

## 2. Why some rows have no 7B number

Not an oversight; each has a specific cause worth knowing before you try to fill it.

| Method | Blocker |
|---|---|
| routing_scaffold k=10 (1B: 0.7509) | No scaffolded 7B base exists — `make_scaffolded_base.py` was only ever run for 1B. Needs a 7B scaffold LoRA → baked base → 10 re-trained experts. |
| sift_masks (1B: 0.737) | 200-task **fp32 full-FT** at 7B. The single most expensive missing cell in the table. |
| clamu (1B: 0.647) | Same shape as SIFT, plus per-cluster STE mask optimization. |
| memsinks (1B: 0.6417) | Out-of-tree (`external/memsinks_tofu`), 1B only — and its Group-B serve run (H-GB3) was never executed at *any* scale. |
| memory_adapters (1B: 0.869 `[OU]`) | Out-of-tree, 1B only, and on the OU chat-template scale — a 7B run still would not be mu-comparable. |
| sepmlp (1B: ≈0.795) | Out-of-tree, 1B only, and a **recall** metric rather than `model_utility`. |

Verified, so you needn't re-check: `external/{memsinks_tofu,memadapt_tofu,sepmlp_tofu}` contain no
7B results of any kind.

## 3. Orphan behavior at 7B

An **orphan** is one of the deleted authors' questions after its expert/mask/proxy is dropped. The
question is where it goes and whether landing there discloses anything.

### 3.1 The k=200 per-author pool — the original 7B battery

`Llama-2-7B-chat-hf_k200_r32_e25_lr1e4/results/router_leak/rl_family_k200.json`. Two drop sets:
**d199** (1 author, 20 orphans, 199 survivors) and **d180–199** (forget10, 400 orphans, 180
survivors). `n_eff` = 1/HHI = effective number of destinations. `adequacy` = masked/unmasked top-1
similarity ratio (≈1 ⇒ the wrong sibling matches as well as the deleted expert did).

| Router | drop | top1 | top3 | n_eff | busiest | adequacy | retain shift | conf AUC |
|---|---|---:|---:|---:|---|---:|---:|---|
| `key_exact` | d199 | 1.000 | 1.000 | 1.0 | s0 (fallback) | — | 0.0000 | no-match op: orphan 1.000 / retain 0.148 |
| `key_exact` | d180–199 | 1.000 | 1.000 | 1.0 | s0 | — | 0.0000 | orphan 1.000 / retain 0.147 |
| `key_tfidf` | d199 | 0.400 | 0.750 | 4.0 | s88 (8/20) | **0.194** | 0.0000 | **0.999** |
| `key_tfidf` | d180–199 | 0.190 | 0.297 | 17.5 | s88 (76/400) | 0.270 | 0.0003 | **0.989** |
| `centroid_sbert` | d199 | 0.400 | 0.900 | 3.3 | s88 (8/20) | 0.705 | 0.0000 | **0.982** |
| `centroid_sbert` | d180–199 | 0.110 | 0.258 | 24.2 | s88 (44/400) | 0.673 | 0.0008 | **0.984** |
| `centroid_lm` | d199 | **0.700** | 0.900 | **1.9** | s128 (14/20) | **0.976** | 0.0000 | 0.728 |
| `centroid_lm` | d180–199 | 0.170 | 0.310 | 17.4 | s88 (68/400) | 0.962 | 0.0042 | 0.761 |
| `q2author` (oracle) | both | — | — | — | → base | — | 0.0000 | — |

Oracle routing sends **100.0%** of orphans to the base and shifts **0.0%** of retain routes, both
drop sets (`oracle.cells.*.orphan_base_capture` = 1.0) — that is the "0 leak" cell of Table H, and it
is an argument from the routing rule, not a measurement.

**Two readings.** (i) At single-author deletion the leak/detect split is sharp: `centroid_lm`
concentrates (n_eff 1.9) with adequacy 0.976 and is confidence-undetectable (AUC 0.728), while
`key_tfidf`/`centroid_sbert` self-detect almost perfectly (0.982–0.999) precisely *because* their
wrong match is a bad match (adequacy 0.194 / 0.705). (ii) **Concentration is a property of how much
you delete.** Going from 1 to 20 deleted authors spreads every router (n_eff 1.9→17.4, 3.3→24.2,
4.0→17.5): no single survivor can absorb 400 orphans. Do not quote a magnet number without its drop
count.

### 3.2 The 7B granularity dial — k=10 / k=50 / k=200 (added 2026-07-26)

`rl_family_k10_7b_feature.json` + `rl_family_k10_7b_behavioral.json` (k=10, all 9 strategies, drop
sets d9 / d9,8 / d9,8,7,6) and `rl_family_k50_7b.json` (k=50, drop sets d49 / d49,48). Both pools
also carry the routed mu in §1, so utility and orphan behavior come off one artifact.

Single-shard drop, same model, three granularities:

| Router | k=10 (400 orphans, 9 surv.) | k=50 (80 orphans, 49 surv.) | k=200 (20 orphans, 199 surv.) |
|---|---|---|---|
| `key_exact` | n_eff 1.0 · max 1.00 · s0 | n_eff 1.0 · max 1.00 · s0 | n_eff 1.0 · max 1.00 · s0 |
| `key_tfidf` | n_eff 6.6 · max 0.22 · s4 | n_eff 9.6 · max 0.23 · s1 | n_eff 4.0 · max 0.40 · s88 |
| `centroid_sbert` | n_eff 7.2 · max 0.18 · s7 | n_eff 8.2 · max 0.19 · s22 | n_eff 3.3 · max 0.40 · s88 |
| `centroid_lm` | n_eff 2.5 · max 0.59 · s4 | n_eff 7.3 · max 0.24 · s13 | n_eff 1.9 · max 0.70 · s128 |
| **adequacy** `key_tfidf` | **0.667** | **0.385** | **0.194** |
| **adequacy** `centroid_sbert` | **0.967** | **0.884** | **0.705** |
| **adequacy** `centroid_lm` | **0.997** | **0.997** | **0.976** |

⚠ **`n_eff` is not comparable across this row.** Dropping one shard drops a different number of
authors at each k, so the orphan count falls 400 → 80 → 20 while survivors rise 9 → 49 → 199, and
n_eff is bounded by both. The non-monotone n_eff (2.5 → 7.3 → 1.9) is mostly that bound moving, not
a routing effect. **Read the adequacy rows instead** — adequacy is a similarity *ratio*, so it is
scale-free:

> As units get finer, a wrong match becomes a genuinely *worse* match — `key_tfidf` 0.667 → 0.385 →
> 0.194, `centroid_sbert` 0.967 → 0.884 → 0.705, both monotone. **`centroid_lm` is the exception:
> pinned at 0.976–0.997 at every granularity.** That is exactly why the k=200 `key_tfidf` /
> `centroid_sbert` routers self-detect their orphans (AUC 0.989–0.999) while `centroid_lm` cannot
> (0.728): a router matching on generic LM-hidden-state similarity never notices that the expert it
> wanted is gone, however fine you slice the pool.

### 3.3 Is the magnet a property of the model or of the embedding space?

Three arms at k=10, drop shard 9 (400 orphans, 9 survivors), identical author assignment and
queries. **1B-plain vs 7B-plain isolates model scale; 1B-plain vs 1B-scaffolded isolates the
scaffold.** (`Llama-3.2-1B-Instruct/results/router_leak/rl_family_k10_1b_plain_*.json`.)

| Router | reads the LLM? | n_eff 1B-scaf / 1B-plain / 7B-plain | magnet | adequacy |
|---|---|---|---|---|
| `key_exact` | no | 1.0 / 1.0 / 1.0 | s0 / s0 / s0 | — |
| `key_tfidf` | no | 6.6 / 6.6 / 6.6 | s4 / s4 / s4 | 0.667 / 0.667 / 0.667 |
| `centroid_sbert` | no (MiniLM) | 7.2 / 7.2 / 7.2 | s7 / s7 / s7 | 0.967 / 0.967 / 0.967 |
| `centroid_sbert_q` | no (MiniLM) | 7.7 / 7.7 / 7.7 | s7 / s7 / s7 | 0.971 / 0.971 / 0.971 |
| `centroid_lm` | yes | 2.1 / 2.5 / 2.5 | s4 / s4 / s4 | 0.999 / 0.999 / 0.997 |
| `centroid_lm_last` | yes | 1.7 / 2.6 / 1.5 | s4 / s4 / **s2** | 0.998 / 0.998 / 1.000 |
| `ppl` | yes (experts) | 7.0 / 6.7 / 6.4 | s7 / s7 / **s1** | 0.377 / 0.527 / 0.346 |
| `activation_norm` | yes (experts) | 1.4 / 1.1 / **2.4** | s6 / s6 / **s7** | 0.997 / 1.000 / 0.959 |
| `attn_norm` | yes (experts) | 3.7 / 1.1 / 1.1 | **s3 / s5 / s4** | 1.000 / 1.000 / 0.995 |
| `logit_div` | yes (experts) | 5.5 / 3.6 / 2.9 | **s0 / s8 / s7** | 0.953 / 0.936 / 0.838 |

Three findings:

1. **The four routers that never read the model are bit-identical across all three arms** — same
   n_eff, same magnet shard, same adequacy to three decimals. This is forced (identical inputs,
   identical encoder) and is therefore a strong end-to-end check that the harness is sound. Their
   magnet is a property of the **TOFU author-embedding geometry alone**, not of any model.
2. **Every router that reads model internals moves its magnet with scale** — `centroid_lm_last`
   s4→s2, `ppl` s7→s1, `activation_norm` s6→s7 (and n_eff 1.1→2.4), `attn_norm` s5→s4,
   `logit_div` s8→s7. Two of them (`attn_norm`, `logit_div`) also move with *scaffolding* at fixed
   model, so pool provenance matters as much as scale for this family.
3. **`centroid_lm` — the worst leaker — is the stable one.** Magnet s4 in all three arms, adequacy
   ≥0.997 throughout. The router that is simultaneously most concentrated and least detectable is
   *not* a quirk of one checkpoint; it reproduces across a 6× scale change and a scaffold swap.
   That is the finding with the most weight for the thesis, and it argues the leak is structural.

### 3.4 What is **not** measured at 7B

- **legonet** — its mu is 7B, but the two-magnet `n_eff 6.1` (e5/e11/e30) figure everyone quotes is
  the **1B** n=32 pool. There is no 7B legonet routing audit.
- **sea** — mu is 4-bit 7B; its routing is borrowed from the **1B** SIFT per-author centroid
  measurement (n_eff 27.1, magnet author 88), and the Group-A leak verdict is stated from mechanism.
  No end-to-end 7B serve was ever run.
- **s3t / sepmlp** — no router exists, so orphan routing is undefined (not "clean").
- **AUC / tombstone columns** need the `.npz` score sidecars, which are deliberately **not**
  snapshotted (~18 MB). They are recorded with their recompute command in
  [`cells.tsv`](cells.tsv); the sidecars sit beside each audit JSON in the checkpoint store.

## 4. Reproducing any of it

Per-method build → eval chains are in [`METHODS.md`](METHODS.md). The two commands behind this page:

```bash
# utility cell (any 7B pool, k = pool size, forget shard = k-1)
python eval_tofu.py --model_name meta-llama/Llama-2-7B-chat-hf \
  --output_dir <pool> --label routed_key_exact --k <k> --forget_shard_id <k-1> \
  --smoke --out <pool>/results/smoke/routed_key_exact.json

# orphan battery (feature/lexical routers load NO adapters, so the 7B memory law does not bite)
python router_family_audit.py --pool_dir <pool> --base_model meta-llama/Llama-2-7B-chat-hf \
  --k <k> --strategies key_exact key_tfidf centroid_sbert centroid_lm \
  --drop_sets "<k-1>;<k-1>,<k-2>" --queries all --device cuda --dump_sims \
  --out <pool>/results/router_leak/<name>.json
```

Both are wired into drivers: `bash submit_7b_routed_fill.sh eval` and
`bash submit_router_family.sh sevenb` (`STUB=1` previews; both self-skip existing outputs).

⚠ **Behavioral routers** (`ppl`, `activation_norm`, `attn_norm`, `logit_div`) score *through* the
experts, so they load the pool: `router_family_audit.py` refuses them above k=50 by the eval memory
law (13.5 GiB base + k·228 MB/adapter at r32). At k=10 on 7B that is ~16 GiB — fine on an A40.

⚠ **The 1B-vs-7B comparison has a pool-provenance trap.** The 1B k=10 battery in the repo
(`_experts_scaf_k10`) is the **scaffolded** pool; the 7B k=10 battery is the **plain** pool. For
`key_*`/`centroid_sbert*` this is harmless — they never touch the LLM, and the two pools' numbers
come out bit-identical, which is a good end-to-end check — but for `centroid_lm*` and the behavioral
family, scale and scaffolding are entangled. The plain-1B arm
(`Llama-3.2-1B-Instruct/results/router_leak/rl_family_k10_1b_plain_*.json`,
`bash submit_router_family.sh deconfound`) exists to separate them.
