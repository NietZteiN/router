# METHODS — reproducing every Table H method

[`PIPELINES.md`](PIPELINES.md) covers the **merge** pools (Tables A–F′). This is its counterpart for
[Table H](../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md), the serve-time-selection
inventory: one entry per method, naming the CPU gate, the build → eval chain, the pool it writes,
and the JSON field the Table H number is read from.

Shared prerequisites (retain90 oracle + KS reference, the scaffolded base) are in
[PIPELINES.md § Order of operations](PIPELINES.md#order-of-operations) and are not repeated. Every
`submit_*.sh` takes `STUB=1` to preview and self-skips existing outputs.

**Run the CPU gate before every SLURM job.** Each row names its own; they are seconds-fast and catch
the failure modes that otherwise burn GPU-hours.

> ⚠ **GPU cap.** `CLAUDE.md` §1 sets a global 4-GPU ceiling across all queued jobs. Check
> `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"` before submitting and confirm the throttles of
> everything queued still sum to ≤ 4. Job-name collisions are the other trap: absence of a result
> JSON does **not** prove nothing is running (a duplicate array was submitted that way on
> 2026-07-24) — check `squeue -h -o '%j'` too.

## In-tree methods

| Method / Table H row | CPU gate | Build → eval | Number read from |
|---|---|---|---|
| **routing_scaffold** — oracle k=200 e25, **0.8236** | `test_routed_scaffold_merged.py` | `bash submit_k200_routed.sh train` (200-task array, e25 recipe) → `eval` (oracle-routed + lexical arms, `--lazy_adapter_cache 8`) | `_k200_r32_e25_lr1e4/results/smoke/routed_oracle_full.json:model_utility` |
| **routing_scaffold** — scaffold-routed k=10, 0.7509 | `test_routed_scaffold_merged.py` | `train_scaffold.py` → `make_scaffolded_base.py` → `train_lora_shard.py` per shard on the scaffolded base → `eval_routed_scaffold.py --shards_dir` | `_experts_scaf_k10/results/**extended**/routed_scaffold_strong.json` |
| **sisa_lora** — routed arm, 0.7147 @k50 | `test_lazy_adapters.py` (if using the cache) | pool from `submit_scale_grid.sh`, then `eval_tofu.py --label routed_key_exact --k <k> --forget_shard_id <k-1>`; 7B k=4/10/20 via `bash submit_7b_routed_fill.sh eval` | `_k{k}_r32_e5_lr1e4/results/smoke/routed_key_exact.json` |
| **legonet_lora** — 0.6371 (7B) / 0.5011 (1B) | `test_legonet_tofu.py` | `bash submit_legonet_tofu.sh <config> all` (setup → train → unlearn → eval → collect) | `_legonet_n32_k3/results/{smoke,extended}/legonet_**unlearn**.json` |
| **sift_masks** — 0.737 | `test_sift_masks.py` (the exactness gate) | `bash submit_sift_masks_tofu.sh configs/sift_masks_tofu_1b.json all` | `_sift_masks/results/smoke/sift_full.json` |
| **clamu** — 0.647 (K16) / 0.672 (K200) | `test_clamu.py` | `bash submit_clamu_tofu.sh configs/clamu_tofu_1b.json all` | `_clamu/results/smoke/clamu_full.json` |
| **peft_compose** — IA³ routed, 0.5155 | `test_compose_peft.py` | `bash submit_peft_bakeoff.sh configs/peft_bakeoff_1b.json all` | `_peft_ia3_k10/results/smoke/routed_key_exact.json` |
| **s3t** — 0.581 `[F(d)]` | `test_s3t.py`, `test_s3t_sequences.py` | `bash submit_s3t_repro.sh` (depth dirs → F-eval → deltime → finalize) | `_s3t_m5_L4_armB/F_curve.json` — a **curve**, not `model_utility` |
| **merge baseline** (the foil every row is measured against) | `test_merge_extra.py` | see [PIPELINES.md](PIPELINES.md) | `results/smoke/merged_*.json` |

⚠ **`submit_peft_bakeoff.sh` hardcodes the 1B pool** in its compose and eval stages — only *train*
honors `--config`. Pointing it at a 7B config silently composes and evaluates the 1B pool. This is
why the 7B PEFT bake-off exists only on the CISPA A100 box and is *recorded, never verified* here.
Check the rendered body with `STUB=1` before trusting any driver on a new pool.

## Out-of-tree methods (`external/`)

Vendored code + configs + reports; their weight stores were `/storage2` symlinks and are excluded.
Each has its own README. All four are **Llama-3.2-1B only** — there are no 7B results in any of
them.

| Method / Table H row | Home | Note |
|---|---|---|
| **memsinks** 0.6417 | [`../external/memsinks_tofu/`](../external/memsinks_tofu/) | Also a **hard import dependency** of `eval_tofu.py --memsinks_config`. Its Group-B serve run (H-GB3) was never executed — the Table H orphan verdict for this row is *predicted from mechanism*. |
| **sepmlp** ≈0.795 | [`../external/sepmlp_tofu/`](../external/sepmlp_tofu/) | A **recall** metric, not `model_utility`; not comparable to the mu column. |
| **sea** 0.711 | [`../external/sea_tofu/`](../external/sea_tofu/) | 4-bit Llama-2-7B. Routing settled but no end-to-end serve — Group-A verdict from mechanism. |
| **memory_adapters** 0.869 `[OU]` | [`../external/memadapt_tofu/`](../external/memadapt_tofu/) | OU chat-template `Agg` scale — a different metric from `model_utility`. **Never mix the two tracks.** |

## Orphan-behavior batteries

Utility and orphan behavior are separate runs. The orphan side is `router_family_audit.py`, driven
by `submit_router_family.sh`:

| Stage | Pool | Strategies | Output |
|---|---|---|---|
| `j1` / `j2` | 1B **scaffolded** k=10 | feature / behavioral | `_experts_scaf_k10/results/router_leak/rl_family_k10_{feature,behavioral}.json` |
| `j3` | 7B k=200 e25 | feature (memory law) | `_k200_r32_e25_lr1e4/results/router_leak/rl_family_k200.json` |
| `j7` / `j8` | **7B plain k=10** | feature / behavioral | `_k10_r32_e5_lr1e4/results/router_leak/rl_family_k10_7b_{feature,behavioral}.json` |
| `j9` | **7B plain k=50** | feature | `_k50_r32_e5_lr1e4/results/router_leak/rl_family_k50_7b.json` |
| `j10` / `j11` | **1B plain k=10** | feature / behavioral | `Llama-3.2-1B-Instruct/results/router_leak/rl_family_k10_1b_plain_{feature,behavioral}.json` |

```bash
python test_router_family.py                      # CPU gate — always first
python analyze_router_family.py --self_test       # 15/15 PASS
bash submit_router_family.sh sevenb               # j7+j8+j9  (3 GPUs)
bash submit_router_family.sh deconfound           # j10+j11   (2 GPUs)
bash submit_router_family.sh collect              # CPU reduction
```

Three rules that decide whether a battery is even runnable:

1. **Behavioral routers load the pool.** `ppl` / `activation_norm` / `attn_norm` / `logit_div` score
   *through* the experts, so `router_family_audit.py` refuses them above k=50 (eval memory law:
   13.5 GiB base + k·228 MB/adapter at r32; k=200×r32 ≈ 65 GiB > a 46 GiB A40). Feature/lexical
   routers load **no** adapters, which is why k=200 is feature-only.
2. **Never disable `--self_check`.** It asserts the score matrix's argmax reproduces
   `router.route()` per strategy. Every battery here reports 50/50.
3. **Pool provenance is part of the comparison.** The 1B k=10 battery in `j1`/`j2` is the
   *scaffolded* pool; `j7`/`j8` is the *plain* 7B pool. Comparing them conflates model scale with
   scaffolding for every LLM-reading router — `j10`/`j11` (plain 1B) exist to separate the two. See
   [LLAMA2_7B.md §3.3](LLAMA2_7B.md#33-is-the-magnet-a-property-of-the-model-or-of-the-embedding-space).

CPU reductions, no GPU:

```bash
# pool-keyed — handles many pools at once, labels every row with its pool dir
python analyze_orphan_destinations.py --family_json <all audit JSONs> \
    --sims_glob <expanded .npz paths> --out_md reports/orphan_destinations.md \
    --out_csv reports/orphan_destinations.csv

# strategy-keyed — ONE POOL PER k (see the warning below)
python analyze_router_family.py --family_json ... --family_npz ... --routerlora_json ... \
    --enc_json ... --out_json ... --out_md reports/rl_family_leak_table.md --force
```

⚠ **`analyze_router_family.py` keys rows by `<strategy>@k<k>` and cannot distinguish two pools at
the same `k`.** Passing the 1B-scaffolded, 1B-plain and 7B-plain k=10 batteries together renders all
three as `(k=10)` — indistinguishable — and its H-ARCH verdict silently deduplicates them by file
mtime, keeping only the newest. It does print `WARN: duplicate <strategy>@k10 entries`; that warning
is load-bearing. Feed it **one pool per k** (the canonical set is 1B-scaffolded k=10 + 7B k=200 +
`--routerlora_json` + `--enc_json`, which reproduces `rl_family_leak_table.md` bit-for-bit), and use
`analyze_orphan_destinations.py` for any multi-pool comparison.

⚠ `--sims_glob` takes **expanded paths**, not glob patterns — quote them and it silently reports
`0 determinism rows` after a `[WARN] cannot load` line per pattern.
