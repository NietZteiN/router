### Target Date: 2026-07-26 (7B routed-mu ladder complete — dilution is a property of MERGING, not of sharding; plus two analyzer traps)
- **Hypotheses / what we're testing:** Follow-up to
  [2026-07-26_7b-orphan-coverage.md](2026-07-26_7b-orphan-coverage.md) (append-only: that entry
  recorded k=20 only; k=4 and k=10 were still running). Two questions:
  - **H4 (dilution is merge-specific):** the 7B *merge* ladder decays monotonically with k
    (`dare_ties` 0.545 → 0.420, Table C — base + 1/k attenuation). If that decay is a property of
    **merging** rather than of sharding the data, the *routed* ladder on the **same pools** should
    show no comparable trend. CONFIRMED if routed mu is roughly flat over k=4…100; REFUTED if it
    tracks the merge decay.
  - Exploratory: does anything about the reduction tooling break when the same `k` is audited on
    more than one pool for the first time?
- **Setup:** Seed 42, A40, smoke tier. Job **448587** (`bash submit_7b_routed_fill.sh eval`, array
  `0-2%3`), completing k=4/10/20 against the pre-existing k=50/100/200 cells. Command per task:
  `eval_tofu.py --model_name meta-llama/Llama-2-7B-chat-hf --output_dir <pool> --label
  routed_key_exact --k <k> --forget_shard_id <k-1> --smoke --out <pool>/results/smoke/routed_key_exact.json`
  over `Llama-2-7B-chat-hf_k{4,10,20}_r32_e5_lr1e4`. CPU reductions afterwards:
  `analyze_orphan_destinations.py` (8 family JSONs + expanded `.npz` sidecars) and
  `analyze_router_family.py --force`.
- **Results:**
  - **7B ladder, routed `key_exact` vs merged `dare_ties` (same pools):**

    | k | 4 | 10 | 20 | 50 | 100 | 200 |
    |---|---:|---:|---:|---:|---:|---:|
    | routed | **0.7204** | 0.6907 | 0.6940 | **0.7147** | 0.6475 | 0.4728 (r8) |
    | merged | 0.545 | 0.477 | 0.450 | 0.438 | 0.430 | 0.420 (r8) |
    | gap | +0.175 | +0.214 | +0.245 | +0.277 | +0.218 | +0.053 (r8) |

    Supporting cells for the new rows: k=4 fq 0.0065 / f_rouge 0.9402 / r_ppl 1.42; k=10 mu 0.6907;
    k=20 fq 0.0 / f_rouge 0.8383 / r_ppl 1.54. All `metrics_version = ou-2026-06-10`,
    `model_name = meta-llama/Llama-2-7B-chat-hf`.
  - **Reductions:** `orphan_destinations.{md,csv}` regenerated to **111 cells / 38 routers with a
    multi-drop trajectory** (was ~46 cells), including per-author landing determinism for the four
    new batteries (e.g. 1B-plain `activation_norm` mean determinism **0.957**, `attn_norm` 0.948).
  - **`reproduce/` state:** snapshot 955 files, `--check` **955/955** match; `verify_report.py`
    **334 cells / 289 verified / 45 recorded / 0 FAIL / 0 MISSING**, exit 0.
- **What worked / hypothesis verdict:**
  - **H4 — CONFIRMED.** The routed row is flat within ±0.04 across k=4…100 (0.7204 / 0.6907 /
    0.6940 / 0.7147 / 0.6475) against a merge row that falls monotonically by 0.115 over the same
    span. The **gap widens** with granularity, +0.175 → +0.277. Routing serves one expert at full
    strength however many exist, so **dilution is a property of merging, not of sharding the data**
    — the sharper statement of the report's thesis, now measured on one model at six granularities.
    The k=200 cell is the one exception and is **capacity**, not routing: r8 adapters, because
    r32×200 exceeds a 46 GiB A40 (eval memory law). Its +0.053 gap should not be read as routing
    failing at fine granularity.
- **Observations:**
  - **Two analyzer traps hit and confirmed; both are now in `reproduce/CAVEATS.md` §13.**
    **(a) `analyze_router_family.py` keys rows by `<strategy>@k<k>` and cannot represent two pools
    at the same k.** Feeding it the 1B-scaf + 1B-plain + 7B-plain k=10 batteries rendered all three
    as `(k=10)` in `rl_family_leak_table.md` and its H-ARCH verdict deduplicated them by file mtime,
    keeping only the newest — i.e. the regenerated table silently *replaced* the 1B-scaffolded rows
    that ROUTING_MASTER cites. It does print `WARN: duplicate <strategy>@k10 entries`. **Reverted:**
    re-ran with the original one-pool-per-k input set (1B-scaf k=10 + 7B k=200 + routerlora ×3 +
    enc ×2) and diffed — `rl_family_leak_table.md` is now **bit-identical** to the 2026-07-23
    version. Multi-pool comparisons go through `analyze_orphan_destinations.py`, which is pool-keyed.
    **(b) `snapshot_results.py` with no `--ckpt-root`, run from inside `merge-tables-7b/`, resolves
    to the repo's own gitignored (empty) `tofu_sisa_lora/checkpoints`, copies 0 files and overwrites
    `MANIFEST.tsv` with an empty one** — after which `--check` trivially "passes" on nothing. It only
    ever copies, so the 947 JSONs survived; re-running with an explicit
    `--ckpt-root /home/jack/tofu_sisa_lora/checkpoints` rebuilt 955/955.
  - `--sims_glob` takes **expanded paths**, not glob patterns: quoting them yields a `[WARN] cannot
    load` per pattern and a silent `0 determinism rows`.
  - Silent-failure sweep: no NaNs; k=20 `fq` 0.0 and k=4 `fq` 0.0065 are expected for
    **pre-deletion** routed serving (every forget author is still served by its own expert, so its
    truth-ratio distribution is maximally unlike the retain90 oracle's); r_ppl 1.42–1.54 in band;
    f_rouge 0.84–0.94 consistent with correct expert selection.
- **New questions / new hypotheses:** (1) Is the k=200 routed cell really capacity-bound? Predicted
  test: an r16 k=200 pool should land between 0.4728 (r8) and the k=100 r32 0.6475 — if it does not,
  something other than adapter capacity is limiting fine-granularity routing.
  (2) The routed ladder is *non-monotone* in a small way (k=4 0.7204 > k=10 0.6907 < k=50 0.7147);
  single seed, so within-noise is the null. Worth a second seed before any claim about an optimal
  shard count.
- **Next Steps:** none blocking — Table H′, `reproduce/{LLAMA2_7B,METHODS,VENDOR_DRIFT}.md` and the
  `H7B`/`H7B-scale` cells are all updated with these numbers. The two analyzer traps are documented
  rather than fixed; fixing (a) properly means giving `analyze_router_family.py` a pool-qualified
  key, which would change every existing row label and is not worth it mid-campaign.
