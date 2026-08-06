# SEA — Separable Expert Architecture (per-author deletable adapters) on TOFU

**Status:** complete · **Project:** [`sea_tofu/`](../../sea_tofu/) · **Entries:** 3 (2026-06-09 → 2026-06-20)

SEA keeps a frozen 4-bit Llama-2-7B-chat base and isolates each TOFU author's knowledge inside its own per-author LoRA "proxy" adapter (one author = one deletable artifact). Unlearning is then a filesystem operation: deleting an author's proxy reverts the model to the base on that author's data, with no weight surgery and no touching of any other author. The study probes the rank-vs-deletability tradeoff — bigger LoRA rank means deeper personalization but a larger (less "tiny-deletable") artifact (the "deletability tax").

It landed as a clean structural-unlearning result: deletion is a ms `rm`, recall saturates by rank 8, contamination is near-zero, and unlearned-state == retrain-gold == base on the forget set exactly, with model utility unchanged through unlearning by construction.

## What worked
- **Rank knee is sharp and seed-robust.** Extended eval (max_new=128): proxy ROUGE-L r4 **0.673** → r8 **0.991** → r16 **1.000** (base 0.420). Seed variance over {42,43,44} at the knee: r4 ROUGE **0.668±0.006** / Prob **0.699±0.007**, r8 ROUGE **0.992±0.001** / Prob **0.986±0.000** — std ≈ 0.006, so the effect is real, not noise (satisfies CLAUDE.md §4).
- **Recall saturates by r8**, then flat r16–r64 (sweep Prob: r4 0.707, r8 0.986, r16 0.999, r32 1.000, r64 0.999) — the deletability tax lands exactly where the paper predicts; r4 underfits, the knee is r4→r8.
- **Isolation / contamination near zero.** Pilot (authors 180–184, r16): contamination = **0.0** on all 3 probe pairs (no adapter leak). Sweep (noisier, only 3 probe Qs @ 40 tok): max contam **0.06–0.11** across ranks.
- **Deletion = clean unlearning.** Canonical TOFU report @ r16 (job 436005): deleting forget proxies drops Forget ROUGE **1.0→0.403** and Forget Prob **0.999→0.161** (= base), Forget Quality **0.0→1.0**, while **Model Utility stays 0.711 unchanged** — utility preserved through unlearning by construction. Unlearned == Retrain-gold exactly.
- **Forget quality = 1.0** (construction-trivial: base-only candidate == base-only gold), and pilot deletion gate confirmed omission == post-deletion (kl=0.0).
- **Utility in the TOFU-7B range:** model_utility 0.711 (report) / ~0.78 (smoke) — vs OU finetuned 0.63, locuslab ft 0.748.

## What didn't / open problems
- **The "tiny deletable artifact" story breaks under the tax.** Proxy size doubles per rank (r4 16 MB → r8 32 → r16 64 → r32 128 → r64 256 MB, fp32 LoRA on q/k/v/o × 32 layers), all ≫ the paper's 2–5 MB. The only rank near "tiny" (r4) is the one that underfits — the central tension.
- **First per-rank eval jobs all hit SLURM time limits** (each reloaded the 7B base + recomputed constant real/world + ROUGE); fixed by `run_sweep_eval.py` (base loaded once, base-side shared, max_new capped, incremental writes).
- **40-token generation cap depressed absolute ROUGE** (~0.87 not 1.0 in the first sweep); resolved at max_new=128 (→1.0 by r8–r16). The *relative* rank curve was the robust result throughout.
- **Forget quality is construction-trivial**, so it carries no signal — the science is the rank/size tradeoff, isolation, and deletion cost, not the forget-quality number.
- **Extended eval timed out at the 5h wall after r16**; r32/r64 covered only by the earlier (saturated) smoke run.

## Open ideas / next steps
- **GA/NPO baselines** via `tofu_sisa_lora` for deletion-cost Table B (SEA = ms `rm` vs GPU-minute weight surgery) — deferred (separate lift, busy cluster).
- Full retain-utility pass and max_new≥128 for absolute ROUGE on all ranks (r32/r64 still only from smoke).
- Optional: store LoRA in bf16 to halve proxy size and partly relieve the deletability tax.

## Entries (chronological)
- [2026-06-09 — implementation](2026-06-09_implementation.md) — SEA paper stood up as `sea/` on synthetic personas
- [2026-06-18 — tofu-proxy](2026-06-18_tofu-proxy.md) — per-author proxies on TOFU; pilot all gates green
- [2026-06-20 — rank-sweep](2026-06-20_rank-sweep.md) — rank knee r4→r8, seed-robust, clean unlearning report

## Full reports
- [REPORT.md](../../sea_tofu/REPORT.md) — top-level SEA-on-TOFU project report
- [SEA_UNLEARNING_REPORT.md](../../sea_tofu/reports/SEA_UNLEARNING_REPORT.md) — canonical TOFU unlearning report (3 states: Original / Unlearned / Retrain-gold @ r16)
