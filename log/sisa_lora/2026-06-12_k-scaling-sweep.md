### Target Date: 2026-06-12 (k-scaling sweep readout — k ∈ {50,100,200})
- **Goal / Hypothesis:** Morning read-out of the overnight k-scaling sweep (k ∈ {50,100,200}, frozen recipe r32/α64/e5/lr1e-4 + k200-r1 smoke arm): does merged utility keep dilution-decaying toward base while remerge deltas vanish, and does routing-based serving hold utility where merging dilutes?
- **Setup:** Overnight chain 433769–433775 (`submit_scale_grid.sh` 8e6de3f4b778, seed 42; see 2026-06-11 night update). Auto-backup chain fired by the failing r32 memory gate: tr 434337, gate 434338, eval 434339, collect 434340. Morning fix-ups for 6 contention-killed evals (excl. sprint1): K100 434626, R8 434627, collect 434628.
- **Results:**
  - **Pipeline:** all 750 shard trainings OK (550 main + 200 r8 backup; stage-1 finished in ~85 min — high-k shards are ~60-90 s each, model-load dominated). Gate r1 PASSED (200 adapters in 6.6 min = 2.0 s/adapter; 200-way dare_ties merge 0.1 min; peak 14.1 GiB). **Gate r32 FAILED by design** → auto-cancelled r32 evals, auto-submitted r8 backup; whole fallback ran unattended (02:17–03:14). Gate r8 PASSED (peak 24.9 GiB).
  - **Memory law (validated by gate logs):** PEFT `load_adapter` casts adapters to **fp32** (`_cast_adapter_dtype`) → eval memory ≈ 13.5 GiB (7B bf16 base) + k·n_params(rank)·4 B. Measured: k=200 r1 = 14.1 GiB (pred 15.1), r8 = 24.9 (pred 26.4), r32 → ~65 GiB ⇒ **k=200 r32 eval is impossible on a 46 GiB A40 regardless of contention**; k=100 r32 ≈ 39 GiB fits only on a clean card.
  - **Incident:** 6 eval tasks (3× k100, 3× r8) OOMed on sprint1 sharing their GPU with three foreign ~7 GiB processes (PIDs 3431004-6, user bartekmare, present 00:16–03:08 — likely jobs submitted without gres). Re-submitted with `--exclude=sprint4,sprint1` (434626/434627); same-cause, no pipeline bug.
  - **Big table** (smoke, seed 42; mu/fq/f_ppl/f_rouge/f_TR/ret_prob/ret_rouge/ret_ppl/real_rouge/wld_rouge; refs from 2026-06-11 report):

| config | label | mu | fq | f_ppl | f_rouge | f_TR | ret_prob | ret_rouge | ret_ppl | real_rouge | wld_rouge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | base_model | 0.4179 | 0.239 | 15.19 | 0.393 | 0.742 | 0.164 | 0.427 | 15.67 | 0.982 | 0.933 |
| REF | k=1 ft r32e5 | 0.7435 | 0.003 | 1.27 | 0.878 | 0.446 | 0.921 | 0.863 | 1.29 | 0.865 | 0.920 |
| k=4 (KLO) | merged/remerge_dare_ties | 0.5445 / 0.5750 | 0.808 | 4.48→11.93 | | | | | | | |
| k=10 (CENTER) | merged/remerge_dare_ties | 0.4768 / 0.4791 | 0.594/0.393 | 7.82→8.41 | | | | | | | |
| k=20 (KHI) | merged/remerge_dare_ties | 0.4504 / 0.4522 | 0.594 | 10.48→10.54 | | | | | | | |
| k=50 r32 | merged_linear | 0.0433 | 0.0709 | 283.9 | 0.1768 | 0.7711 | 0.0131 | 0.1677 | 303.8 | 0.0133 | 0.0300 |
| k=50 r32 | merged_dare_ties | 0.4379 | 0.2391 | 12.2 | 0.4610 | 0.7740 | 0.1811 | 0.4086 | 12.6 | 0.9820 | 0.9400 |
| k=50 r32 | remerge_linear | 0.0443 | 0.0709 | 338.8 | 0.1612 | 0.7688 | 0.0125 | 0.1714 | 328.9 | 0.0133 | 0.0400 |
| k=50 r32 | remerge_dare_ties | 0.4380 | 0.2391 | 12.2 | 0.4644 | 0.7751 | 0.1818 | 0.4043 | 12.5 | 0.9920 | 0.9400 |
| k=50 r32 | shard_49_only | 0.4717 | 0.0065 | 1.40 | 0.7210 | 0.5208 | 0.2104 | 0.4619 | 8.30 | 0.9220 | 0.9067 |
| k=50 r32 | routed_key_exact | **0.7147** | 0.0065 | **1.40** | 0.6812 | 0.5286 | 0.7492 | 0.6688 | 1.67 | 0.9100 | 0.8867 |
| k=50 r32 | routed_key_exact_no49 | **0.7147** | 0.3929 | **8.54** | 0.4428 | 0.7458 | 0.7492 | 0.6688 | 1.67 | 0.9100 | 0.8867 |
| k=100 r32 | merged_linear | 0.0821 | 0.2391 | 46.1 | 0.2595 | 0.7810 | 0.1706 | 0.3062 | 44.0 | 0.0133 | 0.1333 |
| k=100 r32 | merged_dare_ties | 0.4299 | 0.1350 | 14.1 | 0.4142 | 0.7703 | 0.1707 | 0.4055 | 14.8 | 0.9920 | 0.9333 |
| k=100 r32 | remerge_linear | 0.1462 | 0.3929 | 35.9 | 0.2857 | 0.7820 | 0.1822 | 0.3488 | 33.8 | 0.0333 | 0.1867 |
| k=100 r32 | remerge_dare_ties | *fix 434626 pending* | | | | | | | | | |
| k=100 r32 | shard_99_only | *fix 434626 pending* | | | | | | | | | |
| k=100 r32 | routed_key_exact | **0.6475** | 0.3929 | 2.43 | 0.4644 | 0.6492 | 0.4702 | 0.5133 | 2.62 | 0.9620 | 0.8867 |
| k=100 r32 | routed_key_exact_no99 | *fix 434626 pending* | | | | | | | | | |
| k=200 r1 | merged_linear | 0.4204 | 0.1745 | 17.2 | 0.3475 | 0.7955 | 0.1609 | 0.3876 | 13.6 | 0.9820 | 0.9333 |
| k=200 r1 | merged_dare_ties | 0.4197 | 0.1745 | 17.8 | 0.3511 | 0.7957 | 0.1601 | 0.3882 | 14.1 | 0.9820 | 0.9333 |
| k=200 r1 | remerge_linear | 0.4193 | 0.1745 | 17.2 | 0.3744 | 0.7960 | 0.1609 | 0.3799 | 13.7 | 0.9820 | 0.9333 |
| k=200 r1 | remerge_dare_ties | 0.4203 | 0.1745 | 17.8 | 0.3502 | 0.7951 | 0.1601 | 0.3915 | 14.1 | 0.9820 | 0.9333 |
| k=200 r1 | shard_199_only | 0.4209 | 0.1745 | 17.5 | 0.3731 | 0.7956 | 0.1605 | 0.3890 | 14.0 | 0.9920 | 0.9333 |
| k=200 r1 | routed_key_exact | 0.4212 | 0.1745 | 17.5 | 0.3731 | 0.7955 | 0.1610 | 0.3905 | 13.9 | 0.9820 | 0.9333 |
| k=200 r1 | routed_key_exact_no199 | 0.4212 | 0.1745 | 17.6 | 0.3663 | 0.7960 | 0.1610 | 0.3905 | 13.9 | 0.9820 | 0.9333 |
| k=200 r8 | merged_linear | 0.4503 | 0.1745 | 11.2 | 0.3543 | 0.8064 | 0.2110 | 0.4073 | 9.28 | 0.9920 | 0.9200 |
| k=200 r8 | merged_dare_ties | *fix 434627 pending* | | | | | | | | | |
| k=200 r8 | remerge_linear | 0.4499 | 0.1745 | 11.2 | 0.3561 | 0.8064 | 0.2112 | 0.4063 | 9.27 | 0.9920 | 0.9200 |
| k=200 r8 | remerge_dare_ties | 0.4199 | 0.1745 | 17.7 | 0.3654 | 0.7951 | 0.1599 | 0.3882 | 14.1 | 0.9820 | 0.9333 |
| k=200 r8 | shard_199_only | 0.4398 | 0.3356 | 9.60 | 0.3495 | 0.7798 | 0.1884 | 0.3920 | 10.7 | 0.9920 | 0.9400 |
| k=200 r8 | routed_key_exact | *fix 434627 pending* | | | | | | | | | |
| k=200 r8 | routed_key_exact_no199 | *fix 434627 pending* | | | | | | | | | |
| k=200 r32 | (all 7 labels) | *blocked: eval needs ~65 GiB (fp32-cast adapters) — impossible on A40; r8 backup is the k=200 result* | | | | | | | | | |

- **Observations:**
  - **Routing is the headline.** `routed_key_exact` @k=50: **mu 0.7147** — within 0.03 of the k=1 full-data winner (0.7435) and far above every merge at any k>1 (best sharded merge ever: 0.592). And it gives the first non-trivial unlearning demo at k>4: T1 strong (f_ppl 1.40 ≪ base 15.2 — the router actually serves the memorizer), T2 pass on `_no49` (f_ppl→8.54, f_TR 0.746 ≈ base 0.742, f_rouge 0.44 ≈ base 0.39 = confabulation), T3 perfect (mu identical 0.7147 — retain/real/world untouched by exclusion, O(1) deletion by construction). @k=100 mu 0.6475 — still above the 0.6 bar.
  - **The dilution law completes:** merged dare_ties mu 0.74 (k=1) → 0.54 (4) → 0.48 (10) → 0.45 (20) → 0.44 (50) → 0.43 (100) → ≈0.42 (200) = base 0.418. At k=200 every state ≈ base — total dilution; merging is dead at high k.
  - **The bottleneck flips from merge interference to per-shard undertraining as k grows:** routed ret_prob 0.749 (k=50, 25 opt steps/shard) → 0.470 (k=100, ~12 steps) → r1@k200 is a no-op (~6 steps, rank 1: every label ≈ base). shard_{k-1}_only f_ppl: 1.40 (k=50) vs 9.60 (k=200 r8) — the dedicated 1-author adapter barely memorizes its author at e5. Routing accuracy is k-independent (0.86), so a steps-matched arm should recover most of the k=100/200 routed gap.
  - merged_linear @r1 (inflation √r=1) no longer explodes (mu 0.420 vs 0.000 at r32) and even @r8 reaches 0.4503 > dare_ties values — consistent with the rsLoRA scale-convention story: linear's failure is amplitude, not direction; with tiny deltas it's harmless (but there is also ~nothing to merge).
  - fq is non-discriminative at these forget sizes (KS quantization; pre-registered) — ppl/TR deltas are the signal. Single seed; routed-vs-merged gap (+0.28 mu) and the dilution trend are large effects, the ±0.01 method differences are not claims.
- **Next Steps:** (1) fix evals 434626/434627 + collect 434628 land the 6 missing cells (then CSV refresh). (2) **Steps-matched arm** k50 e12 / k100 e25 / k200 e50 (~62 opt steps), priority on routed labels — hypothesis: routed mu ≈ 0.7 flat in k, giving per-author O(1) deletion at full granularity. (3) Promote routing to a headline serving mode next to merging in the report/CLAUDE.md; consider `routed_centroid_sbert` (no author-name dependence) as the robustness check. (4) Extended-cap eval for k=50 routed full+_no49 (fq power). (5) Seed-variance pass before any paper-style claim.

