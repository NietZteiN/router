# CLAUDE_SCRATCHPAD — memadapt_tofu

## State (2026-07-15 evening) — first full loop CLOSED
Full narrative: `log/memory_adapters/` (3 entries). All stages S0–S6 done.

| Row (ours vs paper) | Util.R | Util.G | Mem | Priv | Agg |
|---|---|---|---|---|---|
| Finetuned | 1.000/1.00 | 1.143/1.14 | 0.088/0.07 | 0.381/0.38 | 0.252/0.21 |
| Retrained | 1.009/1.00 | 1.121/1.11 | 0.590/0.58 | 1.000/1.00 | 0.874/0.87 |
| MemAdapt FT | 1.075/0.93 | 1.024/1.05 | 0.027/0.18 | 0.380/0.39 | 0.097/0.40 |
| MemAdapt unlearned | 1.066/1.00 | 1.013/1.06 | **0.630/0.62** | 0.917/0.98 | **0.869/0.87** |

- G1 PASS (anchors ≤ ±0.011). G2: Agg |Δ|=0.001 + Mem ✓, but Priv/Util.R/Util.G
  outside ±0.03; H4 refuted here (ΔUtil.R −0.010). One mechanism: near-uniform
  frozen router (softmax entropy ≈ ln 32, cross-source read mass 0.10).
- H5 confirmed: unlearn = 0.027 s CPU (5,120 entries), apply 0.021 s.
- Checkpoint: `checkpoints/memadapt_1b_l8_s42` (+ blocklists/forget10.json).
  Evals: `checkpoints/evals/{calib_*,memadapt_ft,memadapt_unlearned}/`.
- OU branch `memadapt-eval` carries 3 env fixes (hub cache_dir; datasets_ou301
  isolated cache — prebuilt on login node; fp32-logits restore in get_model —
  required for BOTH crash fix and reference parity).

## Next (in order)
1. **H6 router-temperature ablation**: add `key_scale` (or top-k softmax
   temperature) knob to `memory_layer.py`; re-profile→assign→train at ~×2/×4
   sharpness (arrays %2, ~40 min each). Prediction: Priv ↑, FT Util.R ↓,
   ΔUtil.R > 0.
2. Seeds 43/44 on the winning config before headline claims.
3. S8 baselines RMU/GradDiff/NPO (independent of H6; eval-then-delete policy
   pre-approved).
4. Report both composition variants (priv_absdiff read 0.953).

## Standing constraints
- No git commits (user rule 2026-07-15; provenance = meta.json sha256s).
- Queue: check `squeue -u jack` before every submit; global 4-GPU cap; %2 arrays.
- Other threads (merge_mechanism) share the cluster — expect their jobs in queue.
