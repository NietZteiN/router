### Target Date: 2026-06-15 (Joint-Diagonalization LoRA compression + selective-keep unlearning — Compress-then-Serve port; separate thread from the S³T entry above)
- **Goal / Hypothesis:** Implement the *Compress-then-Serve* JD method (Gabrielsson et al., 2025)
  as collection-agnostic infra, and wire it to our unlearning so "choose which adapters to keep"
  is an O(1) Σ_i add/drop. JD compresses a collection per module slot into a shared basis U,V
  (per cluster) + per-adapter Σ_i; the kept subset combines into one delta
  Σ_j U_j(Σ_i w_i norm_i Σ_i)V_jᵀ, rank-compressed to the scaffold. Validate on TOFU shards first;
  design for hundreds/thousands of adapters (the fp32 high-k memory wall is what JD removes).
- **Setup:** New `jd_compress.py` (pure-torch core: JD-Full alternating dominant-subspace iter,
  JD-Diag coordinate descent, k-means clustering, `JDCompressed.merge_keepset`/`reconstruction_error`/
  `select_num_clusters`); `jd_collection.py` (mode B: reads adapter safetensors on CPU, builds/saves
  a compressed artifact under /storage2, materializes a kept-subset merge as a PEFT adapter dir;
  `build`/`select`/`merge` CLI). Wired into `merge_extra.jd_merge_adapters` (mode A, in-model) and
  `merge_lora` (`jd_full`/`jd_diag` in MERGE_METHODS + EXPERIMENTAL tier; `_split_jd_suffix` parses
  `_c{N}`/`_r{R}`; labels `merged_/remerge_jd_{full,diag}[_cN][_rR]`; JD excluded from tree merges).
  True-scale method (divides scaffold scaling out, like knots/tsv/subtract_orth). Subspace step:
  thin-SVD when nr≤d else Gram-SVD (avoids an MKL SSYEVD eigh bug on large symmetric matrices);
  zero-pads bases to uniform rank r. Not a git repo (no commit hashes). Env test-env.
- **Results:** `test_merge_extra.py` green incl. new `test_jd` — lossless@full-rank recon 2.1e-4
  (Prop. 1), Diag≥Full recon, O(1)-deletion identity (drop Σ_f ≡ refit), identical-shards JD merge
  == single delta 1.3e-6 (confirms true scale). `test_ou_equivalence.py` still green (metrics
  untouched). Mode-B on the 10 real TinyLlama shards: build c=1/r=16 in 122s, **recon_err 0.62**
  (near the paper's 0.6/99%-perf threshold even WITHOUT clustering — trained LoRAs share structure,
  as the paper predicts; clustering will drop it further), save/load exact, materialized adapter
  reproduces the merge to 1e-7 at uniform rank 8.
- **Observations:** JD is conceptually distinct from the existing registry (it does not fuse
  adapters during compression — each stays addressable via Σ_i), which is exactly what makes
  selective keep/delete O(1). Full clustered build over all 154 modules on CPU is minutes-scale
  (timed out at 300s for c=4) → belongs on SLURM; `select_num_clusters` uses one probe slot for
  cheap tuning. recon_err 0.62 at c=1 on real shards is the encouraging headline.
- **Next Steps:** run the main experiment (plan file) — utility vs forget_quality across
  k∈{1,4,10,50,100,200}, with `{slug}_ft` (k=1) as the 1-LoRA baseline and the retain90 oracle as
  the forgetting ceiling; `merged_jd_*` vs `remerge_jd_*` (selective-keep unlearn) vs
  `remerge_dare_ties`/`subtract_orth`. Submit smoke eval of `remerge_jd_full_c4` via
  `submit_eval_smoke.sh` (SLURM, not login node). Secondary: c/r sweep → recon-error↔(utility,
  forget_quality) tradeoff; mode-B build at k≥100 to show flat memory.
- **Update (fidelity pass vs the authoritative technical reference):** audited every method
  detail against the paper. Already-faithful: UΣVᵀ factorization, JD-Full (Σ=UᵀDV, orthonormal
  basis, alternating M/N), JD-Diag (Hadamard + r×r systems), 10 iters, per-adapter unit-norm
  normalization + restore, k=n≡r-SVD, Prop-1 lossless, <0.6/middle-module/rank-16 cluster
  heuristic, true-scale write-back. Fixed 4 gaps: (1) **clustering now iterates assign↔refit to
  convergence** (Appendix A.3 Step2→Step1, max_rounds cap) — was a single reassignment round;
  (2) dropped the artificial JD-rank cap at the scaffold rank (paper §6.5 uses (n/2)+7, capped
  only by matrix dims); (3) select_num_clusters candidates → paper schedule (1,2,4,7,8,10,16,25);
  (4) added recommend_jd_settings(n) (§6.5/§F). New tests green: cluster recon non-increasing
  (tiny 0.836→0.691→0.548; real TinyLlama subset 0.536→0.454→0.297), deterministic. Explicitly
  out of scope per the doc: the vLLM/Punica serving kernels (§5/§10) and the A.2 SVD-free
  eigenvalue-iteration variant (we use A.1 @ 10 iters, the paper's main-experiment setting).

