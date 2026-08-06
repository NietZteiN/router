### Target Date: 2026-07-08 (LoRA parameter-space t-SNE figure — the HydraLoRA Fig-1(b) analog on TOFU)
- **Hypotheses / what we're testing:** User asked for a WMDP-style "chaos in the LoRA
  parameter space" figure (HydraLoRA Fig 1(b): each point = one LoRA's parameters,
  t-SNE-projected, colored by data category) for TOFU.
  **HT1 (chaos):** the t-SNE of the 200 per-author delta cosines shows NO forget/retain
  separation and NO semantic clustering — predicted from the near-uniform off-diagonal
  cosines of the k=200 matrix (mean 0.0012, range [0.0006, 0.0026]; 07-07 entry found no
  block structure). CONFIRM = silhouette on the precomputed 1−cos distances < 0.05 for
  every labeling AND a visually mixed scatter stable across perplexities. REFUTE =
  silhouette ≥ 0.1 or a coherent forget/semantic cluster.
- **Setup:** Pure CPU post-processing of an existing report JSON — no training, no GPU, no
  SLURM (same class as `author_similarity_report.py`).
  - Input: `reports/subspace_overlap_k200_r32.json` (job 440863, 2026-07-07; 200×200
    factored Frobenius cosine of ΔW=scaling·B·A over the `_k200_r32_e5_lr1e4` per-author
    shards, sort -V ⇒ matrix index == author id) + semantic labels from k-means (K=6,
    seed 42, size-ordered relabel) over the frozen legonet MiniLM author embeddings
    `checkpoints/Llama-2-7B-chat-hf_legonet_n32_k3/legonet/author_emb.npy` (200×384).
  - New script `tofu_sisa_lora/plot_author_tsne.py` (+ CPU gate
    `test_plot_author_tsne.py`, 6/6 green): distance = clip(1−cos,0) symmetrized, sklearn
    `TSNE(metric='precomputed', init='random', random_state=42)`, perplexity sweep
    {5,15,30,50}, main figure at 30. **Base anaconda python** (matplotlib/sklearn 1.7.2).
  - Command: `/home/jack/anaconda3/bin/python plot_author_tsne.py --json
    reports/subspace_overlap_k200_r32.json --author_emb
    checkpoints/Llama-2-7B-chat-hf_legonet_n32_k3/legonet/author_emb.npy --out_dir
    reports/figures/lora_tsne` (seed 42; smoke on the n=8 `_smoke8` JSON first).
- **Results:** Silhouette on the precomputed distances: forget-binary **−0.0000**,
  forget-4-class **−0.0001**, semantic k-means K=6 **−0.0000**. Visual: forget authors
  (180–199, ordinal blue ramp) scatter uniformly through the 180 gray retain points; the
  6 semantic clusters are fully interleaved; the mixing is unchanged across perplexities
  5/15/30/50. Off-diag cosine restated in the sidecar: mean 0.0012, range [0.0006,
  0.0026]. Artifacts: `reports/figures/lora_tsne/tsne_k200_r32_{forget,semantic}.{png,pdf}`,
  `_perplexity_sweep.png`, `_coords.csv`, `_meta.json`.
- **What worked / hypothesis verdict:** **HT1 SUPPORTED** — silhouettes ≈ 0 (−0.0001…−0.0000,
  all < 0.05) for every labeling and the scatter is visually structureless at every
  perplexity. TOFU per-author LoRA space replicates the WMDP "chaos" picture: neither
  forget membership nor semantic category is recoverable from full-delta geometry.
- **Observations:** (1) This is the figure-ready face of the 07-07 finding — the deltas
  are near-orthogonal as whole vectors, so ANY partition looks like chaos; the real
  structure (shared col(B) output subspace, 92× chance energy) lives in subspace angles,
  which full-delta cosine t-SNE cannot see. Don't present the figure as "no shared
  structure exists" — caption it as "no author/category clustering". (2) The name-token
  effect (0.0014 vs 0.0012, p≈5e-4) is ~15% relative and invisible at t-SNE scale, as
  expected. (3) Silent-failure checks: seed-fixed rerun reproduces coords bit-identically
  (gate test); n=8 smoke exercised the perplexity clamp; no NaNs; silhouette guard
  prevents over-reading t-SNE shapes. (4) For the paper narrative this motivates
  structural separation (SISA/routing): the forget set is NOT a weight-space-identifiable
  region you could excise post-hoc — you have to build the boundary in at training time.
- **New questions / new hypotheses:** (a) **H-tsne-colB (open):** a t-SNE over col(B)
  principal-angle distances (the collision-carrying geometry) WOULD show structure —
  needs a pairwise-angle matrix dump (subspace_overlap stores only means; ~200×200 QR/SVD
  pass, SLURM CPU). (b) Does the r8 family (`_k200_r8`, cosine matrix not yet computed)
  look identical, tying into H-rank? (c) Reuse: overlaying scaffolded vs plain expert
  collections in one embedding as an H-scaf visual.
- **Next Steps:** Fold `tsne_k200_r32_forget` into the merge-mechanism report as the
  motivation figure; decide whether H-tsne-colB is worth the pairwise-angle SLURM pass
  after the H-scaf control lands.
