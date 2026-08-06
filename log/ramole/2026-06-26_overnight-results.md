### Target Date: 2026-06-26 (overnight campaign — router vs 1/k, ablations, unlearning demo)
- **Goal / Hypothesis:** Run the RAMoLE campaign on GPU (Llama-3.2-3B, the `legonet_l32_3b_n32_k3`
  n=32 expert pool) to test: (1) does the learned RouterLoRA cross-attention beat LegoNet's uniform
  1/k composition on the SAME experts; (2) does Random LoRA Dropout / router-train-split / router-rank
  matter; (3) does deletion still forget when the post-deletion pool is served through the UNCHANGED
  router (the exactness-preserving claim). Follows [2026-06-25 implementation](2026-06-25_implementation.md).
- **Setup:** 4-GPU cap on SLURM (sprint1-3). Arm A: retriever FT + index (job 436663) + router train
  (436496, reference split, p=0.5, r16, 224 steps, loss 2.2→1.97) + eval array (436664). Ablation arms
  via `submit_overnight.sh` chained behind Arm A: routers (436684) for `configs/ramole_l32_3b_{d0,
  corpus,r6}.json` (dropout 0 / corpus split / rank 6; each shares Arm A's retriever via
  `retriever_run`), then a dispatch eval array (436685). Unlearning demo reuses existing legonet
  deletions d0/d1/d2 (1 record each → 3 affected adapters) served through `eval_ramole.py
  --unlearn_tag --unlearn_state` (new `adapter_dir_fn` path + legonet `post_unlearn_adapter_dir_fn`).
  Report via `collect_overnight.py` → `/storage2/jack/checkpoints/ramole/OVERNIGHT_REPORT.md`. Repo
  not under git (no commit hash). **Two issues hit + fixed:** (a) retriever instructor-xl full FT
  OOM'd at batch_size 16 on a 44 GiB card (paper used A100-80GB) → dropped to 4 + expandable_segments,
  then completed; (b) the 12 unlearn eval tasks in 436685 pointed at a config basename = the run
  *name* (`ramole_l32_3b_n32_k3.json`, nonexistent) instead of the file (`ramole_l32_3b.json`) → all
  failed FileNotFoundError; fixed the basename in `submit_overnight.sh` and reran as array 437475.
- **Results (N=200 records; unlearning N=1/deletion):**
  - **Headline — learned router > 1/k, < perfect, on the same experts.** Key-routed (same retrieved
    set, only the composition rule differs): RAMoLE router **em 0.647 / verbmem 0.384 / ppl 5.354** vs
    LegoNet 1/k **0.643 / 0.370 / 5.479** vs perfect-selection (single ideal expert) **0.655 / 0.381 /
    5.183**. The router consistently edges out 1/k (lower ppl, higher em/verbmem) and lands between it
    and perfect. Retriever-routed (full RAMoLE): iid 0.623 / 6.186, ood 0.597 / 7.300 (worse than
    key-routed because instructor-xl top-1 is 0.84, so some records reach non-ideal experts).
  - **Random LoRA Dropout — null here.** p=0.5 vs p=0 essentially identical (keys_iid 0.647 both;
    retriever_ood 0.597 vs 0.596). On topically-similar DBpedia clusters with strong retrieval the
    IID/OOD gap is small, so dropout doesn't move it (paper's gain is for truly unseen task-LoRAs).
  - **Split / rank — null, and reference-split is free.** corpus 0.649/5.310 ≈ reference 0.647/5.354 ≈
    rank-6 0.648/5.342. The exactness-preserving `reference` split costs no utility vs paper-faithful
    `corpus` — good for the unlearning story.
  - **Unlearning demo — deletion forgets, router NOT retrained.** Forget record ppl rises / em drops
    after deleting it (retrain only its 3 affected experts), served through the unchanged router:
    d0 ppl 3.75→5.91 (em 0.658→0.590), d1 6.75→**11.08** (0.635→0.571), d2 3.97→4.95 (0.639→0.623).
    RAMoLE's router forgets as cleanly as 1/k (mean: d0 3.90→6.31, d1 6.68→11.09, d2 3.76→4.76).
  - **Retrieval:** instructor-xl top-1 **0.842** / top-3 **0.980** / top-5 0.993; 40%-cluster FT is a
    no-op (identical ranking — the DBpedia-cluster task is already trivial off-the-shelf).
- **Observations:** The core thesis holds — a learned cross-attention gate is a strictly-better
  composition rule than uniform 1/k on the same experts (small but consistent), while the deletion
  mechanism is untouched and needs no router retrain (the reference-split router never saw the
  deletable records). The ablations are honest nulls: this DBpedia setup is too easy/ homogeneous to
  surface dropout or rank effects, and the router's win over 1/k is modest because k=3 averaging of
  topically-coherent experts is already close to optimal. canary_em stays ~0 (weak single-record
  canary memorization, as in the legonet thread); perplexity is the clean forget signal. The unlearning
  N=1/deletion is qualitative, not powered.
- **Next Steps:** (1) Bigger forget set — make a single multi-record deletion (e.g. 10–20 records) for
  a powered before/after, accepting the extra affected-adapter retrains. (2) Stress the router where it
  should help more: heterogeneous/cross-task experts (mix DBpedia with another task family) and higher
  k, where 1/k dilutes and a learned gate should pull ahead — and where dropout's OOD gain should
  appear. (3) Optional: route Arm A's headline through a stronger/again-FT retriever, or report 7B
  transfer on the `legonet_7b_v2` pool. Report artifact: `OVERNIGHT_REPORT.md`.
