### Target Date: 2026-07-02 (routing+scaffold: reproduce the core claim from committed code)
- **Hypotheses / what we're testing:** H1: routed isolated experts + a public (Alpaca) scaffold reach
  ≥ full-FT model_utility (claimed 0.664 > full-FT 0.599). CONFIRM = routed+scaffold ≥ full-FT; REFUTE =
  below. H2: deletion is exact/O(1) — dropping a shard forgets exactly its authors, nothing else. The
  scaffold trainer did not previously exist, so the 0.664 had never been reproduced from committed code.
- **Setup:** Llama-3.2-1B, smoke, seed 42. New code: `train_scaffold.py` (LoRA on 2k public Alpaca,
  `skill_data.load_alpaca`, non-rslora r16/α32/3ep), `make_scaffolded_base.py` (merge_and_unload scaffold
  into base → full model at `/storage2/.../Llama-3.2-1B-Instruct_scaffolded_alpaca2k`), `eval_routed_scaffold.py`
  (OOD-aware router: TOFU-author query via exact `q2author`→shard expert, OOD→scaffold-only; `--delete_shard`
  = exact-deletion demo). Experts = existing SISA k=10 shards (`checkpoints/Llama-3.2-1B-Instruct/shard_0..9`,
  legacy r8/α16). Baselines: full-FT `_ft/shard_0`, scaffold-floor (`--preloaded_adapter` scaffold),
  routing-only (`routed_key_exact`). Jobs 440166 (scaffold train), 440168 (matrix), 440188 (OOD-aware),
  440202 (deletion). Report: `../../tofu_sisa_lora/reports/ROUTING_SCAFFOLD_REPRO_2026-07-01.md`.
- **Results:** model_utility — routing-only 0.458, scaffold-floor 0.404, **routed+scaffold (key_exact) 0.474**,
  full-FT 0.530. Component decomposition: scaffold-floor real_prob 0.631/world_prob 0.656, but
  routed+scaffold (key_exact) real 0.370/world 0.495 — the expert corrupts OOD answers. **OOD-aware fix:
  routed+scaffold mu 0.556** (real 0.630, world 0.656 restored; route_stats routed 990 / ood 1208).
  Exact-deletion (`--delete_shard 9`): forget_quality 0.135→**0.393**, forget_rouge 0.532→0.465,
  **model_utility 0.5559→0.5559 identical**, retain_prob 0.304→0.304.
- **What worked / hypothesis verdict:** **H1 SUPPORTED but 0.664 REFUTED** — the committed-code number is
  0.474 (< full-FT), and the *corrected* (OOD-aware) method is **0.556 > full-FT 0.530** — beats full-FT,
  modestly (+0.026), not by 0.07. **H2 SUPPORTED** — deletion is exact/O(1): dropped authors fall to the
  scaffold-floor forget_quality (0.393 = never-trained) with all other authors byte-unaffected (mu identical).
- **Observations:** The key bug: `routed_key_exact` (name-substring) routes *every* query — incl.
  real_authors/world_facts — to a TOFU expert (name-free → shard-0 fallback), and that expert's delta
  corrupts the scaffold's general-knowledge answer. OOD-aware routing (author→expert, else scaffold-only)
  fixes it and is what makes the scaffold pay off. retain_prob (0.304) trails full-FT (0.390) because the
  k=10 experts are the weak r8/α16 legacy recipe — the margin should widen with stronger experts.
- **New questions / new hypotheses:** Do r16/more-epoch experts push retain toward full-FT (→ mu ~0.62,
  the estimate)? Does the result hold at extended caps and vs a properly-trained full-FT? Can a
  *training-free* router make the author↔OOD decision without a learned (leaky) gate?
- **Next Steps:** (1) stronger experts + extended-cap eval + fair full-FT baseline; (2) fold the OOD-aware
  routing into the legonet arm (it currently routes OOD to nearest cluster, not scaffold); (3) exactness
  audit of the whole served pipeline (scaffold is public → clean; router is training-free → clean).
