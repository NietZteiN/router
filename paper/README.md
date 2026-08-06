# paper/ — which code produces which claim

This repo is the evidence base for **Router-Free Modular Storage for Knowledge Unlearning in Large
Language Models** (MUSR), an AAAI submission.

> **The manuscript itself is not in this repo.** It is an anonymized submission under review,
> carrying *"Distribution, citation, or public sharing of this manuscript is strictly
> prohibited."* Neither the PDFs nor the LaTeX sources are committed here, and
> [`../.gitignore`](../.gitignore) keeps them out. This page is a map from each claim to the code,
> report and result file that produces it — the numbers below are all reproduced in
> `../tofu_sisa_lora/reports/` and `../log/`, which are the primary records.

The paper's method is **MUSR** — per-source gated FFN adapters in parallel to the frozen MLP,
`MLP(x) + Σₖ φₖ(x)`, with the foreign-output penalty `L_out = Σ_{j≠k} ‖φⱼ(x)‖²` at `λ_out = 10`,
and deletion = zero `W°ₖ`. In this repo that is **[`sepmlp_tofu/`](../sepmlp_tofu/)**
(`sepmlp_model.py`, `bank_layer.py`, `train_sepmlp.py`), with
**[`blocktc_tofu/`](../blocktc_tofu/)** as its 11.8×-smaller block-transcoder successor.

## Appendix D — model merging

| claim | repo |
|---|---|
| 12 merge operators at N=200, 7B r8; all within 0.04 of base 0.426, joint-ft 0.756 | [`merge_lora.py`](../tofu_sisa_lora/merge_lora.py), [`merge_extra.py`](../tofu_sisa_lora/merge_extra.py) · [`MERGE_METHODS_RESULTS_2026-07-21.md`](../tofu_sisa_lora/reports/MERGE_METHODS_RESULTS_2026-07-21.md) · [`merge_tables_7b/RESULTS_TABLES.md`](../merge_tables_7b/RESULTS_TABLES.md) |
| merged decays 0.545 → 0.420 over k=4…200 while routed stays flat | [`log/router_leak/2026-07-26_7b-routed-ladder-and-tooling.md`](../log/router_leak/2026-07-26_7b-routed-ladder-and-tooling.md) (H4-7B) · [`MERGE_VS_ROUTING_MASTER_2026-07-24.md`](../tofu_sisa_lora/reports/MERGE_VS_ROUTING_MASTER_2026-07-24.md) |
| shared write direction — mean pairwise cosine 0.00125, ≈23,000× the random-matrix floor; dependence entirely output-side | [`subspace_overlap.py`](../tofu_sisa_lora/subspace_overlap.py), [`author_similarity_report.py`](../tofu_sisa_lora/author_similarity_report.py) · [`AUTHOR_SIMILARITY_K200_2026-07-07.md`](../tofu_sisa_lora/reports/AUTHOR_SIMILARITY_K200_2026-07-07.md) · [`log/merge_mechanism/`](../log/merge_mechanism/) (27 entries) |
| selection over identical summed weights: sum 0.407 / global 0.351 / EMR 0.388 / TALL 0.405 / optimized K=200 0.672 / SIFT 0.737, base 0.398 | [`clamu.py`](../tofu_sisa_lora/clamu.py), [`clamu_model.py`](../tofu_sisa_lora/clamu_model.py), [`sift_masks*.py`](../tofu_sisa_lora/) · [`CLAMU_REPORT_2026-07-02.md`](../tofu_sisa_lora/reports/CLAMU_REPORT_2026-07-02.md) · [`log/clamu/`](../log/clamu/), [`log/sift_masks/`](../log/sift_masks/) |
| DoRA / IA³ / VeRA / LoRA composed vs routed — the ceiling is weight-space composition, not LoRA | [`compose_peft.py`](../tofu_sisa_lora/compose_peft.py), [`prefix_concat.py`](../tofu_sisa_lora/prefix_concat.py) · [`PEFT_BAKEOFF_2026-07.md`](../tofu_sisa_lora/reports/PEFT_BAKEOFF_2026-07.md) · [`log/peft_compose/`](../log/peft_compose/) |
| recall retained vs N merged (11% at N=200); perplexity 8.5 vs never-trained 14.6 | [`analyze_nmerge.py`](../tofu_sisa_lora/analyze_nmerge.py), [`plot_nmerge.py`](../tofu_sisa_lora/plot_nmerge.py) · [`NMERGE_REPORT_2026-07-08.md`](../tofu_sisa_lora/reports/NMERGE_REPORT_2026-07-08.md) |

## Appendix E — routing under deletion

This appendix **is** the [`log/router_leak/`](../log/router_leak/) campaign (18 entries).

| claim | repo |
|---|---|
| the 10 selection mechanisms and what deletion does to each | [`router.py`](../tofu_sisa_lora/router.py) (8 feature-space + 4 behavioral), [`legonet_lora/`](../legonet_lora/) (keyed banks), [`ramole/`](../ramole/) (learned cross-attn gate), [`memadapt_tofu/`](../memadapt_tofu/) (memory + block-list), [`apa_uniform_sum/`](../apa_uniform_sum/) (APA attention aggregation), `sift_masks*`/`clamu*` (per-source / per-cluster masks) · [`ROUTING_MASTER_2026-07-23.md`](../tofu_sisa_lora/reports/ROUTING_MASTER_2026-07-23.md) |
| orphan destinations after a 20-source deletion — 400/400 reassigned; busiest 0.11–0.82; n_eff 1.4–27.1 | [`router_family_audit.py`](../tofu_sisa_lora/router_family_audit.py), [`analyze_router_family.py`](../tofu_sisa_lora/analyze_router_family.py), [`analyze_orphan_destinations.py`](../tofu_sisa_lora/analyze_orphan_destinations.py) · [`orphan_destinations.md`](../tofu_sisa_lora/reports/orphan_destinations.md), [`rl_family_leak_table.md`](../tofu_sisa_lora/reports/rl_family_leak_table.md) |
| confidence refusal fails — AUC 0.57–0.61; catching 360/400 orphans needs refusing 82–88% of retained traffic | [`analyze_router_leak.py`](../tofu_sisa_lora/analyze_router_leak.py), [`routing_audit_tofu.py`](../tofu_sisa_lora/routing_audit_tofu.py) · [`log/ramole/2026-07-07_routing-fix-arms.md`](../log/ramole/2026-07-07_routing-fix-arms.md) (the refutation) |
| a learned gate separates orphans at AUC 0.556 ≈ chance | [`train_router_tofu.py`](../tofu_sisa_lora/train_router_tofu.py), [`ramole/router_lora.py`](../ramole/router_lora.py) · [`log/router_leak/`](../log/router_leak/) H-TRAINED (3 seeds) |
| a per-source sentinel separates orphans at AUC 0.982 | the tombstone provenance ladder in [`router_family_audit.py`](../tofu_sisa_lora/router_family_audit.py) · [`ROUTER_LEAK_REPORT_2026-07-18.md`](../tofu_sisa_lora/reports/ROUTER_LEAK_REPORT_2026-07-18.md) |
| oracle routing reaches 0.824 utility; a real embedding router costs 0.064 before any deletion | [`eval_routed_scaffold.py`](../tofu_sisa_lora/eval_routed_scaffold.py) · [`K200_ORACLE_ROUTING_REPORT_2026-07-20.md`](../tofu_sisa_lora/reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md) (**0.8236**, Δmu 0.0000 on deletion) · [`log/routing_scaffold/`](../log/routing_scaffold/) |
| deleting 20 of 200 sources moves 5.8% of retained queries to a different expert | [`routing_audit_tofu.py`](../tofu_sisa_lora/routing_audit_tofu.py) · [`ROUTING_AUDIT_REPORT_2026-07-06.md`](../tofu_sisa_lora/reports/ROUTING_AUDIT_REPORT_2026-07-06.md) |

## Baselines

| baseline | repo |
|---|---|
| S³T | [`s3t_*.py`](../tofu_sisa_lora/), [`train_s3t_shard.py`](../tofu_sisa_lora/train_s3t_shard.py) · [`log/s3t/`](../log/s3t/) · upstream pinned by [`../fetch_upstream.sh`](../fetch_upstream.sh) |
| APA | [`apa_uniform_sum/`](../apa_uniform_sum/) (full repo: Exp A/B/C) |
| Memory Adapters | [`memadapt_tofu/`](../memadapt_tofu/) · [`log/memory_adapters/`](../log/memory_adapters/) |
| MemSinks (the precursor NULL builds on) | [`memsinks_tofu/`](../memsinks_tofu/) · [`log/memsinks/`](../log/memsinks/) · upstream pinned |
| GradDiff / GA / KL / IDK | [`train_tofu_unlearn.py`](../tofu_sisa_lora/train_tofu_unlearn.py) · [`log/tofu_baselines/`](../log/tofu_baselines/) |
| MIA / privacy battery | [`attack_mia.py`](../tofu_sisa_lora/attack_mia.py), [`mia_attacks.py`](../tofu_sisa_lora/mia_attacks.py) · [`DELETION_AUDIT_REPORT_2026-07-06.md`](../tofu_sisa_lora/reports/DELETION_AUDIT_REPORT_2026-07-06.md) |

**Baselines with no code here — stated, not hidden.** `GRAM`, `NULL`, `SGTM`, `NPO`, `SimNPO` and
`RMU` appear in the main-paper tables but have no implementation in this tree. NPO/SimNPO/RMU are
open-unlearning methods (reachable via [`../ou_integration/`](../ou_integration/) once
`fetch_upstream.sh` has cloned the fork); GRAM/NULL/SGTM were run outside this tree and would have
to be brought over separately. [`../STATUS.md`](../STATUS.md) carries this as an open gap.

**MUSE.** The MUSE-News results are not reproducible from this repo either — the benchmark and its
Llama-2-7B fine-tuned/retrained targets are not vendored, and no MUSE loader exists in the tree.
TOFU is fully covered.
