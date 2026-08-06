### Target Date: 2026-07-03 (§9-D synthesis + base-pinned & dropped-expert routing audit)

- **Hypotheses / what we're testing:** Completes the §9-D "post-deletion routing" experiment
  ([gap analysis §9-D](../EXACT_UNLEARNING_GAP_ANALYSIS_2026-06-29.md)) whose results table is
  still `tbd`. The [2026-07-02 battery](2026-07-02_followup-battery.md) measured orphan routing /
  selection shift / index displacement, but its embed-route numbers were produced with the
  **fine-tuned retriever** (`encoder_source` = retriever dir — the E0 footgun recorded, not yet
  re-run base-pinned), and no policy measures a **literal drop-an-expert** condition (this pool's
  deletion retrains in place, so orphans routing "into" experts is benign there — the drop case is
  the §9-D scenario proper).
  - **H1 (encoder confound is material):** a base-pinned audit (`encoder_pin:"base"`, new
    `_encbase` index cache) routes orphans DIFFERENTLY from the FT-encoder audit; per the
    [2026-06-29 finding](2026-06-29_retriever-ft.md) (retriever FT made forget-author routing
    worse: unlearn fq 0.484→0.180), predict base-pinned **sibling_top1 < 0.315** (the FT number).
    CONFIRM: a non-trivial gap in orphan metrics, base < FT sibling rate. REFUTE: routing
    identical (confound immaterial) or base > FT.
  - **H2 (dropped-expert fallback leak):** under a new `dropped` policy (the 15 affected experts
    masked from the index, re-ranked) — note K(a) ⊆ affected for every forget author by manifest
    construction, so "top-1 leaves K(a)" is TRIVIAL; the falsifiable content is *where* orphans
    land: embed retrieval has no abstain route, so predict orphan top-1 mass CONCENTRATES on a
    few near-duplicate siblings rather than diffusing — CONFIRM: top-3 most-hit surviving
    experts capture ≥ 50 % of orphan top-1 mass AND mean top1-sim ratio (masked/unmasked
    cosine) ≥ 0.9 (the sibling is nearly as good a match as the dropped expert ⇒ leak-prone).
    REFUTE: near-uniform spread (normalized top-1 entropy ≥ 0.9) or sim ratio ≤ 0.75 (orphans
    land on poor matches ⇒ generic answers, benign).
  - **H3 (retain collateral of a drop):** masking the affected experts shifts retain routing
    (those experts also serve retain authors' queries in their top-3): predict retain
    shift_top1(dropped vs stale) in the 0.3–0.6 range on this pool (15/32 experts masked,
    affected experts hold retain members too). Either way it quantifies §9-D metric 4 for the
    drop case. (Exploratory bound, not a sharp prediction.)
  - **H4 (author-key row at extended caps — runs in the `routing_scaffold` thread):** the O(1)
    `--delete_shard 9` drop on the strong-experts scaffold arm keeps `model_utility` unchanged
    and fq at the scaffold floor at EXTENDED caps (smoke: mu 0.7509→0.7509, fq 0.0003→0.3929).
    CONFIRM: extended Δmu ≈ 0 and del9 fq ≈ the never-trained floor. REFUTE: any measurable
    retain-side hit. Cross-referenced companion entry in `log/routing_scaffold/`.
- **Setup:** pool `/storage2/jack/checkpoints/tofu_sisa_lora/Llama-3.2-1B-Instruct_legonet_n32_k3`
  (n=32, k=3, tag `forget10`, 15 affected / 17 untouched). Code changes (this session):
  (i) `ramole_tofu.py::build_expert_index` — cache filename now encodes the encoder pin
  (`encoder_pin:"base"` → `expert_index_n32_encbase[.._ex…].npy`); previously a base-pinned run
  silently returned the FT-built cache (`expert_index_n32.npy`, bytes hash-asserted, untouched);
  (ii) `routing_audit_tofu.py` — new `dropped` policy (affected experts masked to −inf before
  `argsort`); (iii) new config `configs/ramole_tofu_1b_basepin.json` = `ramole_tofu_1b.json` +
  `"encoder_pin":"base"`; (iv) CPU regression additions in `test_routing_audit_tofu.py`
  (base-pin cache isolation + dropped-policy invariants). CPU gates all green before submission
  (`test_routing_audit_tofu.py`, `test_ramole_tofu.py`, `ramole/tests/test_alpha_capture.py`,
  `ramole/tests/test_routing_audit.py`). Audit commands (1 GPU each, SLURM, seed = base_seed 42):
  `python routing_audit_tofu.py --config configs/ramole_tofu_1b_basepin.json --tag forget10
  --policies stale rebuilt dropped key --device cuda --out results/routing_audit_forget10_basepin.json`
  and the FT re-run with `--policies dropped` → `results/routing_audit_forget10_ftdrop.json`.
  H4: extended KS reference COPIED from the SISA 1B dir's `results/extended/retain_tr_scores.npy`
  (method-independent; the smoke ref in `_experts_scaf_k10` is already a byte-identical copy of
  the SISA smoke ref, md5 2df9f8ff…) — no prepare_eval job needed — then
  `eval_routed_scaffold.py --extended` full / `--delete_shard 9`. All four jobs submitted by the
  new committed driver `submit_routing_audit_9d.sh` (the 2026-07-02 strong-experts runs
  440232/440233 were submitted ad hoc from a scratchpad — that provenance gap is now closed).
  Script sha256 (12-hex): routing_audit_tofu.py 1388e317de7a, ramole_tofu.py f7d04961b33b,
  submit_routing_audit_9d.sh ac1a59879b90, configs/ramole_tofu_1b_basepin.json 1e5696121d0c,
  submit_followup.sh 1a734c235d40. SLURM job IDs (2026-07-03): **440480** basepin audit,
  **440481** ftdrop audit, **440482** scaf extended full, **440483** scaf extended del9.
- **Results:** *(provisional §9-D table from the on-disk 2026-07-02 artifacts — jobs 440214/440216/
  440219/440222/440223 (pool) and 440232/440233 (scaffold arm, previously unlogged: strong-recipe
  experts r32/α64/e5/lr1e-4 on the scaffolded base, smoke mu 0.7509); base-pin/dropped/extended
  rows to be filled when today's jobs land.)*

  | §9-D row (router) | deletion op | (1) orphan→scaffold/orig | (2) leak: fq vs oracle | (3) retain Δmu | (4) retain shift | source |
  |---|---|---|---|---|---|---|
  | author-key (hard) | drop expert (scaffold arm) | orphans → base+scaffold by construction | del9 fq 0.393 = scaffold floor (smoke) | **0.7509→0.7509** (smoke) | 0 (routing is a frozen lookup) | `routed_scaffold_strong{,_del9}.json` |
  | key + router (RAMoLE key) | retrain-in-place | orig_top1 1.000 (audit `key`) | fq **0.890** (mu 0.507) | +0.013 (0.494→0.507) | **0.000** (asserted) | audit + `routerkey_*.json` |
  | encoder cluster-ID (embed, FT — confounded) | retrain-in-place | orig_top1 0.685 / sibling 0.315 (stale, FT) | ramoleft fq **0.180** | +0.006 | stale-vs-rebuilt top1 0.083 | audit + `ramoleft_*.json` |
  | encoder cluster-ID (embed, base-pinned) | retrain-in-place | *tbd (H1)* | ramole fq **0.484** (mu 0.477) | +0.010 | *tbd* | today's `_basepin` audit |
  | embed, dropped-expert | **drop** (masked) | *tbd (H2: pred sibling ≈ 1.0)* | n/a (routing-only) | n/a | *tbd (H3)* | today's `dropped` policy |
  | soft RAMoLE (alpha level) | retrain-in-place | router UNCHANGED; forget ppl 2.97→11.82 | fq 0.484 (embed) / 0.890 (key) | ≈ +0.01 | retain H_norm 0.818→0.842 (more uniform, not sharper) | `alpha_diag_key_*.json` |

- **What worked / hypothesis verdict:** *(to fill when today's jobs complete)*
- **Observations:** *(to fill; note already: the §9-D "largest utility hit" prediction for the
  learned router looks REFUTED on this pool — every unlearn Δmu is small and POSITIVE; and the
  fq floor artifact means the author-key row must be judged against the scaffold floor 0.393,
  not the 0.89 retain-oracle ceiling.)*
- **New questions / new hypotheses:** *(to fill)*
- **Next Steps:** *(to fill)*
