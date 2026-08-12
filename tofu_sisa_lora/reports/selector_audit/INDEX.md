# selector_audit — artifact index

`tofu_sisa_lora/reports/` holds output from every campaign this repo has run (merge_mechanism, ctv,
sift, clamu, s3t, peft_compose, router_leak, …) — 200+ files. This index names the subset belonging
to the **selector_audit** follow-up, what each answers, and how to regenerate it.

Files are left where they are rather than moved: log entries and result JSONs reference these paths
by name, and relocating them would silently break those references for a cosmetic gain.

Master synthesis: [`../../../SELECTOR_AUDIT_REPORT.md`](../../../SELECTOR_AUDIT_REPORT.md).
Dated narrative: [`../../../log/selector_audit/`](../../../log/selector_audit/).

---

## §4.10 — the metric is blind to substitution

| Artifact | Answers | Regenerate |
|---|---|---|
| `../h29_forget_quality_ci.{json,md}` | Intervals on `forget_quality`: marginal per-cell CI (±0.35) **and** the paired spread CI [0.2245, 0.6975] that decides whether the destination spread is real | `selector_audit/bootstrap_fq.py --results_dir <pool>/results/extended_ci --ks_ref <pool>/results/extended/retain_tr_scores.npy --published_dir <pool>/results/extended` |
| `<pool>/results/extended/routed_{reroute_f10_s*,oracle_del_f10}.json` | The 8-arm destination sweep (extended tier). Read `route_stats` **before** any metric | `TIER=extended bash submit_e5_destination_sweep.sh` |
| `<pool>/results/extended_ci/*.forget_tr.npy` | Raw forget truth-ratio arrays — what makes a CI possible without re-serving the model | written by default; `--no_dump_forget_tr` suppresses |

⚠ `extended_ci/` is a **tagged rerun** (`RES_TAG=_ci`) that reproduces the published cells 8/8. The
KS reference loads off the tier flag, never off `--out`, so a tagged rerun scores against the
byte-identical reference.

## §4.3 — what the system says (CSAR)

| Artifact | Answers | Regenerate |
|---|---|---|
| `<pool>/results/router_leak/csar_k200_f10_qpa{5,20}[_name_stripped\|_indirect\|_centroid_sbert-random].{json,md}` | CSAR, refusal, own-disclosure per router × transform, incl. the random-destination floor | `bash submit_csar_audit.sh [gen\|score] QPA=20 QT=<transform>` |
| `h15/csar_decompose.{json,md}` | Is CSAR mostly a swapped name? Splits cross-source into **substantive** vs **name-only**; prints `unclassifiable_frac` per cell | `selector_audit/csar_decompose.py --csar_json <...> --out_json J --out_md M` |
| `*.label_me.jsonl` | 300-record hand-labelling samples — **the blocking input for §4.3** | `selector_audit/csar.py --sample_for_labeling 300` |

⚠ Read `unclassifiable_frac` before quoting any decomposed cell: 0.08–0.16 gold-form (fine),
0.27–0.33 `name_stripped`, **0.824** for `indirect`/`key_tfidf` (not quotable — the unit-88 magnet).

## §4.6 — the defense frontier

| Artifact | Answers | Regenerate |
|---|---|---|
| `h26/cost_{r32_e25,r32_e5,r8_e5}_{gold,name_stripped,indirect}.{json,md}` | Full frontier: AUC vs m, own-expert recall, and `operating_points` (catch → retained-traffic false refusal) — H24, H26 **and** H27 | `analyze_selector_cost.py --pool_dir <pool> --variant <_name_stripped\|_indirect\|> --drop_set 180-199` |
| `../cost_ppl_*.{json,md}` | The earlier single-pool H24 run, superseded by `h26/` | as above |

⚠ `--pool_dir` defaults to the **name-stripped** matrices; `--variant ''` selects gold-form.

## §4.9 — probe, granularity ladder, query shift

| Artifact | Answers | Regenerate |
|---|---|---|
| `../router_probe_*.{json,md}` | E1 per pool/rung: probe AUC, its shuffle + oracle-ceiling controls, confidence comparators, score-access attribution | `analyze_router_probe.py --family_npz GLOB --drop_set 180-199` |
| `../granularity_ladder.{json,md}` | The ladder at **constant deletion size** (k=10/50/200), monotonicity computed not eyeballed | `analyze_router_probe.py --rung LABEL:GLOB:DROPSET ...` (repeatable) |
| `../router_shift_k{10,50,200}.{json,md}` | Routing accuracy + detection under 6 query conditions, with attacker capture | `analyze_router_shift.py --k 200 --drop_set 180-199` |
| `h30/router_shift_h30.{json,md}` | Adds **`para_stripped`** and carries the probe's **shuffle control** into the table | as above (current code) |

⚠ The published k=50 ladder cells are `d49`/`d49_48` (4 and 8 authors) while `is_forget` marks all
400 forget10 rows — those cells label 16 of 20 authors as orphans with their expert still present.

## Cross-cutting

| Artifact | Answers |
|---|---|
| `../SELECTOR_AUDIT_OVERNIGHT.{json,md}` | Unattended campaign roll-up; lists what is missing rather than failing |
| `<pool>/results/router_leak/rl_family_k200*.{json,npz}` | The FAMILY NPZ CONTRACT matrices every CPU analysis consumes: `scores[n_q,k]`, `is_forget`, `author_of_q`, `author_sent_scores` |

---

## Reading rules that apply to every table here

1. Judge a **spread** with a paired bootstrap; marginal per-cell CIs are the wrong yardstick.
2. `forget_quality` resolution is the exact lattice **1/lcm(n, m)** — the sampled "achievable
   p-values" count is a lower bound and quotes nothing.
3. Exclude or flag the **18 authors with no extractable name** in any survivor-conditioned number.
4. `indirect` is **not** the hard transform — it carries the target's own facts. Use
   `name_stripped`, or `para_stripped` for a fully independent surface.
5. A sub-chance AUC is meaningless without its own shuffle control; here that control spans
   0.336–0.532, so fitted-probe differences under ~0.1 are not resolvable.
6. Feature-space routers read no expert weights, so a per-pool feature table is one column
   repeated. Recipe questions need the behavioral family.
