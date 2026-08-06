# routing_scaffold — the core method: routed isolated experts + public scaffold, exact O(1) deletion

**Status:** active (started 2026-07-02). **2026-08-06: the tree is portable and published** — [github.com/NietZteiN/router](https://github.com/NietZteiN/router) (public, commit `857afe7`); porting to a new cluster is one `cluster_env.<site>.sh`. The AAAI manuscript is deliberately **not** in it (anonymized, under review) — gitignored and gate-checked. First reproduction of the project's headline
("routing+scaffold beats full-FT with exact deletion") **from committed code** — the scaffold trainer
did not previously exist and the 0.664 number was never reproducible. Report:
`../../tofu_sisa_lora/reports/ROUTING_SCAFFOLD_REPRO_2026-07-01.md`.

**Headline (2026-07-20): ORACLE routing over 200 per-author task vectors (e25, no scaffold)
mu 0.8236 — best of any track, utility-flat in k to single-author granularity, deletion exactly
utility-free (Δmu 0.0000, f_ppl 1.05→17.72)** — the never-run "steps-matched k=200 routed arm,"
closed. Full self-contained report:
[`K200_ORACLE_ROUTING_REPORT_2026-07-20.md`](../../tofu_sisa_lora/reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md). Previous headline (2026-07-06): routed strong experts + scaffold mu 0.7509 vs
matched-capacity full-FT 0.6372 (+0.114, identical ingredients) — with verified byte-identical
O(1) deletion. Mechanism: routing doesn't memorize better (retain 0.854 vs 0.874), it ISOLATES
fine-tuning damage — FT on all authors degrades general knowledge (real 0.63→0.44, world 0.66→0.55),
routing structurally protects it.**

**Control closed (2026-07-07): the mechanism survived the scaffold×composition 2×2** — merging the
same strong experts on the same scaffolded base caps at mu 0.4938 (OOD-aware, retain_prob 0.199 ≈
the no-scaffold ceiling), so the −0.26 routed-vs-merged gap is pure composition. The scaffold cannot
rescue a merge; routing was doing the work.

## Hypotheses — open
- **[open]** H-k200-scaf (2026-07-20): per-author experts trained ON the scaffolded base +
  oracle routing lift the OOD floor and push mu beyond 0.8236 (the k=10 scaffold evidence
  says yes).
- **[open]** H5 (rescue sweep): no post-hoc merge operator (knots/tsv/della/jd/regmean/fisher/
  lorahub) reaches routed — best 0.55–0.65 OOD-aware, all < 0.70. Jobs 441021/441023.
- **[open]** H6 (rescue): prediction-level `ensemble_probs` beats weight averaging on retain
  recall but stays < routed (k× serving cost). Job 441022.
- **[open]** H7 (SIFT-on-scaffold): masked merging rebuilt on the scaffolded base reaches
  mu ≥ 0.74 (decisive ≥ 0.7509) with OOD at the scaffold floor. Chain 441024–441027.
- **[open]** H8 (SIFT-scaf deletion): re-derive-and-subtract preserves mu.
- (next planned: Alpaca-replay matched-FT control, extended+multi-seed on the headline pair.)

## Hypotheses — resolved
- **H-k200-1 (2026-07-20): oracle routing over 200 well-trained per-author task vectors
  holds mu ≥ 0.70.** ✓ SUPPORTED emphatically: e25 oracle-routed mu **0.8236 — best of any
  track in the repo** (ret_prob 0.999/rouge 1.000/ppl 1.05 per-author; OOD at the intact
  base floor 0.778/0.679). Routing is utility-flat in k to full per-author granularity.
  [2026-07-20_k200-oracle-routing-results.md]
- **H-k200-2: June's k=200 bottleneck was training dose, not routing.** ✓ SUPPORTED:
  e25−e5 oracle gap +0.233 (0.8236 vs 0.5908; June r8/e5 was 0.4728).
- **H-k200-3: author-level deletion utility-free.** ✓ SUPPORTED: Δmu = 0.0000 both pools,
  all retain/real/world components identical to 4 decimals; f_ppl 1.05→17.72 ≈ base,
  del-arm forget rows bit-identical across pools (deletion surface pool-independent).
- **H-k200-4: lexical router pays for its shard_0 OOD fallback on strong experts.**
  ✓ SUPPORTED: −0.0437 mu vs oracle on e25 (real 0.778→0.687, world 0.679→0.643, ret_prob
  0.999→0.917); only −0.004 on e5 — dose-dependent, as predicted.
- **H1: routing (isolated experts) + public scaffold ≥ full-FT utility.** ✓ SUPPORTED (modestly):
  OOD-aware routed+scaffold **mu 0.556 > full-FT 0.530** (1B, smoke). The claimed **0.664 is REFUTED**
  as inflated — not reproducible.
- **H2: deletion is exact and O(1).** ✓ SUPPORTED: dropping a shard raises its authors'
  forget_quality 0.135→0.393 (= the scaffold-floor / never-trained level) with **model_utility
  identical (0.5559→0.5559)** — every other author byte-unaffected. Re-verified at strength
  (0.7509→0.7509, fq 0.0003→0.3929).
- **H3: the scaffold's value is generic QA competence, added additively.** ✓ but with a caveat:
  it only helps if OOD queries are routed to scaffold-ONLY; a TOFU expert applied to OOD queries
  destroys it (see What didn't).
- **H4: weak experts were the margin limiter.** ✓ SUPPORTED: frozen recipe (r32/α64/e5) trained ON
  the scaffolded base → retain_prob 0.304→0.854, mu 0.556→**0.7509**.
- **H5: headline survives extended caps.** ✓ SUPPORTED: 0.5564 extended ≈ 0.5559 smoke (weak config).
- **H6: beats the MATCHED-capacity single model.** ✓ SUPPORTED decisively: 0.7509 vs 0.6372 (+0.114),
  same base/scaffold/recipe/data. See [2026-07-06_strong-experts-fair-fight.md].
- **H1 (scafmerge control): merged-on-scaffold collapses to the interference ceiling.**
  ✓ SUPPORTED decisively: OOD-aware merged mu **0.4938/0.4435** (additive/dare), retain_prob
  0.199/0.171 vs routed 0.854 — ≈ the no-scaffold merge ceiling; with identical OOD serving the
  full −0.26 gap to routed 0.7509 is composition. **The scaffold cannot rescue a merge — the
  routing-isolates-damage mechanism survives its control.** [2026-07-07_scafmerge-control-results.md]
- **H2 (scafmerge): merged-everywhere damages OOD; OOD-aware restores floor.** ✓ SUPPORTED
  (one marginal cell: dare world −0.022 < 0.03): additive real/world −0.181/−0.104, worsening with
  λ; arm B restores 0.6305/0.6556 exactly, all rows.
- **H3 (scafmerge): merged deletion utility-neutral but NOT serving-inert.** ✓ SUPPORTED on core
  (|Δmu| ≤ 0.006 all pairs; retain weights move on deletion vs routing's byte-identical drop);
  fq sub-prediction mixed/weakly-informative (dilution ceiling + KS grid); real deletion signal =
  forget_ppl 9.56→15.67.
- **H4 (scafmerge): λ can't rescue the merge.** ✓ SUPPORTED: mu 0.4557/0.3987/0.2567 for
  λ=0.10/0.15/0.20 (retain_ppl 8.6→31.8 norm-overshoot cliff, ≡ the 7B sweep shape).

## What worked
- **The 2×2 control (2026-07-07):** merged strong experts on the SAME scaffolded base cap at mu
  0.44–0.50 with retain_prob ≤ 0.20 under both merge conventions and every λ — the headline's causal
  decomposition is now complete: *same base, same scaffold, same experts — merge them → 0.49, route
  them → 0.75.* Bonus contrast for the write-up: merged deletion is data-exact but moves every
  retain author's serving weights; routed deletion is byte-identical.
- **OOD-aware routing** (`eval_routed_scaffold.py`): TOFU-author query → shard expert (exact q2author,
  76/76); OOD (real/world) → scaffold-only (60/60). Lifts mu 0.474 → **0.556** by restoring real/world
  prob to the scaffold-floor 0.63/0.66.
- **Additive composition via a scaffolded base** (`make_scaffolded_base.py`): merge scaffold into base,
  route experts on top → base+scaffold+expert, no eval_tofu change.
- **Exact O(1) deletion** demonstrated (`--delete_shard`).

## What didn't
- The **0.664 headline does not reproduce** — as committed, routed+scaffold = 0.474 < full-FT 0.530
  (a composition bug: `routed_key_exact` applies a TOFU expert to *every* query, corrupting OOD answers
  real 0.63→0.37, world 0.66→0.50). Fixed by OOD-aware routing.
- retain_prob (0.304) < full-FT (0.390): the k=10 experts are the weak r8/α16 legacy recipe.

## Open ideas
- Extended-cap + multi-seed (43/44) on the headline pair (strong routed 0.7509 vs matched-FT 0.6372).
- **Alpaca-replay matched-FT control**: mix scaffold data into the matched baseline's training (classic
  CF mitigation) — does routed still win?
- Realistic router: encoder cluster-ID instead of exact q2author; measure the routing-error cost.
- k=20/50 experts (smaller deletion units) — does the margin hold as experts shrink?
- A **training-free router** for the author↔OOD decision without a leaky learned gate.

## Entries
- [2026-07-02_scaffold-repro.md](2026-07-02_scaffold-repro.md) — repro (0.474) → bug diagnosis → OOD-aware fix (0.556 > full-FT) → exact-deletion demo.
- [2026-07-06_strong-experts-fair-fight.md](2026-07-06_strong-experts-fair-fight.md) — strong experts 0.7509; extended replicates; fair fight won +0.114; mechanism = routing isolates fine-tuning damage.
- [2026-07-07_scafmerge-control-design.md](2026-07-07_scafmerge-control-design.md) — pre-registration: scaffold×composition 2×2 control (the never-run scaffold+MERGED cell, arms A/B + λ-robustness); `--merged_label` serving built, CPU gates 5/5 + merge_extra green, SLURM staged pending human review.
- [2026-07-07_scafmerge-control-results.md](2026-07-07_scafmerge-control-results.md) — RESULTS (jobs 440914/440916): H1–H4 all supported; OOD-aware merged mu 0.4938/0.4435 ≈ no-scaffold ceiling; routing, not the scaffold, is the mechanism; λ ladder 0.456→0.399→0.257; dare_ties unseeded-mask provenance caveat.
- [2026-07-07_scafmerge-rescue-design.md](2026-07-07_scafmerge-rescue-design.md) — pre-registration H5–H8: post-hoc merge-family sweep (knots/tsv/della/jd/regmean/fisher/lorahub, 441021/441023) + `ensemble_probs` (441022) + **SIFT-Masks rebuilt on the scaffolded base** (chain 441024–441027; free `merge_full` bonus cell = full-FT confirmation of H1); gates green, all queued.
- [2026-07-19_k200-oracle-routing-design.md](2026-07-19_k200-oracle-routing-design.md) — pre-registration H-k200-1..4: 200 per-author task vectors + ORACLE (q2author) routing — the never-run steps-matched k=200 routed arm. Completes the e25 pool (180 new shards, job 445711) then 8 routed evals (445712: oracle full/del199 + lexical key_exact/no199 × e5/e25 pools) via the new `--lazy_adapter_cache` fp32-memory-wall fix (gate `test_lazy_adapters.py` green).
- [2026-07-20_k200-oracle-routing-results.md](2026-07-20_k200-oracle-routing-results.md) — RESULTS (445711/445712, zero failures): **H-k200-1..4 all supported** — oracle e25 mu **0.8236** (repo best; ret 0.999/1.000/1.05, OOD = intact base), e25−e5 +0.233 (dose, not routing), deletion Δmu 0.0000 with f_ppl 1.05→17.72, lexical router −0.0437 on strong experts. mu now capped by base OOD components → H-k200-scaf opened. Report: [K200_ORACLE_ROUTING_REPORT_2026-07-20.md](../../tofu_sisa_lora/reports/K200_ORACLE_ROUTING_REPORT_2026-07-20.md).
- [2026-07-20_k200-scaf-design.md](2026-07-20_k200-scaf-design.md) — pre-registration H-scaf-k200-1..3 (scaffolded per-author experts + oracle routing); chain 446371–446374 submitted then **cancelled by the user before any task ran** (0 GPU-h) — hypotheses stay open, pre-registration valid for a future launch.
- [2026-08-06_routing-repo-export.md](2026-08-06_routing-repo-export.md) — **INFRASTRUCTURE (no GPU): the whole routing/selection tree exported as a portable repo (`~/tofu-routing`, commit `7f74118`, 2,361 files / 96 MiB).** Ten projects flattened to siblings + the manuscript + 136 ledger entries + a 476-file / 42 MB `results_snapshot` that makes every CPU analysis runnable with no GPU and no scratch FS. **113 files** de-absolutized off `/home/jack`, **246** (incl. **212 configs**) off `/storage2` into site variables expanded by `repo_env.py` with a hard error on unset. Gates: self-containment 12/12, routing 11/11, sepmlp 77, blocktc 91, memadapt 24, legonet 9, memsinks 22, ramole 6/6, snapshot 476/476 sha256; `sync_from_tree.sh --check` drift 0 / missing 0. Four of my own automated rewrites broke code and were caught by gates rather than review — the worst gutted `cluster_env.sprint.sh` itself, so every driver silently resolved to system python. Also captured the three **uncommitted** open-unlearning files that were reachable from no git remote. Push pending (the user creates the empty **private** repo — `paper/pdf/` is an anonymized submission under review).
- [2026-08-06_repo-public-push.md](2026-08-06_repo-public-push.md) — **PUSHED, and the plan above corrected on both counts.** The repo the user created is `NietZteiN/router` (not `tofu-routing`) and is **public** — caught by an unauthenticated `api.github.com` check *before* the push, not after. The AAAI submission was therefore withheld: both the PDFs **and** the three `.tex` sources, since publishing the LaTeX while withholding the compiled PDF discloses the same Appendix D/E prose. A late `git rm` is not enough — the blobs were in the first commit — so history was rebuilt to a single commit `857afe7` and the object store purged; **verified by content hash** (`hash-object` → `cat-file -e`, all five absent) and by auditing the published tree from outside (2,636 paths, `paper/` = `README.md` only, 0 violations). New gate `test_manuscript_absent` checks the working tree **and** `git ls-files`, because `.gitignore` does not apply to an already-tracked file. `papers/` (22 publisher PDFs) stays public at the user's explicit call, recorded in `STATUS.md`.
