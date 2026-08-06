# apa-uniform-sum

Everything needed to run three experiments on **uniform-summation aggregation of parallel
per-author LoRA adapters** — the MUSR/APA architecture, where deletion is supposed to be
"drop the module" and there is no learned router.

Clone this one directory and you have the code, the configs, the reference tables the analyses
compare against, and the reasoning behind each choice. No sibling projects, no `sys.path`
tricks: `python test_repo_selfcontained.py` is a gate that proves it.

```
Δ = Σᵢ sᵢ BᵢAᵢ        aggregate N per-author adapters at equal weight
delete author X   ⇒   Δ(P∖X) = Δ(P) − wΔ_X       (exact, at FIXED w)
```

## The three experiments

| | Question | Headline output |
|---|---|---|
| **A** | How much utility does summing N adapters cost? | `figA1` utility/perplexity/MMLU vs N, three arms |
| **B** | Does author X's adapter actually *own* X — does deleting it remove X and nobody else? | `figB1` slopes, `figB2` leakage, `S(X)` |
| **C** | What happens to a query **no adapter owns**? | `figC1` Δ-vs-base per tier, `figC2` attribution |

C is the mechanism that explains A. With no router a new query is not ignored: every adapter
fires, so the aggregate injects a superposition of N irrelevant deltas.

### What is pre-registered

Stated before the runs, with the result that would refute each:

- **A** — uniform summation does *not* collapse at N ≤ 20. Per-author deltas are near-orthogonal
  (mean |cos| 0.0009–0.0051), so ‖Σ_N‖/‖Δ₁‖ tracks √N almost exactly (measured: 1.42 / 2.26 /
  3.19 / **4.54** at N = 2/5/10/20 against √20 = 4.47). Expect a **graded** decline, so the
  ladder runs past 20 or it measures a plateau. The earlier `dare0p9sum` collapse was DARE's
  1/(1−p) rescaling inflating each delta ~3.2×, not the sum rule.
- **B** — from the firing baseline (`reports/key_firing_e5.json`: `gate_median` 1.1018, verdict
  **LAZY**, `frac_ratio_lt_2` = 1.0) every adapter fires on every query type within ~10%. So
  `foreign_mass(X) ≈ 19/(19+1.10) ≈ 0.945`: dropping X removes ~5% of the aggregate's response
  to X's own questions ⇒ **S(X) ≈ 1**. *Falsifier:* `S(X) ≥ 3` with `ρ_Y ≥ 0.95`.
- **C** — unowned-query damage grows **√N** under λ=1 and shrinks **1/√N** under 1/N. Strong
  form: damage is a function of `rel_pert = ‖Σ‖_F/‖W₀‖_F` *alone*, so the arms collapse onto one
  curve (`figA4`). *Falsifier:* damage that does not track `rel_pert`, or attribution that
  concentrates (n_eff ≪ N) — which would mean the pool self-routes and the cost is mis-routing,
  not superposition.

## Quickstart — verify it before running anything

CPU, seconds, no GPU, no model, no network, no cluster:

```bash
python test_repo_selfcontained.py    # the repo runs from one directory
python test_ou_equivalence.py        # metric math == open-unlearning's, exactly
python test_merge_subset.py          # merge algebra + exact drop-a-term
python test_expa.py                  # Exp-A/C label grammar + data premises
python test_expb_selectivity.py      # Exp-B deletion contract + coverage
python test_mmlu_primitives.py       # the vendored MMLU scorers
python test_cluster_env.py           # the site abstraction
python test_plot_style.py --colors-only   # the palette, computed not eyeballed
```

Then preview every SLURM script without submitting anything:

```bash
STUB=1 bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json merge
STUB=1 bash submit_expb.sh  configs/expb_selectivity_7b.json plan
TOFU_SITE=cispa STUB=1 bash submit_expa.sh configs/nmerge_sum_expA_7b.json mmlu
```

The last line renders **another cluster's** job scripts from this machine. That is how the "no
`--mem` on CISPA" class of error gets caught before anything is submitted.

## Running it

Weights are not in git. See [SETUP.md](SETUP.md) — an env from `requirements.txt`, an HF token
(Llama-2 is gated), and ~6.5 GPU-hours to retrain the adapter pools.

```bash
bash submit_pool.sh anchors        # ft + retain90 — FIRST; the KS reference derives from the oracle
bash submit_pool.sh pilot          # 2 authors, the end-to-end gate
bash submit_pool.sh r32            # the 200-adapter e5 pool  (Exp A / C)
bash submit_pool.sh e25            # the 200-adapter e25 pool (Exp B)

bash submit_nmerge.sh configs/nmerge_sum_expA_7b.json plan     # then merge -> eval -> collect
bash submit_expb.sh  configs/expb_selectivity_7b.json plan     # prints its own dependency chain
```

`plan` prints the submission order and the `DEP=` chain to paste. **Never leave two GPU arrays
queued at once** — each is `%${TOFU_ARRAY_CAP}`, and on the `sprint` site that cap *is* the
global GPU budget.

### Another cluster is a file, not a patch

Site settings live in `cluster_env.<site>.sh`, selected by `$TOFU_SITE`:

| site | what it is |
|---|---|
| `sprint` | A40 46 GiB, partition `all`, `--mem` honoured, 4-GPU global cap |
| `cispa` | A100 40 GB, partition `xe8545`, account `testing`, **`--mem` must not be emitted** |
| `local` | one workstation, with or without SLURM; sets no absolute path |

Drivers emit resources through `tofu_sbatch_resources <gpus> <cpus> <mem>` and the in-job
environment through `tofu_job_prologue`, so a memory policy is a site fact rather than an edit
across every driver. `gpus=0` emits no `--gres`, keeping CPU stages off the GPU cap.

**No cluster at all?** Use `TOFU_SITE=local` and `STUB=1`: every driver prints its job script
instead of submitting, and the printed `python …` line runs directly.

## Layout

```
merge_subset.py            the aggregation: --authors / --lam / -n, and the `norms` ladder
eval_tofu.py               canonical metrics, frozen (test_ou_equivalence.py is the guarantee)
eval_mmlu.py               MMLU + --ood dbpedia,alpaca on the served adapter
measure_adapter_selectivity.py   Exp B driver; --extra_splits adds the Exp-C tiers
measure_expb_contrib.py    per-author SIGNED contribution decomposition + cancellation index
analyze_expa.py collect_expb.py  CSV assembly
plot_style.py plot_exp{a,b,c}.py figures  (run under ${TOFU_PLOT_PYTHON})
submit_{nmerge,expa,expb,pool}.sh  SLURM drivers; every one honours STUB=1 and DEP=
cluster_env*.sh slurm_nodes.sh     the site abstraction
configs/                   the campaign configs — hyperparameters live here, not in flags
reports/                   reference tables the analyses compare against
log/merge_mechanism/       the dated research ledger for this thread
MANIFEST.files sync_from_tree.sh   the allow-list this code was vendored through
```

## Conventions worth knowing before changing anything

- **Metrics are frozen.** `eval_tofu.py` reproduces open-unlearning's formulas numerically.
  `test_ou_equivalence.py` must stay green or every historical number becomes incomparable.
- **Compare within a convention.** The cross-campaign noise floor is ~0.01 mu and the whole
  merge band is 0.04 wide. Same pool, same rank, same eval split — or it is not a comparison.
  The only legitimate ~0.45 comparators here are same-pool: `additive_mean` 0.459,
  `centered_pool`, `centered_lowrank`, and the anchors base 0.426 / ft_r32 0.7563 /
  retain90 0.5627.
- **Deletion means FIXED weights.** `Δ(P∖X) = Δ(P) − wΔ_X` holds only when w does not depend on
  N. Renormalizing the survivors to 1/(N−1) is a re-merge, not a deletion — and it still runs
  and still reports numbers. Pinned by `test_expb_selectivity.py`.
- **Configs, not flags.** Hyperparameters live in `configs/*.json` so a run is reproducible from
  a path.
- **CPU gate first.** Every campaign has one. They exist because a bad merge or a mis-parsed
  label fails *silently* and produces plausible numbers.
- **Report effect sizes, not p-values.** KS at n = m = 20 has 20 attainable p-values and needs
  D ≥ 0.45 for α = 0.05; with 5 targets the Wilcoxon floor is 0.0625. `ks_statistic` and
  `forget_truth_ratio` are the effect sizes; `forget_quality` is a descriptive ordinal.
- **The ledger is append-only.** Corrections go in a new dated entry referencing the old one.

## Two data facts that shape everything

1. **Perturbed coverage is all-or-nothing per author.** `forget10_perturbed` → authors
   **180–199**, `retain_perturbed` → **0–19**, exactly 20 rows each, 0 unjoinable. Authors
   **20–179 have none.** Since the retain90 oracle trained on 0–179, the only authors that are
   both paraphrase-covered *and* outside the oracle — i.e. where Forget Quality means what it
   says — are **180–199**. That is why Experiment B's aggregate is exactly those 20.
2. **`holdout10` is genuinely never trained** — 0/400 of its questions appear in `full`. It is
   the honest "new query" set for Experiment C, not merely an MIA non-member pool.

Both are re-verified against the real dataset by `test_expa.py` and `test_expb_selectivity.py`.

## Provenance

This tree was carved out of a larger research repo through an explicit allow-list
([MANIFEST.files](MANIFEST.files)); [PROVENANCE.md](PROVENANCE.md) records where each file came
from and how to keep the two in sync without the fork that a blind `rsync` produces.
[ARMS.md](ARMS.md) lists the other-track code deliberately left out.
