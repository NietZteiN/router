# tofu_sisa_lora

Research code for **exact machine unlearning in LLMs** on the
[TOFU](https://huggingface.co/datasets/locuslab/TOFU) benchmark: if a model's knowledge of a
given author lives in one removable module, deleting that author should be dropping the module —
cheap, deterministic, O(1) — rather than approximate weight surgery like gradient ascent.

The tree is a working research tree, not a library. It carries several parallel investigations
that share one eval harness:

| Track | Idea |
|---|---|
| **SISA-LoRA** | one LoRA adapter per data shard; unlearn by re-merging without that shard |
| **Merge mechanism** | *why* merging adapters destroys recall — subspace collision, dilution, the N-merge ladder |
| **Routing + scaffold** | serve one expert per query instead of merging; a public scaffold supplies general competence |
| **SIFT-Masks / ClAMU** | full-FT task vectors with sign constraints or optimized cluster masks; exact unlearn by subtraction |
| **LegoNet / RAMoLE / SEA / S³T** | comparison methods, reproduced on the same metrics |
| **Router leak** | what a router *discloses* about deleted data, and where orphaned queries land |

The headline finding the tree keeps re-confirming: **every separable weight-space merge lands in
a narrow utility band (mu ≈ 0.42–0.48, barely above the base model), while serving one expert
per query clears 0.55+.** Merging is where the utility goes.

## Layout

```
eval_tofu.py            the canonical metric harness — ROUGE-L recall, answer probability,
                        truth ratio, model_utility, forget_quality. Ported to match
                        open-unlearning exactly; guarded by test_ou_equivalence.py.
merge_lora.py           merge-method registry (~23 methods) + eval label dispatch
merge_subset.py         N-merge ladders: merge arbitrary author subsets, materialized on CPU
train_lora_shard.py     the shard trainer (frozen recipe: r32/α64/5ep/lr1e-4)
submit_*.sh             SLURM drivers, one per campaign; STUB=1 previews without submitting
configs/*.json          every hyperparameter; nothing important is a CLI flag
reports/                results CSVs and written reports
test_*.py               CPU gates — run before any GPU job
cluster_env.<site>.sh   per-cluster settings; see SETUP.md
```

## Getting started

**Verify the metrics without a cluster** (CPU, seconds, no model, no network):

```bash
python test_ou_equivalence.py    # metric math == open-unlearning's
python test_eval_rows.py         # forget/retain row arithmetic
python test_merge_subset.py      # merge algebra + exact drop-a-term
python test_cluster_env.py       # site abstraction
```

**Run jobs:** see [SETUP.md](SETUP.md) — environment, HF token, and how to retrain the adapter
pools (~6.5 GPU-hours; they are far too large for git).

## Conventions worth knowing before you change anything

- **Metrics are frozen.** `eval_tofu.py` reproduces open-unlearning's formulas numerically;
  `test_ou_equivalence.py` must stay green or every historical number becomes incomparable.
- **Compare within a convention.** The cross-campaign noise floor is ~0.01 mu, and the whole
  merge band is only 0.04 wide. Same pool, same rank, same eval split, or it is not a comparison.
- **Configs, not flags.** Hyperparameters live in `configs/*.json` so a run is reproducible from
  a path.
- **CPU gate first.** Every campaign has one; they exist because a bad merge or a mis-parsed
  label fails *silently* and produces plausible numbers.
- **Provenance.** Result JSONs carry `script_sha256`, `git_hash`, config path and SLURM job id.

The dated experiment ledger — hypotheses stated before each run, verdicts after — lives in the
separate `log/` repo. `CLAUDE.md` is the detailed how-it-works reference for the code.
