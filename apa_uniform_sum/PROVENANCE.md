# PROVENANCE

Where this code came from, and how to keep it in sync without creating a fork.

## The source tree

Everything here was carved out of `tofu-unlearning`, a larger research repo whose
`tofu_sisa_lora/` holds ~386 files across a dozen research tracks. This repo carries the
**import closure of Experiments A, B and C** — computed with `ast`, distinguishing top-level
from function-level imports — plus the site abstraction, the CPU gates, four reference tables
and one thread of the research ledger.

[`MANIFEST.files`](MANIFEST.files) is the allow-list: one line per vendored path, naming its
source. [`sync_from_tree.sh`](sync_from_tree.sh) is the only supported way to move code between
the two.

```bash
bash sync_from_tree.sh --check    # report drift, write nothing (exit 1 if any)
bash sync_from_tree.sh --pull     # tree -> repo, for UNEDITED entries only
```

Statuses: `SAME`, `EDITED` (differs, and the manifest says why), `DRIFT` (differs with no
reason recorded — the tree moved and this copy did not), `MISSING`, `NOSRC`.

## Why an allow-list instead of `rsync`

A previous vendoring of the same tree used a blanket copy. Its own `VENDOR_DRIFT.md` now
records the outcome:

> the two `tofu_sisa_lora/` trees have diverged, **in both directions** … The repo copy is not
> a stale snapshot — it is a fork.
>
> 52 files differ · 16 exist only in the repo · 220 exist only in home

The divergence was not random. Some files were newer in the repo *on purpose* (a cluster port),
others were simply stale. A naive sync in either direction destroys real work, and by the time
anyone notices, there is no record of which direction was intended for which file.

So: the direction is fixed (tree → repo), the set is explicit, and any file this repo edits
carries an `edited:<reason>` marker so a future sync is a deliberate merge rather than an
overwrite. There is deliberately **no `--push`** — move a change home by hand, one file at a
time, so the decision is made once per file rather than once per rsync.

## What was changed after vendoring

| file | change | why |
|---|---|---|
| `eval_mmlu.py` | vendored the 3 MMLU scoring primitives; added `--ood` | it `sys.path`-injected a sibling `legonet_lora/` tree resolved as `dirname(dirname(__file__))`, which on the development machine happened to exist — so a clone failed and local runs did not |
| `tofu_env.py` | added `hf_home()` | six modules each carried `os.environ.setdefault("HF_HOME", "/storage2/jack/...")` — another cluster's disk, which does not fail elsewhere, it just points HF at a missing directory |
| `eval_tofu.py`, `merge_subset.py`, `eval_baseline.py`, `prepare_eval.py`, `train_lora_shard.py` | HF_HOME via `tofu_env.hf_home()` | same |
| `plot_nmerge.py` | context CSV via `$TOFU_CKPT_ROOT` | was an absolute path |
| `submit_nmerge.sh` | `merge`/`eval`/`overlap` emit resources via `tofu_sbatch_resources`; config paths go through `expandvars` | those three stages hardcoded `--partition=all` and `--mem=`, so they failed at submit on a cluster where memory is not consumable; and the portable `${TOFU_CKPT_ROOT}` config form yielded a literal path |
| `measure_adapter_selectivity.py` | `--extra_splits` | Experiment C's behavioural tiers |
| `measure_expb_contrib.py` | (unchanged logic; marked for the tier surface) | already carried the Exp-C tiers |
| `test_expa.py`, `test_ou_equivalence.py` | portable fixture paths | pinned `/storage2` snapshot dirs |

`eval_tofu.py` is otherwise **byte-identical** to the working tree. Its metrics are frozen —
`test_ou_equivalence.py` is the guarantee that every historical number stays comparable — so the
only edit it carries is the one-line HF_HOME default. What makes that possible is that its
other-track arms are imported *lazily*; `test_repo_selfcontained.py::test_absent_arms_are_lazy_only`
pins exactly that.

## Files this repo owns

Never synced, no source line in the manifest:

```
submit_expb.sh                    configs/expb_selectivity_7b.json
collect_expb.py                   cluster_env.local.sh
plot_style.py                     stage_hf_cache.sh
plot_expa.py plot_expb.py plot_expc.py
test_expb_selectivity.py test_mmlu_primitives.py
test_repo_selfcontained.py test_plot_style.py
MANIFEST.files sync_from_tree.sh requirements-plots.txt
README.md SETUP.md PROVENANCE.md ARMS.md
```

## Run provenance

Results are not tracked by git hash. Each run records its own provenance in the artifact it
writes — `script_sha256`, the config path, the seed, and `SLURM_JOB_ID` — because a git hash
describes the tree, not the interpreter, the pool, or the cluster that produced the number.
`merge_meta.json` beside every materialized merge records the author set, the weights and the
compression, which is what makes a merge rebuildable and a `merges/` directory reclaimable.
