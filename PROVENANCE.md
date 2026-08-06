# PROVENANCE

Where every file came from, what was changed on the way in, and how to keep the two copies in
step without creating a fork.

## Source trees

| root | default | what came from it |
|---|---|---|
| `$TOFU_TREE` | `/home/jack/tofu-unlearning` | `tofu_sisa_lora/`, `log/` |
| `$TOFU_HOME` | `/home/jack` | the eight sibling projects, `papers/`, the open-unlearning patches |
| `$APA_TREE` | `/home/jack/apa-uniform-sum` | `apa_uniform_sum/`, and the site layer (`cluster_env*.sh`, `slurm_nodes.sh`, `stage_hf_cache.sh`) |
| `$MT7B_TREE` | `/home/jack/merge-tables-7b` | `merge_tables_7b/` — **only the parts that repo owns** (`reproduce/` + 6 top-level docs). Its `tofu_sisa_lora/`, `ramole/`, `legonet_lora/` and `external/` are a vendored duplicate of what this repo carries first-hand |

[`MANIFEST.files`](MANIFEST.files) is the allow-list: one line per vendored path, naming its root
and source. [`sync_from_tree.sh`](sync_from_tree.sh) is the only supported way to move code between
the two.

```bash
bash sync_from_tree.sh --check    # report drift, write nothing (exit 1 if any)
bash sync_from_tree.sh --pull     # tree -> repo, UNEDITED entries only
```

Statuses: `SAME` · `EDITED` (differs, and the manifest records why) · `DRIFT` (differs with no
reason recorded — the tree moved and this copy did not) · `MISSING` · `NOSRC` · `ONLYREPO`.

At export: **same 1490, edited 328, drift 0, missing 0** (plus 9 files that exist only here —
generated output and the docs listed below).

## Why an allow-list and not `rsync -a`

`merge-tables-7b` vendored the same tree with a blanket copy. Its own
[`reproduce/VENDOR_DRIFT.md`](merge_tables_7b/reproduce/VENDOR_DRIFT.md) now records:

> 52 files differ, 16 only in the repo, 220 only in home … the repo copy is not a stale snapshot,
> it is a fork.

A blind sync in either direction destroys real work. There is deliberately **no `--push`**: the
working tree is where the campaign runs. Move a change home by hand, one file at a time.

## What was edited on the way in

Every vendored `.py`/`.sh` is marked `edited:deabsolutized`. Two rewrites, both mechanical, both
now gate-enforced by [`test_repo_selfcontained.py`](test_repo_selfcontained.py):

| | files | change |
|---|---|---|
| sibling paths | 113 | `/home/jack/<proj>` → `os.environ.get("<PROJ>_DIR", os.path.join(_REPO_ROOT, "<proj>"))`, anchored on `__file__` |
| storage paths | 246 (212 of them configs) | `/storage2/jack/...` → `${HF_HOME}` / `${TOFU_CKPT_ROOT}` / `${TOFU_CKPT_STORE}` / `${TOFU_DATA_ROOT}`, expanded by [`repo_env.py`](repo_env.py) at config-load with a hard error on unset |
| interpreters | all drivers | `/home/jack/anaconda3/envs/*/bin/python` → `${TOFU_PYTHON}` / `${TOFU_OU_PYTHON}` / `${TOFU_PLOT_PYTHON}` |
| SLURM policy | 6 sibling `slurm_nodes.sh` | hardcoded `sprint4` + a literal cap of 4 → derived from the site file, with the 4-GPU ceiling clamped centrally |

Rationale for each is in [`PORTING.md`](PORTING.md) §5.

Deliberately **not** rewritten, and why:

- `log/`, `papers/`, `paper/` — prose. A `/home/jack` path in the ledger is a historical record of
  where a job ran; rewriting it would falsify the record.
- `results_snapshot/**` and `merge_tables_7b/reproduce/results_snapshot/**` — result data. The
  paths inside are provenance written by the job that produced them.
- `cluster_env.sprint.sh`, `cluster_env.cispa.sh` — a site file's whole purpose is to name that
  cluster's paths.
- `merge_tables_7b/` — another repo, vendored verbatim. Rewriting it would fork it, which is the
  exact mistake its own `VENDOR_DRIFT.md` documents.
- `sync_from_tree.sh`, `repo_env.py` — they name the source trees / the literals they replaced.

## Files this repo OWNS

Never synced, never overwritten by `--pull`; they have no source line in the manifest.

| file | what it is |
|---|---|
| `README.md`, `SETUP.md`, `PORTING.md`, `STATUS.md`, `PROVENANCE.md` | the export's own documentation |
| `MANIFEST.files`, `sync_from_tree.sh` | the vendoring contract |
| `repo_env.py` | site-path resolution shared by all ten projects |
| `fetch_upstream.sh` | the third-party pins |
| `snapshot_results.py`, `results_snapshot/` | the offline evidence (generated) |
| `test_repo_selfcontained.py` | the portability gate |
| `requirements-ou.txt` | the second environment |
| `paper/README.md` | the per-claim map from the paper's claims to the code (the manuscript itself is absent) |
| `ou_integration/` | the open-unlearning fork state |

## The manuscript: vendored, then withdrawn

`tofu_sisa_lora/paper/*.tex` and the two AAAI PDFs were relocated to `paper/` during the export,
then **removed before the first push** when the GitHub repo turned out to be public: the
submission is anonymized and under review, and its notice forbids distribution. Both forms are
gitignored and gate-checked, and the originals stay in the working tree
(`/home/jack/tofu-unlearning/tofu_sisa_lora/paper/`, `/home/jack/papers/`).

`sync_from_tree.sh` still knows about the paths, so they do not report as `MISSING` forever — a
`MISSING` list that is always non-empty trains a reader to ignore it, and that list is the one
that catches a genuinely dropped file.

## Third-party, pinned rather than copied

S3T and MemSinks ship **no LICENSE file** (S3T shows an MIT badge in its README; MemSinks has
nothing), so vendoring their source would redistribute code under terms nobody stated. A pinned
commit gives the same reproducibility with none of that. open-unlearning *is* MIT, but it is a
live fork with local commits, so a pin plus a patch beats a snapshot that silently drifts.

| upstream | pin |
|---|---|
| `github.com/brcsomnath/S3T` | `3ecb73b86ff96f8c8eb38e2bd2afa30899a7400e` |
| `github.com/AR-FORUM/MemSinks` | `a00511991e2be2c3d9016585b3498b9d5ea914d8` |
| `github.com/locuslab/open-unlearning` | `93e9cd5d8808bb43641d133b38bb34466f9aae2e` |

`ou_integration/patches/` additionally carries three files that were **uncommitted** in that fork
— a modified `src/model/__init__.py`, an untracked `src/model/sepmlp_registry.py`, and
`configs/model/SepMlp-Llama-3.2-1B.yaml`. They existed only as a dirty working tree on the
original cluster and are reachable from no git remote. Captured here or lost.
