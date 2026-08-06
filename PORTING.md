# PORTING — moving this tree to another cluster

Porting is **one file per site-file tree**. Everything cluster-specific lives in
`cluster_env.<site>.sh`; nothing else in the repo names a machine, a filesystem, or an interpreter.

That was not true before the export. This document records what had to change, so that the next
port is a copy of one file rather than a re-derivation.

## 1. Write your site file

Three directories carry their own `cluster_env.sh`, and each resolves site files **next to
itself** — the repo root, `tofu_sisa_lora/` and `apa_uniform_sum/`. A site that exists in one but
not the others fails at the site-file check for every driver under the directory that is missing
it, which is what `TOFU_SITE=local` did under `tofu_sisa_lora/` until 2026-08-06.

```bash
for d in . tofu_sisa_lora apa_uniform_sum; do
  cp "$d/cluster_env.local.sh" "$d/cluster_env.mysite.sh"
done
$EDITOR ./cluster_env.mysite.sh tofu_sisa_lora/cluster_env.mysite.sh apa_uniform_sum/cluster_env.mysite.sh
export TOFU_SITE=mysite
```

Four values have no sensible default and fail loudly, naming the export that fixes them:

| variable | what it is |
|---|---|
| `TOFU_PYTHON` | an interpreter with `tofu_sisa_lora/requirements.txt` installed |
| `HF_HOME` | the HuggingFace cache. Must contain `hub/`; put the Llama-2 token in `$HF_HOME/token` |
| `TOFU_CKPT_ROOT` | where pools, merges and results are written. Never `/home` |
| `TOFU_PARTITION` | the SLURM partition |

Three more are **derived** from `TOFU_CKPT_ROOT` and only need overriding if your storage is split
across filesystems: `TOFU_CKPT_STORE` (its parent — how a project reaches a sibling's
checkpoints), `TOFU_STORAGE_ROOT`, `TOFU_DATA_ROOT` (non-HF datasets).

Optional: `TOFU_ACCOUNT`, `TOFU_EXCLUDE` (empty ⇒ the `--account`/`--exclude` lines are not
emitted at all), `TOFU_SUPPORTS_MEM=0` if your partition rejects `--mem` (CISPA does — nodes
report `RealMemory=1`, so *any* `--mem` fails at submit and must be dropped, not lowered),
`TOFU_OU_PYTHON` for the second environment (see below).

Check it without submitting anything:

```bash
python test_repo_selfcontained.py                              # 12 checks
python tofu_sisa_lora/test_cluster_env.py
TOFU_SITE=mysite STUB=1 bash tofu_sisa_lora/submit_router_family.sh   # prints the sbatch
```

`cluster_env.sprint.sh` and `cluster_env.cispa.sh` ship as worked examples — a slurm-with-`--mem`
site and a slurm-without-`--mem` site.

## 2. Two Python environments, not one

| env | pins | used by |
|---|---|---|
| `requirements.txt` | torch 2.5.1 / transformers 4.48.3 | everything: routing, merging, training, analysis |
| `requirements-ou.txt` | torch 2.4.1 / transformers 4.51.3 | the open-unlearning eval track only (sepmlp / blocktc / memadapt Table-1 rows) |

They are **not interchangeable**. The two `transformers` majors resolve the Llama-3.2 chat
template differently, so a metric computed under the wrong one is quietly wrong rather than an
error. `cluster_env.sh` exposes both as `$TOFU_PYTHON` and `$TOFU_OU_PYTHON`.

`requirements.txt` deliberately omits `matplotlib` — the `plot_*.py` scripts run under
`$TOFU_PLOT_PYTHON` (`requirements-plots.txt`).

## 3. Upstream clones

```bash
bash fetch_upstream.sh          # clone at the pinned commits + apply the OU patch
bash fetch_upstream.sh --check  # verify
```

S3T, MemSinks and open-unlearning are third-party and `.gitignore`'d. Until you run this, three
gates fail *by design*: `memsinks_tofu/test_memsinks.py` (hashes MemSinks'
`src/src/SeqTDModel.py` as its reference) and three `memadapt_tofu/tests/test_data.py` cases
(compare against OU's tokenizer). Override the location with `MEMSINKS_UPSTREAM_DIR`, `OU_DIR`,
`S3T_UPSTREAM_DIR` if you already have clones elsewhere.

## 4. The GPU cap is enforced, not documented

`CLAUDE.md` §1 sets a **global** ceiling of 4 concurrent GPUs summed across every queued job.
`cluster_env.sh` clamps `TOFU_ARRAY_CAP` to it and says so on stderr; each project's
`slurm_nodes.sh` derives its `%N` array throttle from that value. Raising it is a deliberate edit
to `TOFU_GPU_CAP_CEILING`, not an accident in a site file.

## 5. The contract this repo maintains — and what it cost

Two classes of hard-coded path had to go. Both are now gate-enforced by
[`test_repo_selfcontained.py`](test_repo_selfcontained.py), so they cannot creep back.

**Sibling projects (113 files).** Modules reached each other as `/home/jack/<project>`. They now
resolve as a repo-relative sibling with an env override:

```python
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAMOLE_DIR = os.environ.get("RAMOLE_DIR", os.path.join(_REPO_ROOT, "ramole"))
```

Anchored on `__file__`, never on cwd, so a script works from anywhere. **This is why the repo is
flat**: `<repo>/<project>` is exactly where `/home/jack/<project>` used to be, so the resolution
is a rename, not a redesign. Do not nest the projects under a subdirectory.

**Storage (246 files, of which 212 are configs).** Paths named one cluster's scratch filesystem.
Configs now say `"${TOFU_CKPT_ROOT}/..."`, and [`repo_env.py`](repo_env.py) expands them at
load time — with an **unset variable being a hard error**. `os.path.expandvars` leaves `${FOO}`
untouched when `FOO` is undefined, so a path-shaped value gets created on disk verbatim; that
already happened once upstream (a literal `${TOFU_CKPT_ROOT}` directory is still sitting in
`apa-uniform-sum`). Failing at config-load costs a second. Failing at checkpoint-write costs the
run.

Modules that read `os.environ["TOFU_*"]` at import call `repo_env.ensure_site_env()` first, which
sources the site file once. Without it, `import memadapt_common` from a plain shell dies with a
bare `KeyError` naming a variable the reader has never heard of.

## 6. What still needs the original cluster

Nothing in the *code*. But the adapter pools are 674 GB for `tofu_sisa_lora` alone and are not in
git. [`STATUS.md`](STATUS.md) says, per result, whether it is verifiable from
`results_snapshot/` with no GPU or needs a retrain, and what the retrain costs.
