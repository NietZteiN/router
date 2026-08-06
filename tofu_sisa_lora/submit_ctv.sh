#!/bin/bash
# composable_tv (ctv) Wave-0 SLURM driver
# (log/composable_tv/2026-07-16_thread-preregistration.md; arms ctrl|lin|wd|ds + the [w5]
# post-hoc sparsification wave).
#
# Usage: bash submit_ctv.sh CONFIG [gate|prep|train|train_unc|verify|merge|eval|w5_build|collect|all]
#
#   gate     (login, CPU — allowed) run the arm's CPU test suite + the shared
#            test_verify_subtraction.py; writes ${out_dir}/ctv_state/gate_ok on green.
#            Every sbatch stage REFUSES to submit without a green stamp (STUB previews exempt).
#   prep     (login, light) copy the forget_quality KS reference (cfg.retain_tr_source) into
#            ${out_dir}/results/<cap>/ (COPY not symlink — the sift relative-symlink lesson)
#            + write eval_manifest_<config-basename>.txt / merge_manifest_<config-basename>.txt
#            (basename-keyed so configs sharing an out_dir never collide) from the
#            runtime-derived author pool (merge_subset.subset_authors/probe_authors — never
#            hardcoded).
#   train    GPU array over the arm pool (pool_size tasks; wd = pool_size x 2 variants),
#            1 GPU / 00:45:00 each, sprint1-3 only; self-skips existing shard dirs.
#   train_unc (arm ds only) 5-task GPU array: train_ds_support --no_support on the probe
#            authors — the H-ds-1 unconstrained comparator bakes (self-skips baked dirs).
#   verify   1 CPU no-gres job: verify_subtraction.py (+ verify_struct.py [wd] / ds locality
#            probe [ds]). Gates merge: on failure it scancels the dependent merge+eval jobs
#            itself (kill_invalid_depend is OFF cluster-wide — submit_scale_grid precedent).
#   merge    CPU no-gres array: materialize the N-ladder merges per scale_conditions
#            (ctrl/wd/lin[serve_mode=standard] -> merge_subset.py additive_sum|additive_mean
#            on a DERIVED per-variant merge config; lin[linear serve] + ds materialize
#            NOTHING — their rows serve in-place); self-skips existing merges/<label>/ dirs.
#   eval     GPU array over the eval manifest (label \t serve-spec \t sid \t rids); 1B rows
#            00:45:00, [lin] linearized-serve / 7B rows 01:30:00; self-skips existing JSONs.
#   w5_build ONE CPU no-gres job (64G): sparsify_pool.py --config <sparsify_7b.json> --dx1 --dx2.
#   collect  (login, light) collect_results.py + analyze_ctv.py.
#   all      gate -> prep -> train -> verify(afterok) -> merge(afterok) -> eval(afterok)
#            (arm w5: gate -> w5_build).
#
# GPU-CAP GUARD (hard rule, ~/CLAUDE.md §1): before EVERY sbatch the driver runs
# `squeue -u jack -o "%.10i %.20j %.10T %.10b %F"`, computes the worst-case concurrent GPUs
# of everything already queued (per array: min(%N throttle, task count) x gpus/task) and
# REFUSES to submit if existing + this submission's throttle > 4, printing the arithmetic.
# Default array throttle %2 (ARRAY_CAP env override, hard-capped at 4). STUB=1 prints every
# sbatch script without submitting and skips the squeue guard.
#
# ── ARM TOOL INTERFACE CONTRACT (verified against each tool's REAL argparse; every
#    remaining mismatch fails loudly inside the SLURM task) ──
#   trainers (1 GPU task / author):
#     ctrl|wd -> train_struct_tv.py --config C --author A --arm <control|orthblock|rowslice>
#                (adapters at ${out_dir}/<arm>/shard_<A>/ — the control arm INCLUDED)
#     lin  -> train_linear_tv.py  --config C --author A      (${out_dir}/shard_<A>/)
#     ds   -> train_ds_support.py --config C --author A [--density d]
#             (${out_dir}/ds/tau_a<A>[_d<density>]/{tau_sparse.pt,meta.json})
#   mergers (CPU; arms ctrl|wd|lin[serve_mode=standard] ONLY):
#     merge_subset.py merge --config ${out_dir}/merge_cfg_<v|default>.json
#       --method additive_<sum|mean> --n N --seed S      (no --variant flag exists).
#     The DERIVED config satisfies merge_subset.load_config (model_name, shards_dir =
#     ${out_dir}[/<variant>], out_dir = ${out_dir}/mtmp_<v|default>, n_ladder,
#     subset_seeds=[pool_seed], eval); the tool writes mtmp_<v>/merges/<merge_label(...)>
#     and the driver symlinks ${out_dir}/merges/<ctv label> -> that ABSOLUTE path.
#     lin[linear serve] and ds materialize NOTHING — their rows serve in-place (below).
#   verifiers (CPU, real interfaces): verify_subtraction.py --merged D --remerged D --tau D
#     [--declared_class]; verify_struct.py --config C --arm <control|orthblock|rowslice>;
#     ds_support.py locality --config C (nonzero exit = gate failure). VERIFY_EXTRA
#     overrides the per-arm extra command.
#   eval serve-specs (manifest col 2):
#     <dir>                                      -> eval_tofu --preloaded_adapter <dir>
#     lin:authors=<ids> | lin:n=<N>[,sub=<ids>]  -> eval_tofu --linear_tv_config C +
#         --linear_tv_authors / --linear_tv_n [--linear_tv_subtract] (NEVER combined
#         with --preloaded_adapter — that would silently serve the nonlinear adapter)
#     ds:authors=<ids>  | ds:n=<N>[,sub=<ids>]   -> eval_tofu --ds_config C + --ds_*
#     model:<dir> -> eval_baseline on a baked full-model dir (ds headline bakes)
#     BASE        -> eval_baseline on the base model (floor rows).
#
# Env overrides: ARRAY_CAP (GPU array throttle, default 2, never >4) | TRAIN_TIME (00:45:00) |
#   EVAL_TIME (default 00:45:00; 03:00:00 for arm=lin, 01:30:00 for 7B models) | MERGE_TIME (02:00:00) |
#   MERGE_MEM (32G) | VERIFY_TIME (02:00:00) | VERIFY_MEM (32G) | W5_TIME (12:00:00) |
#   W5_MEM (64G) | TRAIN_ARRAY / TRAIN_UNC_ARRAY / MERGE_ARRAY / EVAL_ARRAY (explicit --array
#   specs; keep any %N <= ARRAY_CAP) | EVAL_MANIFEST (ad-hoc rows) | VERIFY_EXTRA | CTV_GATE_TESTS
set -euo pipefail

CONFIG="$(readlink -f "${1:?config path required}")"
STAGE="${2:-all}"
[ -f "${CONFIG}" ] || { echo "[submit_ctv] config not found: ${CONFIG}" >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm_nodes.sh
source "${SCRIPT_DIR}/slurm_nodes.sh"
PYTHON="${TOFU_PYTHON:-python3}"
HF_HOME="${HF_HOME:?set HF_HOME, or source the site layer (see PORTING.md)}"

ARRAY_CAP="${ARRAY_CAP:-2}"   # %2 default while Exp-6/anchor arrays hold cap share
if ! [[ "${ARRAY_CAP}" =~ ^[0-9]+$ ]] || [ "${ARRAY_CAP}" -lt 1 ] \
   || [ "${ARRAY_CAP}" -gt 4 ] || [ "${ARRAY_CAP}" -gt "${TOFU_ARRAY_CAP}" ]; then
  echo "[submit_ctv] ARRAY_CAP=${ARRAY_CAP} invalid — must be 1..min(4, TOFU_ARRAY_CAP=${TOFU_ARRAY_CAP})" >&2
  exit 1
fi
TRAIN_TIME="${TRAIN_TIME:-00:45:00}"   # straggler margin over the ~30 min median (F4)
MERGE_TIME="${MERGE_TIME:-02:00:00}"
MERGE_MEM="${MERGE_MEM:-32G}"
MERGE_CPUS="${MERGE_CPUS:-8}"
VERIFY_TIME="${VERIFY_TIME:-02:00:00}"
VERIFY_MEM="${VERIFY_MEM:-32G}"
W5_TIME="${W5_TIME:-12:00:00}"
W5_MEM="${W5_MEM:-64G}"

# ── config read (single python pass; '-' marks a missing optional key) ──────────
CFG_VALUES="$("${PYTHON}" - "${CONFIG}" <<'CFGEOF'
import json, sys

try:
    cfg = json.load(open(sys.argv[1]))
except Exception as e:  # unreadable/invalid JSON -> clear refusal
    sys.exit(f"[submit_ctv] cannot read config {sys.argv[1]}: {e}")
if "out_dir" not in cfg:
    sys.exit(f"[submit_ctv] config {sys.argv[1]} missing required key 'out_dir'")

def g(*path, default="-"):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

arm = g("arm")
# Arm defaults (config keys override): wd trains/merges both write-disjoint granularities;
# ctrl is the ONE "control" variant of train_struct_tv (adapters at control/shard_<a>).
variants = cfg.get("variants") or {"wd": ["orthblock", "rowslice"],
                                   "ctrl": ["control"]}.get(arm, [])
# SCALES is informational only here (do_prep re-derives per-variant conditions). The
# dict form {variant: [scales], "extras_at_n8": [...]} normalizes to the sorted unique
# union of its per-variant lists (extras_at_n8 entries are variant_scale pairs, not
# scales — joining dict KEYS was the old bug).
sc = cfg.get("scale_conditions")
if isinstance(sc, dict):
    scales = sorted({s for k, v in sc.items() if k != "extras_at_n8" for s in v})
elif isinstance(sc, list):
    scales = list(sc)
else:
    scales = ["sum", "mean"] if arm == "ctrl" else ["sum"]
vals = [arm, g("model_name"), cfg["out_dir"], g("eval", "k"), g("eval", "forget_shard_id"),
        g("eval", "cap"), g("pool_seed"), g("pool_size"), g("retain_tr_source"),
        ",".join(variants) or "-", ",".join(scales) or "-"]
bad = [str(v) for v in vals if " " in str(v)]
if bad:
    sys.exit(f"[submit_ctv] config values must not contain spaces: {bad}")
print(*[str(v) for v in vals])
CFGEOF
)" || exit 1
read -r ARM MODEL_NAME OUT_DIR K FID CAP POOL_SEED POOL_SIZE RETAIN_TR_SOURCE \
        VARIANTS SCALES <<< "${CFG_VALUES}"

need() {  # need <value> <config key> — clear refusal on a missing key (concurrent configs)
  if [ -z "${1:-}" ] || [ "${1}" = "-" ]; then
    echo "[submit_ctv] config ${CONFIG} is missing key '${2}' (required for stage '${STAGE}')" >&2
    exit 1
  fi
}

case "${OUT_DIR}" in /*) ;; *) OUT_DIR="${SCRIPT_DIR}/${OUT_DIR}" ;; esac
if [ "${RETAIN_TR_SOURCE}" != "-" ]; then
  case "${RETAIN_TR_SOURCE}" in /*) ;; *) RETAIN_TR_SOURCE="${SCRIPT_DIR}/${RETAIN_TR_SOURCE}" ;; esac
fi
RESULTS_DIR="${OUT_DIR}/results/${CAP}"
LOG_DIR="${OUT_DIR}/logs"
STATE="${OUT_DIR}/ctv_state"
# Manifests key on the CONFIG basename so two configs sharing an out_dir (e.g.
# ctv_1b_lin + ctv_1b_lin_nlserve) never clobber each other's rows.
CFG_BASE="$(basename "${CONFIG}" .json)"
EVAL_MANIFEST="${EVAL_MANIFEST:-${OUT_DIR}/eval_manifest_${CFG_BASE}.txt}"
MERGE_MANIFEST="${OUT_DIR}/merge_manifest_${CFG_BASE}.txt"
CAP_FLAG=""
[ "${CAP}" != "-" ] && CAP_FLAG="--${CAP}"
if [ "${VARIANTS}" = "-" ]; then
  NV=1
else
  IFS=',' read -r -a _VARR <<< "${VARIANTS}"
  NV=${#_VARR[@]}
fi
EVAL_TIME_DEFAULT="00:45:00"
[ "${ARM}" = "lin" ] && EVAL_TIME_DEFAULT="03:00:00"   # linearized serve: no KV cache (F4: 01:30 timed out)
case "${MODEL_NAME}" in *7[Bb]*) EVAL_TIME_DEFAULT="01:30:00" ;; esac
EVAL_TIME="${EVAL_TIME:-${EVAL_TIME_DEFAULT}}"

submit() {  # submit <script-text>; honors STUB=1 (preview, nothing reaches sbatch)
  if [ "${STUB:-0}" = "1" ]; then
    echo "----- STUB: sbatch script (not submitted) -----"
    printf '%s\n' "$1"
    echo "-----------------------------------------------"
    LAST_JOB=""
  else
    LAST_JOB="$(printf '%s\n' "$1" | sbatch --parsable)"
    echo "submitted job ${LAST_JOB}"
  fi
}

cap_guard() {  # cap_guard <this submission's worst-case concurrent GPUs>
  local new="$1"
  if [ "${STUB:-0}" = "1" ]; then
    echo "[cap] STUB=1 — squeue guard skipped (this submission would add up to ${new} concurrent GPU(s))"
    return 0
  fi
  "${PYTHON}" - "${new}" <<'CAPEOF'
import re
import subprocess
import sys

NEW = int(sys.argv[1])
CAP = 4  # GLOBAL concurrent-GPU cap across ALL our jobs (~/CLAUDE.md §1)

# The mandated audit view (job ids can truncate at 10 chars — display only).
audit = subprocess.run(["squeue", "-u", "jack", "-o", "%.10i %.20j %.10T %.10b %F"],
                       capture_output=True, text=True)
if audit.returncode != 0:
    sys.exit(f"[cap] squeue failed ({audit.stderr.strip()}) — refusing to submit blind")
print("[cap] squeue -u jack:")
print(audit.stdout.rstrip() or "  (queue empty)")
# Untruncated fields for the arithmetic (the audit view can cut array %N throttles off).
raw = subprocess.run(["squeue", "-u", "jack", "-h", "-o", "%i|%T|%b|%F"],
                     capture_output=True, text=True).stdout

def gpus_of(tres):
    # TRAILING count of an optionally-typed gres: "gres/gpu:1" and "gres/gpu:a40:1"
    # are both 1 GPU (the old r"gpu\D*(\d+)" read the 40 out of the a40 TYPE);
    # a bare "gpu" with no count is 1, no gpu at all is 0.
    m = re.search(r"gpu(?::[A-Za-z][\w-]*)?:(\d+)", tres)
    return int(m.group(1)) if m else (1 if "gpu" in tres else 0)

groups = {}
for ln in raw.splitlines():
    parts = [p.strip() for p in ln.split("|")]
    if len(parts) < 4:
        continue
    jid, state, tres, arr = parts[:4]
    if state in ("COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"):
        continue
    g = gpus_of(tres)
    if g == 0:
        continue
    grp = groups.setdefault(arr or jid.split("_")[0], {"g": 0, "tasks": 0, "cap": None})
    grp["g"] = max(grp["g"], g)
    m = re.search(r"\[([^\]]*)\]", jid)
    if m:  # pending array line, e.g. 0-19%2 or 3,5-9%2: count tasks + read the throttle
        spec = m.group(1)
        c = re.search(r"%(\d+)", spec)
        if c:
            grp["cap"] = int(c.group(1))
        for piece in spec.split("%")[0].split(","):
            if "-" in piece:
                try:
                    a, b = piece.split("-")[:2]
                    grp["tasks"] += int(b) - int(a) + 1
                except ValueError:
                    grp["tasks"] += 1  # unparseable range -> count 1 (throttle still bounds it)
            elif piece:
                grp["tasks"] += 1
    else:
        grp["tasks"] += 1

# Assume every pending job can start NOW: per array, worst case = min(throttle, tasks).
existing = 0
for key, grp in sorted(groups.items()):
    worst = grp["tasks"] if grp["cap"] is None else min(grp["cap"], grp["tasks"])
    print(f"[cap]   job {key}: worst-case {worst} concurrent task(s) x {grp['g']} GPU = {worst * grp['g']}")
    existing += worst * grp["g"]
total = existing + NEW
print(f"[cap] arithmetic: existing worst-case {existing} + this submission {NEW} = {total} (global cap {CAP})")
if total > CAP:
    sys.exit(f"[cap] REFUSING to submit: {total} > {CAP} — wait for the queue to drain or chain with --dependency")
print("[cap] OK")
CAPEOF
}

require_gate() {  # sbatch stages refuse without a green gate stamp (STUB previews exempt)
  if [ "${STUB:-0}" = "1" ]; then
    echo "[gate] STUB=1 preview — gate stamp not enforced"
    return 0
  fi
  if [ ! -f "${STATE}/gate_ok" ]; then
    echo "[gate] REFUSED: no green gate stamp at ${STATE}/gate_ok" >&2
    echo "[gate] run first: bash submit_ctv.sh ${CONFIG} gate" >&2
    exit 1
  fi
}

# ── gate: the arm's CPU suite + the shared subtraction gate, run directly (CPU-only) ──
do_gate() {
  local tests
  case "${ARM}" in
    ctrl) tests="test_merge_subset.py" ;;  # control trains the frozen recipe; its sum/mean composition math lives in merge_subset
    lin)  tests="test_linear_tv.py" ;;
    wd)   tests="test_struct_tv.py" ;;
    ds)   tests="test_ds_support.py" ;;
    w5)   tests="test_sparsify_pool.py" ;;
    *) echo "[gate] unknown arm '${ARM}' in ${CONFIG} (expected ctrl|lin|wd|ds|w5)" >&2; exit 1 ;;
  esac
  tests="${CTV_GATE_TESTS:-${tests} test_verify_subtraction.py}"
  rm -f "${STATE}/gate_ok"
  local t
  for t in ${tests}; do
    if [ ! -f "${SCRIPT_DIR}/${t}" ]; then
      echo "[gate] RED: ${SCRIPT_DIR}/${t} not found (Wave-0 build incomplete?) — later stages refused" >&2
      exit 1
    fi
    echo "[gate] running ${t} ..."
    "${PYTHON}" "${SCRIPT_DIR}/${t}" || { echo "[gate] RED: ${t} failed — later stages refused" >&2; exit 1; }
  done
  (cd "${SCRIPT_DIR}" && sha256sum ${tests}) > "${STATE}/gate_ok"
  echo "[gate] GREEN (${tests}) -> ${STATE}/gate_ok"
}

# ── prep: KS reference copy + manifests derived from the runtime author pool ──────
do_prep() {
  need "${CAP}" "eval.cap"
  mkdir -p "${RESULTS_DIR}"
  if [ "${RETAIN_TR_SOURCE}" != "-" ] && [ -f "${RETAIN_TR_SOURCE}" ]; then
    # COPY, never symlink: a relative-target symlink resolves from the link's dir and
    # breaks -> np.load fails -> forget_quality NaN (the sift-masks driver lesson).
    cp -f "${RETAIN_TR_SOURCE}" "${RESULTS_DIR}/retain_tr_scores.npy"
    echo "[prep] KS reference ${RETAIN_TR_SOURCE} -> ${RESULTS_DIR}/retain_tr_scores.npy"
  else
    echo "[prep] WARN: retain_tr_source '${RETAIN_TR_SOURCE}' missing -> forget_quality will be NaN"
  fi
  "${PYTHON}" - "${CONFIG}" "${OUT_DIR}" "${SCRIPT_DIR}" "${EVAL_MANIFEST}" "${MERGE_MANIFEST}" <<'PREPEOF'
import json
import os
import sys

cfg_path, out_dir, script_dir, eval_manifest, merge_manifest = sys.argv[1:6]
sys.path.insert(0, script_dir)
cfg = json.load(open(cfg_path))
missing = [k for k in ("arm", "pool_seed", "pool_size", "n_ladder") if k not in cfg]
if missing:
    sys.exit(f"[prep] config {cfg_path} missing required key(s) for manifests: {', '.join(missing)}")
# Author pools derive at runtime from the seed permutation — never hardcoded.
from merge_subset import probe_authors, subset_authors

arm = cfg["arm"]
seed = int(cfg["pool_seed"])
pool_size = int(cfg["pool_size"])
pool = [int(a) for a in subset_authors(seed, pool_size)]
n_probes = len(cfg.get("probe_authors") or []) or 5
probes = [int(a) for a in probe_authors(seed, pool_size, n_probes)]
declared = [int(a) for a in (cfg.get("probe_authors") or [])]
if declared and declared != probes:
    sys.exit(f"[prep] config probe_authors {declared} != derived {probes} "
             f"(probe_authors(seed={seed})) — fix the config, do not hardcode pools")
ladder = [int(n) for n in cfg["n_ladder"]]
over = [n for n in ladder if n > pool_size]
if over:
    sys.exit(f"[prep] n_ladder entries {over} exceed pool_size {pool_size}")
# variants: config override, else arm defaults. ctrl is ONE "control" variant (its
# adapters land at control/shard_<a> like the wd layout); lin/ds have none.
variants = [v or "" for v in (cfg.get("variants")
                              or {"wd": ["orthblock", "rowslice"],
                                  "ctrl": ["control"]}.get(arm, [""]))]
real_variants = [v for v in variants if v]

# scale conditions: dict form maps variant -> scales and carries the optional
# "extras_at_n8" cross-checks ("orthblock_mean" -> (variant, scale)); list form
# applies to every variant; absent -> arm defaults.
sc = cfg.get("scale_conditions")
extras_n8 = []
if isinstance(sc, dict):
    extras_n8 = [tuple(e.rsplit("_", 1)) for e in (sc.get("extras_at_n8") or [])]
    def scales_for(v):
        return list(sc.get(v or "control", ["sum"]))
elif isinstance(sc, list):
    def scales_for(v):
        return list(sc)
else:
    def scales_for(v):
        return ["sum", "mean"] if arm == "ctrl" else ["sum"]
all_scales = sorted({s for v in variants for s in scales_for(v)}
                    | {s for _, s in extras_n8})
bad = [s for s in all_scales if s not in ("sum", "mean")]
if bad:
    sys.exit(f"[prep] unknown scale_conditions {bad} (only sum|mean)")
bad_v = [v for v, _ in extras_n8 if v not in real_variants]
if bad_v:
    sys.exit(f"[prep] extras_at_n8 names unknown variants {bad_v} (variants={real_variants})")
if extras_n8 and pool_size < 8:
    sys.exit(f"[prep] extras_at_n8 needs pool_size >= 8 (got {pool_size})")

# serve mode: lin serves LINEARIZED in-place (lin:...) unless serve_mode=standard
# (H-lin-2b: the same tangent-trained shards under plain NONLINEAR PEFT serving);
# ds always serves the merged full model in-place (ds:...). ctrl/wd serve dirs.
lin_standard = (arm == "lin" and cfg.get("serve_mode") == "standard")
iso_pattern = cfg.get("iso_pattern",
                      "{v}/shard_{author}" if real_variants else "shard_{author}")

def shard_path(a, v):
    return os.path.join(out_dir, iso_pattern.format(
        v=v or "", author=a, vsuf=f"_{v}" if v else ""))

def vtok(v):
    # ctrl drops the variant token entirely: labels ctv_ctrl_<scale>_N<n>_s<seed>,
    # iso rows iso_a<a> (the config _comment contract); wd/lin keep it.
    return "" if arm == "ctrl" else (v or "")

def iso_serve(a, v):
    if arm == "ds":
        return f"ds:authors={a}"
    if arm == "lin" and not lin_standard:
        return f"lin:authors={a}"
    return shard_path(a, v)

eval_rows, merge_rows = [], []
# G1 solo rows: every pool adapter on its OWN author (sid=a) with retain_prob restricted
# to that author (rids=a) -> iso_a<a>__own<a>.json carries both own recall and own_prob.
for v in variants:
    for a in pool:
        label = f"iso_a{a}" + (f"_{vtok(v)}" if vtok(v) else "")
        eval_rows.append((label, iso_serve(a, v), str(a), str(a)))
# Per-probe base floors (extractable-fraction denominator anchors).
for p in probes:
    eval_rows.append(("base_model", "BASE", str(p), str(p)))

emitted = set()
def emit_ladder_point(v, s, n):
    label = "_".join(t for t in ("ctv", arm, vtok(v), s, f"N{n}", f"s{seed}") if t)
    if label in emitted:      # extras_at_n8 duplicating a normal ladder point
        return
    emitted.add(label)
    if arm == "ds":
        serve = f"ds:n={n}"                       # in-place merged serve (incl. N=1)
    elif arm == "lin" and not lin_standard:
        serve = f"lin:n={n}"                      # linearized serve (incl. N=1)
    elif n == 1:                                  # N=1 == the raw solo adapter of perm[0]
        serve = shard_path(pool[0], v)
    else:
        serve = os.path.join(out_dir, "merges", label)
        merge_rows.append((v or "-", s, str(n), label))
    eval_rows.append((label, serve, "-", "-"))    # global mu/retain row
    eval_rows.append((label, serve, "-",
                      ",".join(str(a) for a in subset_authors(seed, n))))  # __subset
    for p in probe_authors(seed, n, n_probes):    # probes = the permutation head — in every nested subset
        eval_rows.append((label, serve, str(int(p)), str(int(p))))

for v in variants:
    for s in scales_for(v):                       # per-VARIANT ladder, not one global list
        for n in ladder:
            emit_ladder_point(v, s, n)
for v, s in extras_n8:                            # the n=8 scale cross-checks (wd)
    emit_ladder_point(v, s, 8)

# [ds] H-ds-1 denominator: per-probe UNCONSTRAINED full-FT comparators (train_ds_support
# --no_support bakes theta0+tau to ds_unconstrained/a<p>_model; trained via the
# train_unc stage; served through the model: eval_baseline path, sid = rids = probe).
if arm == "ds":
    for p in probes:
        eval_rows.append((f"iso_dsunc_a{p}",
                          "model:" + os.path.join(out_dir, "ds_unconstrained", f"a{p}_model"),
                          str(p), str(p)))

os.makedirs(out_dir, exist_ok=True)
with open(eval_manifest, "w") as f:
    for r in eval_rows:
        f.write("\t".join(r) + "\n")
with open(merge_manifest, "w") as f:
    for r in merge_rows:
        f.write("\t".join(r) + "\n")
print(f"[prep] pool({pool_size}) = {pool}")
print(f"[prep] probes = {probes}; variants = {[v or '-' for v in variants]}; "
      f"scales = {{{', '.join(f'{v or chr(45)}: {scales_for(v)}' for v in variants)}}}"
      f"{f'; extras_at_n8 = {extras_n8}' if extras_n8 else ''}")
print(f"[prep] wrote {eval_manifest} ({len(eval_rows)} rows) + {merge_manifest} ({len(merge_rows)} merges)")
PREPEOF
}

# ── train: GPU array over the pool (author = permutation[i % pool_size]) ──────────
do_train() {
  require_gate
  need "${MODEL_NAME}" "model_name"
  need "${POOL_SEED}" "pool_seed"
  need "${POOL_SIZE}" "pool_size"
  local train_cmd skip_block
  case "${ARM}" in
    # ctrl trains via train_struct_tv --arm control (the frozen recipe, no callback);
    # its config sets variants=["control"], so the shared VARIANT template applies.
    ctrl|wd) train_cmd="${PYTHON} \"${SCRIPT_DIR}/train_struct_tv.py\" --config \"${CONFIG}\" --author \"\${AUTHOR}\" --arm \"\${VARIANT}\"" ;;
    lin)     train_cmd="${PYTHON} \"${SCRIPT_DIR}/train_linear_tv.py\" --config \"${CONFIG}\" --author \"\${AUTHOR}\"" ;;
    ds)      train_cmd="${PYTHON} \"${SCRIPT_DIR}/train_ds_support.py\" --config \"${CONFIG}\" --author \"\${AUTHOR}\"" ;;
    *) echo "[train] no trainer for arm '${ARM}'" >&2; exit 1 ;;
  esac
  # self-skip paths match each trainer's REAL artifact layout
  case "${ARM}" in
    ds)
      read -r -d '' skip_block <<SKIPEOF || true
if [ -f "${OUT_DIR}/ds/tau_a\${AUTHOR}/tau_sparse.pt" ]; then
  echo "Skip existing ${OUT_DIR}/ds/tau_a\${AUTHOR}"; exit 0
fi
SKIPEOF
      ;;
    *)
      local done_dir="${OUT_DIR}/shard_\${AUTHOR}"
      case "${ARM}" in ctrl|wd) done_dir="${OUT_DIR}/\${VARIANT}/shard_\${AUTHOR}" ;; esac
      read -r -d '' skip_block <<SKIPEOF || true
DONE="${done_dir}"
if [ -f "\${DONE}/adapter_model.safetensors" ] || [ -f "\${DONE}/adapter_config.json" ]; then
  echo "Skip existing \${DONE}"; exit 0
fi
SKIPEOF
      ;;
  esac
  local n_tasks=$((POOL_SIZE * NV))
  local spec="${TRAIN_ARRAY:-0-$((n_tasks - 1))%${ARRAY_CAP}}"
  local mine=$(( n_tasks < ARRAY_CAP ? n_tasks : ARRAY_CAP ))
  cap_guard "${mine}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-${ARM}-train
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=${TRAIN_TIME}
#SBATCH --output=${LOG_DIR}/train_%A_%a.log
#SBATCH --error=${LOG_DIR}/train_%A_%a.log
set -eo pipefail   # F2: never mask a failing trainer/derivation inside the job

# author index cycles fastest; the variant (wd/ctrl) advances every ${POOL_SIZE} tasks
AUTHOR=\$(${PYTHON} -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from merge_subset import subset_authors; print(int(subset_authors(${POOL_SEED}, ${POOL_SIZE})[\${SLURM_ARRAY_TASK_ID} % ${POOL_SIZE}]))")
VARIANT=""
if [ "${VARIANTS}" != "-" ]; then
  IFS=',' read -r -a VARR <<< "${VARIANTS}"
  VARIANT="\${VARR[\$((SLURM_ARRAY_TASK_ID / ${POOL_SIZE}))]}"
fi
${skip_block}
echo "=== ctv ${ARM} train task \${SLURM_ARRAY_TASK_ID}: author \${AUTHOR} variant \${VARIANT:--} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
# optional token read: genuinely non-fatal, guard it under set -e
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token" || true)"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${train_cmd}
date
EOF
  echo "train array: ${n_tasks} tasks (spec ${spec}), ${TRAIN_TIME}/task, exclude ${TOFU_EXCLUDE}"
  submit "${S}"
  TRAIN_JOB="${LAST_JOB:-}"
  if [ "${ARM}" = "ds" ]; then
    echo "[train] arm ds: H-ds-1 unconstrained comparators (5 probe tasks) launch separately:"
    echo "[train]   bash submit_ctv.sh ${CONFIG} train_unc"
  fi
}

# ── train_unc (arm ds only): the H-ds-1 unconstrained comparator — 5-task GPU array
#    running train_ds_support --no_support on the probe authors; each task bakes
#    theta0+tau to ${OUT_DIR}/ds_unconstrained/a<p>_model (no tau_sparse.pt), served
#    by the eval stage's iso_dsunc_a<p> model: rows. ────────────────────────────────
do_train_unc() {
  require_gate
  if [ "${ARM}" != "ds" ]; then
    echo "[train_unc] arm '${ARM}' has no unconstrained comparator (ds only)" >&2
    exit 1
  fi
  need "${MODEL_NAME}" "model_name"
  need "${POOL_SEED}" "pool_seed"
  need "${POOL_SIZE}" "pool_size"
  local n_tasks=5
  local spec="${TRAIN_UNC_ARRAY:-0-$((n_tasks - 1))%${ARRAY_CAP}}"
  local mine=$(( n_tasks < ARRAY_CAP ? n_tasks : ARRAY_CAP ))
  cap_guard "${mine}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-${ARM}-trainunc
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=${TRAIN_TIME}
#SBATCH --output=${LOG_DIR}/trainunc_%A_%a.log
#SBATCH --error=${LOG_DIR}/trainunc_%A_%a.log
set -eo pipefail   # F2: never mask a failing trainer inside the job

# probes derive at runtime (merge_subset.probe_authors — never hardcoded)
AUTHOR=\$(${PYTHON} -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from merge_subset import probe_authors; print(int(probe_authors(${POOL_SEED}, ${POOL_SIZE}, 5)[\${SLURM_ARRAY_TASK_ID}]))")
DONE="${OUT_DIR}/ds_unconstrained/a\${AUTHOR}_model"
if [ -f "\${DONE}/config.json" ] && [ -f "\${DONE}/meta.json" ]; then
  echo "Skip existing \${DONE}"; exit 0
fi
echo "=== ctv ${ARM} train_unc task \${SLURM_ARRAY_TASK_ID}: author \${AUTHOR} (H-ds-1 comparator) ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
# optional token read: genuinely non-fatal, guard it under set -e
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token" || true)"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

${PYTHON} "${SCRIPT_DIR}/train_ds_support.py" --config "${CONFIG}" --author "\${AUTHOR}" --no_support
date
EOF
  echo "train_unc array: ${n_tasks} probe tasks (spec ${spec}), ${TRAIN_TIME}/task, exclude ${TOFU_EXCLUDE}"
  submit "${S}"
  TRAIN_UNC_JOB="${LAST_JOB:-}"
}

# ── verify: CPU exactness/locality gate; scancels its dependents on failure ───────
do_verify() {
  require_gate
  local dep="${1:-}" dep_line=""
  [ -n "${dep}" ] && dep_line="#SBATCH --dependency=afterok:${dep}"
  # Post-TRAIN structural/locality gate (pre-merge). verify_subtraction.py is NOT run
  # here: it compares materialized merged/remerged/tau triples, which only exist after
  # the merge stage — it is the G3 exactness certificate, invoked ad-hoc on the triples
  # (see the header contract). Arm rules: wd gates on both variants' own-energy
  # certificates; ctrl runs verify_struct report-only (--arm control never gates);
  # ds gates on the support-locality audit; lin has no structural invariant to verify
  # (tangent factors are unconstrained) — the stage is skipped and merge chains
  # directly behind train.
  local extra=""
  case "${ARM}" in
    ctrl) extra="${PYTHON} \"${SCRIPT_DIR}/verify_struct.py\" --config \"${CONFIG}\" --arm control" ;;
    wd) extra="${PYTHON} \"${SCRIPT_DIR}/verify_struct.py\" --config \"${CONFIG}\" --arm orthblock || fail
${PYTHON} \"${SCRIPT_DIR}/verify_struct.py\" --config \"${CONFIG}\" --arm rowslice" ;;
    ds) extra="${PYTHON} \"${SCRIPT_DIR}/ds_support.py\" locality --config \"${CONFIG}\"" ;;
    lin)
      echo "[verify] arm lin: no structural invariant to verify post-train — stage skipped"
      VERIFY_JOB="${dep}"   # pass the train dep through so merge still chains behind train
      return 0
      ;;
  esac
  extra="${VERIFY_EXTRA:-${extra}}"
  cap_guard 0
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-${ARM}-verify
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --mem=${VERIFY_MEM}
#SBATCH --cpus-per-task=8
#SBATCH --time=${VERIFY_TIME}
#SBATCH --output=${LOG_DIR}/verify_%j.log
#SBATCH --error=${LOG_DIR}/verify_%j.log
${dep_line}
set -eo pipefail   # F2: verified safe — every arm extra routes failures through || fail
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
# kill_invalid_depend is OFF cluster-wide: a failed gate must scancel its afterok
# dependents itself or they hang pending forever (submit_scale_grid precedent).
fail() {
  echo "[verify] FAILED — scancelling dependent merge/eval jobs (if chained)"
  for f in merge_jobid.txt eval_jobid.txt; do
    { [ -f "${STATE}/\${f}" ] && scancel "\$(cat "${STATE}/\${f}")"; } || true
  done
  exit 1
}
# set -e alone would exit WITHOUT scancelling on a failure outside the || fail chain
# (e.g. a custom multi-command VERIFY_EXTRA) — route every untrapped failure to fail.
trap fail ERR
date
${extra:+${extra} || fail}
date
EOF
  echo "verify: 1 CPU task (${VERIFY_MEM}); gates merge via afterok + scancel-on-failure"
  submit "${S}"
  VERIFY_JOB="${LAST_JOB:-}"
}

# ── merge: CPU array materializing the N-ladder per scale condition ───────────────
do_merge() {
  require_gate
  need "${POOL_SEED}" "pool_seed"
  if [ "${ARM}" = "ds" ]; then
    echo "[merge] arm ds materializes no merges — ladder rows serve the merged model"
    echo "[merge] in-place via eval_tofu --ds_config (ds:n=<N> serve-specs); headline"
    echo "[merge] bakes go through: ${PYTHON} ${SCRIPT_DIR}/ds_support.py bake --config ${CONFIG} --n N --out DIR"
    return 0
  fi
  [ -f "${MERGE_MANIFEST}" ] || { echo "[merge] missing ${MERGE_MANIFEST} — run: bash submit_ctv.sh ${CONFIG} prep" >&2; exit 1; }
  local n_tasks; n_tasks=$(wc -l < "${MERGE_MANIFEST}")
  if [ "${n_tasks}" -eq 0 ]; then
    echo "[merge] empty merge manifest (lin linear-serve rows compose in-place; or ladder all N=1) — nothing to do"
    return 0
  fi
  # merge_subset.load_config REQUIRES {model_name, shards_dir, out_dir, n_ladder,
  # subset_seeds, eval} — the ctv config satisfies none of the layout keys, so derive
  # one tiny merge config per variant: shards_dir = the variant's shard layout
  # (lin -> flat ${OUT_DIR}; ctrl/wd -> ${OUT_DIR}/<variant>), out_dir = an mtmp_<v>
  # sandbox whose merges/ the driver symlinks the ctv labels into.
  "${PYTHON}" - "${CONFIG}" "${OUT_DIR}" "${VARIANTS}" "${ARM}" <<'MCFGEOF' || exit 1
import json
import os
import sys

cfg_path, out_dir, variants_csv, arm = sys.argv[1:5]
cfg = json.load(open(cfg_path))
missing = [k for k in ("model_name", "n_ladder", "pool_seed", "eval") if k not in cfg]
if missing:
    sys.exit(f"[merge] config {cfg_path} missing key(s) for the derived merge configs: {missing}")
variants = [v for v in variants_csv.split(",") if v and v != "-"] or [""]
for v in variants:
    tag = v or "default"
    shards = out_dir if arm == "lin" else os.path.join(out_dir, v or "control")
    mcfg = {
        "model_name": cfg["model_name"],
        "shards_dir": shards,
        "out_dir": os.path.join(out_dir, f"mtmp_{tag}"),
        "n_ladder": cfg["n_ladder"],
        "subset_seeds": [int(cfg["pool_seed"])],
        "eval": cfg["eval"],
    }
    path = os.path.join(out_dir, f"merge_cfg_{tag}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mcfg, f, indent=2)
    print(f"[merge] derived merge config {path} (shards_dir={shards})")
MCFGEOF
  local dep="${1:-}" dep_line=""
  [ -n "${dep}" ] && dep_line="#SBATCH --dependency=afterok:${dep}"
  # merge_subset.py has NO --variant flag: the variant is encoded in the derived
  # config's shards_dir, selected per row via the manifest's V column.
  local merge_cmd="${PYTHON} \"${SCRIPT_DIR}/merge_subset.py\" merge --config \"\${MCFG}\" --method \"additive_\${SCALE}\" --n \"\${N}\" --seed ${POOL_SEED}"
  local spec="${MERGE_ARRAY:-0-$((n_tasks - 1))%4}"   # CPU array — outside the GPU cap
  cap_guard 0
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-${ARM}-merge
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --mem=${MERGE_MEM}
#SBATCH --cpus-per-task=${MERGE_CPUS}
#SBATCH --time=${MERGE_TIME}
#SBATCH --output=${LOG_DIR}/merge_%A_%a.log
#SBATCH --error=${LOG_DIR}/merge_%A_%a.log
${dep_line}
set -eo pipefail   # F2: a failed merge_subset/symlink must fail the task, not be masked

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MERGE_MANIFEST}")
V=\$(printf '%s' "\${LINE}" | cut -f1)
SCALE=\$(printf '%s' "\${LINE}" | cut -f2)
N=\$(printf '%s' "\${LINE}" | cut -f3)
LABEL=\$(printf '%s' "\${LINE}" | cut -f4)
MTAG="\${V}"; [ "\${V}" = "-" ] && MTAG="default"
MCFG="${OUT_DIR}/merge_cfg_\${MTAG}.json"
[ -f "\${MCFG}" ] || { echo "[merge] missing derived merge config \${MCFG} — re-run: bash submit_ctv.sh ${CONFIG} merge"; exit 1; }
DEST="${OUT_DIR}/merges/\${LABEL}"
if [ -e "\${DEST}/adapter_config.json" ] || [ -e "\${DEST}/config.json" ]; then
  echo "Skip existing \${DEST}"; exit 0
fi
echo "=== ctv ${ARM} merge task \${SLURM_ARRAY_TASK_ID}: \${LABEL} (variant=\${MTAG} scale=\${SCALE} N=\${N}) ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export NMERGE_THREADS=${MERGE_CPUS}
${merge_cmd}
# merge_subset writes its own nmerge_* label under mtmp_<v>/merges — link the ctv
# label to it (ABSOLUTE target: a relative symlink would resolve from merges/ and
# break, the sift relative-symlink lesson).
if [ ! -e "\${DEST}" ]; then
  ${PYTHON} - "${SCRIPT_DIR}" "${OUT_DIR}" "\${MTAG}" "\${LABEL}" "\${SCALE}" "\${N}" "${POOL_SEED}" <<'LINKEOF'
import os
import sys
script_dir, out_dir, mtag, label, scale, n, seed = sys.argv[1:8]
dest = os.path.join(out_dir, "merges", label)
if os.path.exists(dest):
    sys.exit(0)
sys.path.insert(0, script_dir)
mtmp_merges = os.path.join(out_dir, f"mtmp_{mtag}", "merges")
cands = []
try:
    from merge_subset import merge_label       # the exact tool-side name
    cands.append(merge_label(f"additive_{scale}", int(n), int(seed)))
except Exception:
    pass
# fallback guess if merge_subset is unimportable (matches merge_label today)
cands.append("nmerge_" + ("sum" if scale == "sum" else "add") + f"_N{n}_s{seed}")
os.makedirs(os.path.join(out_dir, "merges"), exist_ok=True)
for c in cands:
    target = os.path.join(mtmp_merges, c)
    if os.path.isdir(target):
        os.symlink(os.path.abspath(target), dest)
        print("[merge] linked", dest, "->", os.path.abspath(target))
        break
LINKEOF
fi
[ -e "\${DEST}" ] || { echo "[merge] ERROR: tool ran but \${DEST} was not produced — align the arm tool output dir (see the interface contract in submit_ctv.sh)"; exit 1; }
date
EOF
  echo "merge array: ${n_tasks} tasks (spec ${spec}), mem ${MERGE_MEM}, time ${MERGE_TIME}"
  submit "${S}"
  MERGE_JOB="${LAST_JOB:-}"
  [ -n "${MERGE_JOB}" ] && echo "${MERGE_JOB}" > "${STATE}/merge_jobid.txt"
  return 0
}

# ── eval: GPU array over the eval manifest ────────────────────────────────────────
do_eval() {
  require_gate
  need "${MODEL_NAME}" "model_name"
  need "${K}" "eval.k"
  need "${FID}" "eval.forget_shard_id"
  need "${CAP}" "eval.cap"
  [ -f "${EVAL_MANIFEST}" ] || { echo "[eval] missing ${EVAL_MANIFEST} — run: bash submit_ctv.sh ${CONFIG} prep" >&2; exit 1; }
  local dep="${1:-}" dep_line=""
  [ -n "${dep}" ] && dep_line="#SBATCH --dependency=afterok:${dep}"
  local n_tasks; n_tasks=$(wc -l < "${EVAL_MANIFEST}")
  local spec="${EVAL_ARRAY:-0-$((n_tasks - 1))%${ARRAY_CAP}}"
  local mine=$(( n_tasks < ARRAY_CAP ? n_tasks : ARRAY_CAP ))
  cap_guard "${mine}"
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-${ARM}-eval
#SBATCH --array=${spec}
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=${EVAL_TIME}
#SBATCH --output=${LOG_DIR}/eval_%A_%a.log
#SBATCH --error=${LOG_DIR}/eval_%A_%a.log
${dep_line}
set -eo pipefail   # F2: a crashed eval_tofu/eval_baseline must FAIL the task loudly

LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${EVAL_MANIFEST}")
LABEL=\$(printf '%s' "\${LINE}" | cut -f1)
SERVE=\$(printf '%s' "\${LINE}" | cut -f2)
SID=\$(printf '%s' "\${LINE}" | cut -f3)
RIDS=\$(printf '%s' "\${LINE}" | cut -f4)
RIDS="\${RIDS:--}"

if [ "\${SID}" = "-" ]; then
  OUT_JSON="${RESULTS_DIR}/\${LABEL}.json"; SID_ARGS=""
else
  # probe row: forget_* remapped to the probe author's own rows
  OUT_JSON="${RESULTS_DIR}/\${LABEL}__own\${SID}.json"; SID_ARGS="--eval_shard_id \${SID}"
fi
RID_ARGS=""
if [ "\${RIDS}" != "-" ]; then
  RID_ARGS="--retain_author_ids \${RIDS}"
  # rids-only rows are subset-conditioned utility (__subset); probe rows keep the __own
  # name — there rids == the probe author, making retain_prob the own answer-prob channel.
  [ "\${SID}" = "-" ] && OUT_JSON="${RESULTS_DIR}/\${LABEL}__subset.json"
fi
if [ -f "\${OUT_JSON}" ]; then echo "Skip existing \${OUT_JSON}"; exit 0; fi

echo "=== ctv ${ARM} eval task \${SLURM_ARRAY_TASK_ID}: \${LABEL} sid=\${SID} rids=\${RIDS:0:40} ==="
date
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export TOFU_METRICS_CACHE="${HF_HOME}/metrics_cache/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\${TOFU_METRICS_CACHE}"
# optional token read: genuinely non-fatal, guard it under set -e
if [ -z "\${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then export HF_TOKEN="\$(tr -d '\n' < "${HF_HOME}/token" || true)"; fi
export HUGGING_FACE_HUB_TOKEN="\${HUGGING_FACE_HUB_TOKEN:-\${HF_TOKEN:-}}"

case "\${SERVE}" in
  BASE)
    # F1: keep the GLOBAL --forget_shard_id and pass --eval_shard_id for probe rows
    # (the old fid<-sid substitution made SID+rids=SID an empty retain pool crash;
    # eval_baseline now forwards --eval_shard_id into evaluate_model like eval_tofu).
    ${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" \\
      --model_name "${MODEL_NAME}" \\
      --output_dir "${OUT_DIR}" \\
      --k ${K} \\
      --forget_shard_id ${FID} \\
      \${SID_ARGS} \\
      --out "\${OUT_JSON}" \\
      --hf_home "${HF_HOME}" \\
      \${RID_ARGS} \\
      ${CAP_FLAG}
    ;;
  model:*)
    # [ds] baked full model (theta0 + sum tau): serve the model dir itself.
    # Same F1 fid/sid convention as BASE — the iso_dsunc_a<p> comparator rows
    # (sid = rids = the probe author) need the global-fid retain exclusion.
    MDIR="\${SERVE#model:}"
    [ -f "\${MDIR}/config.json" ] || { echo "missing baked model \${MDIR}"; exit 1; }
    ${PYTHON} "${SCRIPT_DIR}/eval_baseline.py" \\
      --model_name "\${MDIR}" \\
      --output_dir "${OUT_DIR}" \\
      --k ${K} \\
      --forget_shard_id ${FID} \\
      \${SID_ARGS} \\
      --out "\${OUT_JSON}" \\
      --hf_home "${HF_HOME}" \\
      \${RID_ARGS} \\
      ${CAP_FLAG}
    ;;
  *)
    # in-place serve-specs vs materialized adapter dirs. lin:/ds: rows pass ONLY the
    # arm config + selection flags — NEVER --preloaded_adapter (build_served_model
    # checks the preloaded branch FIRST, so combining them silently serves the
    # nonlinear adapter; eval_tofu now also hard-raises on the combination).
    SERVE_ARGS=""
    case "\${SERVE}" in
      lin:*|ds:*)
        PFX="\${SERVE%%:*}"; SPEC="\${SERVE#*:}"
        if [ "\${PFX}" = "lin" ]; then
          CFG_FLAG="--linear_tv_config"; A_FLAG="--linear_tv_authors"
          N_FLAG="--linear_tv_n"; S_FLAG="--linear_tv_subtract"
        else
          CFG_FLAG="--ds_config"; A_FLAG="--ds_authors"
          N_FLAG="--ds_n"; S_FLAG="--ds_subtract"
        fi
        SERVE_ARGS="\${CFG_FLAG} ${CONFIG}"
        case "\${SPEC}" in
          authors=*)
            SERVE_ARGS="\${SERVE_ARGS} \${A_FLAG} \${SPEC#authors=}"
            ;;
          n=*)
            NARG="\${SPEC#n=}"; SUBARG=""
            case "\${NARG}" in
              *,sub=*) SUBARG="\${NARG#*,sub=}"; NARG="\${NARG%%,*}" ;;
            esac
            SERVE_ARGS="\${SERVE_ARGS} \${N_FLAG} \${NARG}"
            [ -n "\${SUBARG}" ] && SERVE_ARGS="\${SERVE_ARGS} \${S_FLAG} \${SUBARG}"
            ;;
          *)
            echo "unparseable \${PFX}: serve-spec '\${SERVE}'"; exit 1
            ;;
        esac
        ;;
      *)
        { [ -f "\${SERVE}/adapter_model.safetensors" ] || [ -f "\${SERVE}/adapter_config.json" ]; } \\
          || { echo "missing adapter \${SERVE}"; exit 1; }
        SERVE_ARGS="--preloaded_adapter \${SERVE}"
        ;;
    esac
    ${PYTHON} "${SCRIPT_DIR}/eval_tofu.py" \\
      --model_name "${MODEL_NAME}" \\
      --output_dir "${OUT_DIR}" \\
      --label "\${LABEL}" \\
      --k ${K} \\
      --forget_shard_id ${FID} \\
      \${SID_ARGS} \\
      \${RID_ARGS} \\
      \${SERVE_ARGS} \\
      --out "\${OUT_JSON}" \\
      --hf_home "${HF_HOME}" \\
      ${CAP_FLAG}
    ;;
esac
date
EOF
  echo "eval array: ${n_tasks} tasks (spec ${spec}), cap ${ARRAY_CAP}, time ${EVAL_TIME}, exclude ${TOFU_EXCLUDE}"
  submit "${S}"
  EVAL_JOB="${LAST_JOB:-}"
  [ -n "${EVAL_JOB}" ] && echo "${EVAL_JOB}" > "${STATE}/eval_jobid.txt"
  return 0
}

# ── w5_build: ONE CPU job — post-hoc sparsification grid + DX1/DX2 on the 7B pool ──
do_w5() {
  require_gate
  cap_guard 0
  read -r -d '' S <<EOF || true
#!/bin/bash
#SBATCH --job-name=ctv-w5-sparsify
#SBATCH --partition=all
#SBATCH --exclude=${TOFU_EXCLUDE}
#SBATCH --mem=${W5_MEM}
#SBATCH --cpus-per-task=16
#SBATCH --time=${W5_TIME}
#SBATCH --output=${LOG_DIR}/w5_build_%j.log
#SBATCH --error=${LOG_DIR}/w5_build_%j.log
set -eo pipefail   # F2: a failed sparsify grid must fail the job
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"
export NMERGE_THREADS=16
date
${PYTHON} "${SCRIPT_DIR}/sparsify_pool.py" --config "${CONFIG}" --dx1 --dx2
date
EOF
  echo "w5_build: 1 CPU task (${W5_MEM}, 16 threads) — sparsify grid + DX1/DX2 diagnostics"
  submit "${S}"
}

# ── collect: CSV assembly (login, light) ──────────────────────────────────────────
do_collect() {
  need "${CAP}" "eval.cap"
  "${PYTHON}" "${SCRIPT_DIR}/collect_results.py" --root "$(dirname "${OUT_DIR}")" ${CAP_FLAG}
  "${PYTHON}" "${SCRIPT_DIR}/analyze_ctv.py" --config "${CONFIG}"
}

mkdir -p "${LOG_DIR}" "${STATE}"
if [ "${CAP}" != "-" ]; then mkdir -p "${RESULTS_DIR}"; fi

case "${STAGE}" in
  gate)     do_gate ;;
  prep)     do_prep ;;
  train)    do_train ;;
  train_unc) do_train_unc ;;
  verify)   do_verify ;;
  merge)    do_merge ;;
  eval)     do_eval ;;
  w5_build) do_w5 ;;
  collect)  do_collect ;;
  all)
    do_gate
    if [ "${ARM}" = "w5" ]; then
      do_w5
    else
      do_prep
      do_train
      do_verify "${TRAIN_JOB:-}"
      do_merge "${VERIFY_JOB:-}"
      # Arms without materialized merges (lin linear-serve, ds) leave MERGE_JOB unset —
      # fall back down the chain so evals never race the trains.
      do_eval "${MERGE_JOB:-${VERIFY_JOB:-${TRAIN_JOB:-}}}"
      if [ -n "${MERGE_JOB:-}" ]; then
        echo "NOTE: eval depends afterok:${MERGE_JOB} — kill_invalid_depend is OFF cluster-wide;"
        echo "      a verify failure scancels merge+eval via ${STATE}/, but if the MERGE array"
        echo "      itself fails, scancel the pending eval array (${STATE}/eval_jobid.txt) yourself."
      fi
      echo "after results land: bash submit_ctv.sh ${CONFIG} collect"
    fi
    ;;
  *) echo "unknown stage '${STAGE}' (gate|prep|train|train_unc|verify|merge|eval|w5_build|collect|all)" >&2; exit 1 ;;
esac
