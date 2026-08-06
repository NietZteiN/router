"""CPU gate for the composable_tv driver + analyzer (no GPU, no downloads, no SLURM).

Run before any ctv SLURM job: python test_analyze_ctv.py

  1. analyze_ctv.parse_label on every shape of the ctv label grammar (arm x variant x
     scale, iso rows incl. variants, base/anchor rows, malformed labels -> other).
  2. collect_jsons: __own<sid>/__subset filename splitting, corrupt-JSON tolerance.
  3. End-to-end analyze run on a synthetic results tree: extractable fractions and the
     failure tail against HAND-COMPUTED values, iso/plain/__subset joins, missing-floor
     blanks + the --floor_prob fallback, dist-table median/IQR.
  4. submit_ctv.sh: `bash -n` parses; graceful refusal on a config missing keys and on a
     missing gate stamp; STUB=1 previews of prep/train/verify/merge/eval/w5_build on
     self-contained tmp configs (canonical schema; no other Wave-0 files needed) for the
     ctrl / lin / wd / ds / lin-nlserve shapes — manifest contents checked against the
     merge_subset-derived pool (never hardcoded), config-basename-keyed manifest names
     (shared-out_dir collision fix), array sizes, the %2 throttle, per-arm trainer
     dispatch (train_struct_tv --arm for ctrl AND wd), in-place serve-specs (lin:/ds:
     rows never --preloaded_adapter), derived merge configs + self-skip plumbing.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_ctv
from analyze_ctv import collect_jsons, extractable, parse_label

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMIT = os.path.join(SCRIPT_DIR, "submit_ctv.sh")
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Fixture: synthetic results tree + a canonical-schema tmp config
# ---------------------------------------------------------------------------

def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def build_tree(tmp):
    """Synthetic {out_dir}/results/smoke tree with hand-picked metric values."""
    out_dir = os.path.join(tmp, "ckpt", "Llama-3.2-1B-Instruct_ctv_ctrl_r32_e25")
    res = os.path.join(out_dir, "results", "smoke")
    os.makedirs(res)
    # iso anchors (own rows carry own_prob via retain_prob and own recall via forget_*)
    _dump(os.path.join(res, "iso_a82__own82.json"),
          {"retain_prob": 0.9, "forget_rouge": 0.8, "forget_ppl": 3.0,
           "forget_truth_ratio": 0.2, "model_utility": 0.55})
    _dump(os.path.join(res, "iso_a15__own15.json"),
          {"retain_prob": 0.8, "forget_rouge": 0.7, "forget_ppl": 3.5,
           "forget_truth_ratio": 0.25, "model_utility": 0.54})
    _dump(os.path.join(res, "iso_a111__own111.json"),
          {"retain_prob": 0.75, "forget_rouge": 0.65, "forget_ppl": 3.7,
           "forget_truth_ratio": 0.3, "model_utility": 0.53})
    # per-probe base floors (probe 111 deliberately has NO floor -> ef blank)
    _dump(os.path.join(res, "base_model__own82.json"),
          {"retain_prob": 0.1, "forget_rouge": 0.2, "forget_ppl": 20.0})
    _dump(os.path.join(res, "base_model__own15.json"),
          {"retain_prob": 0.1, "forget_rouge": 0.2, "forget_ppl": 21.0})
    # ladder label: plain + __subset + two probe rows (probe 15 is a tail case)
    _dump(os.path.join(res, "ctv_ctrl_sum_N2_s42.json"),
          {"model_utility": 0.5, "retain_prob": 0.7, "forget_quality": 0.31,
           "retain_ppl": 8.0})
    _dump(os.path.join(res, "ctv_ctrl_sum_N2_s42__subset.json"),
          {"retain_prob": 0.65, "model_utility": 0.49})
    _dump(os.path.join(res, "ctv_ctrl_sum_N2_s42__own82.json"),
          {"retain_prob": 0.6, "forget_rouge": 0.55, "forget_ppl": 4.2,
           "forget_truth_ratio": 0.35, "model_utility": 0.5})
    _dump(os.path.join(res, "ctv_ctrl_sum_N2_s42__own15.json"),
          {"retain_prob": 0.2, "forget_rouge": 0.3, "forget_ppl": 9.0,
           "forget_truth_ratio": 0.6, "model_utility": 0.45})
    # a second scale condition and a probe with no floor anchor
    _dump(os.path.join(res, "ctv_ctrl_mean_N2_s42__own82.json"),
          {"retain_prob": 0.5, "forget_rouge": 0.45, "forget_ppl": 5.0,
           "forget_truth_ratio": 0.4, "model_utility": 0.48})
    _dump(os.path.join(res, "ctv_ctrl_sum_N3_s42__own111.json"),
          {"retain_prob": 0.3, "forget_rouge": 0.35, "forget_ppl": 7.0,
           "forget_truth_ratio": 0.5, "model_utility": 0.46})
    # wd-variant rows landing in the same dir exercise the variant grammar end-to-end
    _dump(os.path.join(res, "iso_a82_orthblock__own82.json"),
          {"retain_prob": 0.85, "forget_rouge": 0.75, "forget_ppl": 3.2})
    _dump(os.path.join(res, "ctv_wd_orthblock_sum_N8_s42__own82.json"),
          {"retain_prob": 0.45, "forget_rouge": 0.4, "forget_ppl": 6.0})
    # tolerance: corrupt JSON + a stray unrelated label + a mu-NaN flag row
    with open(os.path.join(res, "ctv_ctrl_sum_N4_s42__own82.json"), "w") as f:
        f.write("{not json")
    _dump(os.path.join(res, "some_other_label.json"), {"model_utility": 0.1})
    _dump(os.path.join(res, "ctv_ctrl_mean_N3_s42__own82.json"),
          {"retain_prob": 0.4, "forget_rouge": 0.35, "forget_ppl": 6.5,
           "model_utility": float("nan"), "retain_ppl": 5000.0})
    return out_dir, res


def write_config(tmp, out_dir, **overrides):
    """The canonical ctv config schema (self-contained; retain_tr_source is a dummy)."""
    ks_src = os.path.join(tmp, "retain_tr_scores.npy")
    with open(ks_src, "wb") as f:
        f.write(b"\x93NUMPY-DUMMY")
    cfg = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "out_dir": out_dir,
        "arm": "ctrl",
        "pool_seed": 42,
        "pool_size": 20,
        "probe_authors": [82, 15, 111, 177, 76],
        "n_ladder": [1, 2, 3],
        "train": {"rank": 32, "alpha": 64, "epochs": 25, "lr": 1e-4,
                  "rslora": True, "seed": 42},
        "eval": {"k": 200, "forget_shard_id": 199, "cap": "smoke"},
        "retain_tr_source": ks_src,
        "unlearn_tags": ["forget10"],
    }
    cfg.update(overrides)
    path = os.path.join(tmp, "ctv_test_config.json")
    _dump(path, cfg)
    return path


# ---------------------------------------------------------------------------
# 1. label grammar
# ---------------------------------------------------------------------------

def test_parse_label():
    cases = {
        "ctv_ctrl_sum_N8_s42": ("merge", "ctrl", "", "sum", 8, 42),
        "ctv_ctrl_mean_N2_s42": ("merge", "ctrl", "", "mean", 2, 42),
        "ctv_lin_sum_N20_s42": ("merge", "lin", "", "sum", 20, 42),
        "ctv_ds_sum_N16_s43": ("merge", "ds", "", "sum", 16, 43),
        "ctv_wd_orthblock_sum_N8_s42": ("merge", "wd", "orthblock", "sum", 8, 42),
        "ctv_wd_rowslice_mean_N16_s42": ("merge", "wd", "rowslice", "mean", 16, 42),
    }
    for label, (kind, arm, variant, scale, n, seed) in cases.items():
        got = parse_label(label)
        assert got["kind"] == kind and got["arm"] == arm and got["variant"] == variant \
            and got["scale"] == scale and got["n"] == n and got["seed"] == seed, (label, got)
    assert parse_label("iso_a82") == {"kind": "iso", "author": 82, "variant": ""}
    assert parse_label("iso_a7_rowslice") == {"kind": "iso", "author": 7,
                                              "variant": "rowslice"}
    assert parse_label("base_model")["kind"] == "floor"
    assert parse_label("ft_r32")["kind"] == "anchor"
    assert parse_label("retain90_oracle")["kind"] == "anchor"
    for bad in ("ctv_bogus_sum_N8_s42", "ctv_ctrl_sum_N8", "ctv_ctrl_N8_s42",
                "ctv_ctrl_sum_Nx_s42", "nmerge_add_N8_s42", "merged_dare_ties", "iso_x1"):
        assert parse_label(bad)["kind"] == "other", bad
    print("parse_label: all label shapes OK")


# ---------------------------------------------------------------------------
# 2. results-tree collection
# ---------------------------------------------------------------------------

def test_collect(res):
    jsons, subset = collect_jsons(res)
    assert ("ctv_ctrl_sum_N2_s42", 82) in jsons
    assert ("ctv_ctrl_sum_N2_s42", None) in jsons
    assert "ctv_ctrl_sum_N2_s42" in subset and subset["ctv_ctrl_sum_N2_s42"] is not None
    assert jsons[("ctv_ctrl_sum_N4_s42", 82)] is None  # corrupt JSON -> None, not a crash
    assert ("iso_a82", 82) in jsons and ("base_model", 15) in jsons
    # missing dir tolerated at the collect level
    empty, esub = collect_jsons(os.path.join(res, "nope"))
    assert empty == {} and esub == {}
    print("collect_jsons: __own/__subset split + corrupt/missing tolerance OK")


# ---------------------------------------------------------------------------
# 3. analyze end-to-end vs hand-computed values
# ---------------------------------------------------------------------------

def _rows_by(rows, **kv):
    return [r for r in rows if all(r[k] == v for k, v in kv.items())]


def _close(a, b, tol=1e-6):
    return a != "" and abs(float(a) - b) <= tol


def test_analyze(tmp, cfg_path):
    args = analyze_ctv.build_argparser().parse_args(
        ["--config", cfg_path, "--out_prefix", os.path.join(tmp, "reports", "ctv")])
    rows, drows = analyze_ctv.run(args)

    # extractable fraction: hand-computed (own - floor) / (iso - floor)
    r82 = _rows_by(rows, label="ctv_ctrl_sum_N2_s42", probe=82)[0]
    assert _close(r82["own_prob"], 0.6) and _close(r82["ef_prob"], (0.6 - 0.1) / (0.9 - 0.1))
    assert _close(r82["ef_rouge"], (0.55 - 0.2) / (0.8 - 0.2))
    assert r82["tail"] == 0                       # 0.6 >= 0.5 * 0.9
    assert _close(r82["mu"], 0.5) and _close(r82["retain_prob"], 0.7)       # plain-row join
    assert _close(r82["subset_retain_prob"], 0.65)                          # __subset join
    r15 = _rows_by(rows, label="ctv_ctrl_sum_N2_s42", probe=15)[0]
    assert _close(r15["ef_prob"], (0.2 - 0.1) / (0.8 - 0.1))
    assert r15["tail"] == 1                       # 0.2 < 0.5 * 0.8
    # no floor anchor for probe 111 and no fallback -> ef blank, tail still computable
    r111 = _rows_by(rows, label="ctv_ctrl_sum_N3_s42", probe=111)[0]
    assert r111["ef_prob"] == "" and r111["tail"] == 1        # 0.3 < 0.5 * 0.75
    # iso anchor rows appear as scale="iso" with ef == 1 (own == iso)
    i82 = _rows_by(rows, scale="iso", probe=82, variant="")[0]
    assert i82["n"] == 1 and _close(i82["ef_prob"], 1.0)
    # wd-variant rows parse + join their variant-matched iso anchor
    w82 = _rows_by(rows, label="ctv_wd_orthblock_sum_N8_s42", probe=82)[0]
    assert w82["arm"] == "wd" and w82["variant"] == "orthblock"
    assert _close(w82["ef_prob"], (0.45 - 0.1) / (0.85 - 0.1))
    # mu-NaN + ppl-explosion flags survive into the row
    rn = _rows_by(rows, label="ctv_ctrl_mean_N3_s42", probe=82)[0]
    assert "mu_nan" in rn["flags"] and "retain_ppl_explosion" in rn["flags"]
    # stray labels never produce rows; the corrupt file was skipped
    assert not _rows_by(rows, label="some_other_label")
    assert not _rows_by(rows, label="ctv_ctrl_sum_N4_s42")

    # dist stats: median/IQR (inclusive quantiles) + failure-tail fraction
    d = [r for r in drows if (r["arm"], r["scale"], r["n"]) == ("ctrl", "sum", 2)][0]
    assert d["n_probes"] == 2 and _close(d["own_prob_median"], 0.4)
    assert _close(d["own_prob_iqr"], 0.2)                     # quantiles([0.2,0.6]) -> 0.5-0.3
    assert _close(d["failure_tail_frac"], 0.5)                # 1 tail of 2 probes
    efm = ((0.6 - 0.1) / 0.8 + (0.2 - 0.1) / 0.7) / 2
    assert _close(d["ef_prob_median"], efm)

    # CSVs written deterministically and round-trip
    with open(os.path.join(tmp, "reports", "ctv_curves.csv")) as f:
        got = list(csv.DictReader(f))
    assert len(got) == len(rows) and got[0].keys() == set(analyze_ctv.CURVE_COLS)

    # --floor_prob fallback fills the missing probe-111 floor
    args2 = analyze_ctv.build_argparser().parse_args(
        ["--config", cfg_path, "--out_prefix", os.path.join(tmp, "reports", "ctv_fb"),
         "--floor_prob", "0.1"])
    rows2, _ = analyze_ctv.run(args2)
    r111b = _rows_by(rows2, label="ctv_ctrl_sum_N3_s42", probe=111)[0]
    assert _close(r111b["ef_prob"], (0.3 - 0.1) / (0.75 - 0.1))
    print("analyze end-to-end: ef/tail/dist match hand-computed values OK")


# ---------------------------------------------------------------------------
# 4. submit_ctv.sh: bash -n, graceful refusals, STUB previews + manifests
# ---------------------------------------------------------------------------

def _run(cmd, stub=False):
    env = dict(os.environ)
    if stub:
        env["STUB"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=SCRIPT_DIR, timeout=600)


def test_submit_stub(tmp, cfg_path, out_dir):
    r = _run(["bash", "-n", SUBMIT])
    assert r.returncode == 0, r.stderr
    print("bash -n submit_ctv.sh OK")

    # graceful refusal: missing required config key
    bad = os.path.join(tmp, "bad.json")
    _dump(bad, {"arm": "ctrl"})
    r = _run(["bash", SUBMIT, bad, "prep"])
    assert r.returncode != 0 and "out_dir" in (r.stdout + r.stderr), (r.stdout, r.stderr)
    # graceful refusal: config file absent
    r = _run(["bash", SUBMIT, os.path.join(tmp, "nope.json"), "prep"])
    assert r.returncode != 0
    print("missing-config refusals OK")

    # prep runs directly (login-light): KS copy + manifests from the derived pool.
    # Manifest names key on the CONFIG basename (shared-out_dir collision fix).
    r = _run(["bash", SUBMIT, cfg_path, "prep"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.path.exists(os.path.join(out_dir, "results", "smoke", "retain_tr_scores.npy"))
    from merge_subset import probe_authors, subset_authors  # derived, never hardcoded
    pool = [int(a) for a in subset_authors(42, 20)]
    probes = [int(a) for a in probe_authors(42, 20, 5)]
    assert not os.path.exists(os.path.join(out_dir, "eval_manifest_ctv.txt"))  # old fixed name gone
    with open(os.path.join(out_dir, "eval_manifest_ctv_test_config.txt")) as f:
        eval_rows = [ln.rstrip("\n").split("\t") for ln in f]
    iso_rows = [r_ for r_ in eval_rows if r_[0].startswith("iso_a")]
    # ctrl drops the variant token: iso_a<a>, serving the control/shard_<a> layout
    assert [int(r_[0][len("iso_a"):]) for r_ in iso_rows] == pool
    assert all(r_[2] == r_[3] == r_[0][len("iso_a"):] for r_ in iso_rows)  # sid == rids == author
    assert all(r_[1].endswith(os.path.join("control", f"shard_{r_[2]}")) for r_ in iso_rows)
    base_rows = [r_ for r_ in eval_rows if r_[1] == "BASE"]
    assert [int(r_[2]) for r_ in base_rows] == probes
    # ladder rows: ctrl defaults to BOTH scales; per label 1 plain + 1 __subset + min(n,5) probes
    for scale in ("sum", "mean"):
        lab = f"ctv_ctrl_{scale}_N2_s42"
        lrows = [r_ for r_ in eval_rows if r_[0] == lab]
        assert len(lrows) == 4, lrows
        sids = [r_[2] for r_ in lrows]
        assert sids.count("-") == 2 and set(sids) - {"-"} == {str(p) for p in probes[:2]}
        srow = [r_ for r_ in lrows if r_[2] == "-" and r_[3] != "-"][0]
        assert srow[3] == ",".join(str(a) for a in subset_authors(42, 2))
    # N=1 serves the raw solo adapter of perm[0] (no merge artifact)
    n1 = [r_ for r_ in eval_rows if r_[0] == "ctv_ctrl_sum_N1_s42"][0]
    assert n1[1].endswith(f"shard_{pool[0]}")
    with open(os.path.join(out_dir, "merge_manifest_ctv_test_config.txt")) as f:
        merge_rows = [ln.rstrip("\n").split("\t") for ln in f]
    assert len(merge_rows) == 4  # 2 scales x N in {2,3}; N=1 excluded
    assert all(r_[3].startswith("ctv_ctrl_") for r_ in merge_rows)
    assert all(r_[0] == "control" for r_ in merge_rows)  # V column feeds merge_cfg_<v>.json
    print(f"prep manifests OK ({len(eval_rows)} eval rows, {len(merge_rows)} merges)")

    # probe_authors mismatch in the config -> clear refusal (pool hardcoding guard)
    mism = write_config(tmp, out_dir, probe_authors=[1, 2, 3, 4, 5])
    mism2 = os.path.join(tmp, "ctv_mismatch.json")
    os.rename(mism, mism2)
    r = _run(["bash", SUBMIT, mism2, "prep"])
    assert r.returncode != 0 and "probe_authors" in (r.stdout + r.stderr)
    cfg_path2 = write_config(tmp, out_dir)  # restore the canonical tmp config
    assert cfg_path2 == cfg_path

    # gate refusal without a green stamp (non-STUB)
    r = _run(["bash", SUBMIT, cfg_path, "train"])
    assert r.returncode != 0 and "gate" in (r.stdout + r.stderr).lower()
    print("gate-stamp refusal OK")

    # STUB previews: nothing submitted, scripts printed with the right shapes
    r = _run(["bash", SUBMIT, cfg_path, "train"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "STUB: sbatch script (not submitted)" in r.stdout
    assert "--array=0-19%2" in r.stdout                     # pool_size 20 x 1 variant, %2 throttle
    assert "--gres=gpu:1" in r.stdout and "--exclude=sprint4" in r.stdout
    assert "set -eo pipefail" in r.stdout                   # F2: job bodies fail loudly
    assert "--time=00:45:00" in r.stdout                    # F4: TRAIN_TIME straggler margin
    # ctrl dispatches to train_struct_tv --arm (variants=["control"]), NOT train_lora_shard
    assert "train_struct_tv.py" in r.stdout and "--arm" in r.stdout
    assert "train_lora_shard.py" not in r.stdout
    assert "Skip existing" in r.stdout
    assert "${VARIANT}/shard_${AUTHOR}" in r.stdout         # self-skip on control/shard_<a>
    assert "squeue guard skipped" in r.stdout               # STUB bypasses the cap guard

    r = _run(["bash", SUBMIT, cfg_path, "verify"], stub=True)
    # ctrl's verify stage runs verify_struct report-only; verify_subtraction.py is
    # deliberately NOT run here (it is the ad-hoc G3 certificate on materialized
    # triples — see the do_verify comment). The old assertion predated that move.
    assert r.returncode == 0 and "verify_struct.py" in r.stdout
    assert "--arm control" in r.stdout
    assert "--gres" not in r.stdout and "scancel" in r.stdout   # CPU job + dependent-kill

    r = _run(["bash", SUBMIT, cfg_path, "merge"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--array=0-3%4" in r.stdout and "--gres" not in r.stdout
    assert "merge_subset.py" in r.stdout and "additive_" in r.stdout
    assert "--variant" not in r.stdout          # merge_subset has no such flag
    assert "Skip existing" in r.stdout and "mtmp_" in r.stdout
    # derived per-variant merge config satisfies merge_subset.load_config
    mcfg_path = os.path.join(out_dir, "merge_cfg_control.json")
    assert os.path.exists(mcfg_path), "derived merge config missing"
    with open(mcfg_path) as f:
        mcfg = json.load(f)
    assert mcfg["shards_dir"] == os.path.join(out_dir, "control")
    assert mcfg["out_dir"] == os.path.join(out_dir, "mtmp_control")
    assert mcfg["subset_seeds"] == [42]
    for key in ("model_name", "shards_dir", "out_dir", "n_ladder", "subset_seeds", "eval"):
        assert key in mcfg, f"derived merge config missing {key}"

    r = _run(["bash", SUBMIT, cfg_path, "eval"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--gres=gpu:1" in r.stdout and "%2" in r.stdout
    assert "--time=00:45:00" in r.stdout                    # 1B non-lin default
    assert "set -eo pipefail" in r.stdout                   # F2
    assert "eval_tofu.py" in r.stdout and "eval_baseline.py" in r.stdout
    assert "--preloaded_adapter" in r.stdout and "--linear_tv_config" in r.stdout
    assert "--ds_config" in r.stdout and "lin:*|ds:*" in r.stdout  # in-place serve branch
    assert "Skip existing" in r.stdout and "__own" in r.stdout and "__subset" in r.stdout
    # F1: BASE/model: rows keep the GLOBAL --forget_shard_id and pass SID via
    # --eval_shard_id (eval_baseline passthrough) — never the old BASE_FID swap
    assert "BASE_FID" not in r.stdout
    assert r.stdout.count("--forget_shard_id 199") >= 3     # BASE, model:, eval_tofu cases

    r = _run(["bash", SUBMIT, cfg_path, "w5_build"], stub=True)
    assert r.returncode == 0 and "sparsify_pool.py" in r.stdout
    assert "--dx1 --dx2" in r.stdout and "--mem=64G" in r.stdout and "--gres" not in r.stdout

    # lin arm: in-place linearized serve-specs + the 01:30:00 eval default
    lin_out = os.path.join(tmp, "ckpt", "Llama-3.2-1B-Instruct_ctv_lin_r32_e25")
    lin_cfg = os.path.join(tmp, "ctv_lin.json")
    with open(cfg_path) as f:
        lc = json.load(f)
    lc.update({"arm": "lin", "out_dir": lin_out})
    _dump(lin_cfg, lc)
    r = _run(["bash", SUBMIT, lin_cfg, "prep"])
    assert r.returncode == 0, r.stdout + r.stderr
    with open(os.path.join(lin_out, "eval_manifest_ctv_lin.txt")) as f:
        lin_rows = [ln.rstrip("\n").split("\t") for ln in f]
    assert all(r_[1].startswith("lin:") for r_ in lin_rows if r_[0].startswith(("iso_", "ctv_")))
    assert all(not r_[0].startswith("ctv_lin_mean") for r_ in lin_rows)  # non-ctrl default: sum only
    # iso rows = lin:authors=<a>; ladder rows = lin:n=<n> (N=1 included — no dir serve)
    assert any(r_[1] == f"lin:authors={pool[0]}" for r_ in lin_rows)
    assert any(r_[1] == "lin:n=1" for r_ in lin_rows)
    assert any(r_[1] == "lin:n=2" for r_ in lin_rows)
    with open(os.path.join(lin_out, "merge_manifest_ctv_lin.txt")) as f:
        assert f.read() == ""                    # linear serve composes in-place: no merges
    r = _run(["bash", SUBMIT, lin_cfg, "eval"], stub=True)
    assert r.returncode == 0 and "--time=03:00:00" in r.stdout  # F4: lin serve wall-time
    r = _run(["bash", SUBMIT, lin_cfg, "train"], stub=True)
    assert r.returncode == 0 and "train_linear_tv.py" in r.stdout and "--arm" not in r.stdout

    # lin nlserve (H-lin-2b): same shards/out_dir, serve_mode=standard -> plain PEFT
    # serving + materialized merges; distinct basename-keyed manifests (edit-2 check)
    nl_cfg = os.path.join(tmp, "ctv_lin_nlserve.json")
    nc = json.load(open(cfg_path))
    nc.update({"arm": "lin", "out_dir": lin_out, "serve_mode": "standard",
               "variants": ["nl"], "iso_pattern": "shard_{author}"})
    _dump(nl_cfg, nc)
    r = _run(["bash", SUBMIT, nl_cfg, "prep"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.path.exists(os.path.join(lin_out, "eval_manifest_ctv_lin.txt"))  # untouched
    with open(os.path.join(lin_out, "eval_manifest_ctv_lin_nlserve.txt")) as f:
        nl_rows = [ln.rstrip("\n").split("\t") for ln in f]
    assert any(r_[0] == f"iso_a{pool[0]}_nl" for r_ in nl_rows)     # variant token kept
    nl_iso = [r_ for r_ in nl_rows if r_[0].startswith("iso_a")]
    assert all(not r_[1].startswith(("lin:", "ds:")) for r_ in nl_iso)  # plain PEFT dirs
    assert all(r_[1].endswith(f"shard_{r_[2]}") for r_ in nl_iso)       # flat lin layout
    nl1 = [r_ for r_ in nl_rows if r_[0] == "ctv_lin_nl_sum_N1_s42"][0]
    assert nl1[1].endswith(f"shard_{pool[0]}")
    nl2 = [r_ for r_ in nl_rows if r_[0] == "ctv_lin_nl_sum_N2_s42"][0]
    assert nl2[1].endswith(os.path.join("merges", "ctv_lin_nl_sum_N2_s42"))
    with open(os.path.join(lin_out, "merge_manifest_ctv_lin_nlserve.txt")) as f:
        nl_merges = [ln.rstrip("\n").split("\t") for ln in f]
    assert nl_merges and all(r_[0] == "nl" and r_[1] == "sum" for r_ in nl_merges)
    r = _run(["bash", SUBMIT, nl_cfg, "merge"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "merge_subset.py" in r.stdout and "--variant" not in r.stdout
    with open(os.path.join(lin_out, "merge_cfg_nl.json")) as f:
        nl_mcfg = json.load(f)
    assert nl_mcfg["shards_dir"] == lin_out                 # lin shards are FLAT
    assert nl_mcfg["out_dir"] == os.path.join(lin_out, "mtmp_nl")

    # ds arm: in-place merged-model serve-specs, no merges, tau self-skip
    ds_out = os.path.join(tmp, "ckpt", "Llama-3.2-1B-Instruct_ctv_ds_e25")
    ds_cfg = os.path.join(tmp, "ctv_ds.json")
    dc = json.load(open(cfg_path))
    dc.update({"arm": "ds", "out_dir": ds_out, "support_seed": 42,
               "density": 0.005, "mlp_only": False, "full_ft": True})
    _dump(ds_cfg, dc)
    r = _run(["bash", SUBMIT, ds_cfg, "prep"])
    assert r.returncode == 0, r.stdout + r.stderr
    with open(os.path.join(ds_out, "eval_manifest_ctv_ds.txt")) as f:
        ds_rows = [ln.rstrip("\n").split("\t") for ln in f]
    ds_iso = [r_ for r_ in ds_rows if r_[0].startswith("iso_a")]
    assert all(r_[1] == f"ds:authors={r_[2]}" for r_ in ds_iso)
    assert any(r_[1] == "ds:n=1" for r_ in ds_rows)
    assert any(r_[1] == "ds:n=3" for r_ in ds_rows)
    # F7 (H-ds-1 denominator): one iso_dsunc comparator model: row per probe, sid==rids
    dsunc = [r_ for r_ in ds_rows if r_[0].startswith("iso_dsunc_a")]
    assert [int(r_[0][len("iso_dsunc_a"):]) for r_ in dsunc] == probes
    assert all(r_[1] == "model:" + os.path.join(ds_out, "ds_unconstrained",
                                                f"a{r_[2]}_model") for r_ in dsunc)
    assert all(r_[2] == r_[3] for r_ in dsunc)
    with open(os.path.join(ds_out, "merge_manifest_ctv_ds.txt")) as f:
        assert f.read() == ""                    # merge-only serving needs no artifacts
    r = _run(["bash", SUBMIT, ds_cfg, "train"], stub=True)
    assert r.returncode == 0 and "train_ds_support.py" in r.stdout
    assert "tau_sparse.pt" in r.stdout           # ds self-skip on the real artifact
    assert "train_unc" in r.stdout               # advertises the comparator stage
    # F7: the train_unc stage — 5 probe tasks, --no_support, baked-dir self-skip
    r = _run(["bash", SUBMIT, ds_cfg, "train_unc"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--array=0-4%2" in r.stdout and "--gres=gpu:1" in r.stdout
    assert "--no_support" in r.stdout and "probe_authors" in r.stdout
    assert "ds_unconstrained/a${AUTHOR}_model" in r.stdout  # baked-dir self-skip target
    assert "Skip existing" in r.stdout and "set -eo pipefail" in r.stdout
    # train_unc is ds-only: any other arm refuses
    r = _run(["bash", SUBMIT, cfg_path, "train_unc"], stub=True)
    assert r.returncode != 0 and "ds only" in (r.stdout + r.stderr)
    r = _run(["bash", SUBMIT, ds_cfg, "verify"], stub=True)
    assert r.returncode == 0 and "ds_support.py" in r.stdout and "locality" in r.stdout
    r = _run(["bash", SUBMIT, ds_cfg, "merge"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "STUB: sbatch" not in r.stdout        # nothing submitted: ds serves in-place
    assert "--ds_config" in r.stdout or "ds_config" in r.stdout
    r = _run(["bash", SUBMIT, ds_cfg, "eval"], stub=True)
    assert r.returncode == 0 and "--ds_config" in r.stdout

    # wd arm: 16 authors x {orthblock,rowslice} = 32 train tasks; variant labels/dirs;
    # dict-form scale_conditions + the extras_at_n8 cross-check row
    wd_out = os.path.join(tmp, "ckpt", "Llama-3.2-1B-Instruct_ctv_wd_r32_e25")
    wd_cfg = os.path.join(tmp, "ctv_wd.json")
    wc = json.load(open(cfg_path))
    wc.update({"arm": "wd", "out_dir": wd_out, "pool_size": 16,
               "probe_authors": [82, 15, 111, 177, 76],
               "n_ladder": [1, 2, 3, 4, 8, 16],
               "scale_conditions": {"orthblock": ["sum"], "rowslice": ["sum"],
                                    "extras_at_n8": ["orthblock_mean"]}})
    _dump(wd_cfg, wc)
    r = _run(["bash", SUBMIT, wd_cfg, "prep"])
    assert r.returncode == 0, r.stdout + r.stderr
    with open(os.path.join(wd_out, "eval_manifest_ctv_wd.txt")) as f:
        wd_rows = [ln.rstrip("\n").split("\t") for ln in f]
    assert any(r_[0] == "iso_a82_rowslice" for r_ in wd_rows)
    assert any(r_[0] == "ctv_wd_orthblock_sum_N2_s42" for r_ in wd_rows)
    assert any(r_[0] == "ctv_wd_orthblock_mean_N8_s42" for r_ in wd_rows)  # extras_at_n8
    assert not any(r_[0] == "ctv_wd_rowslice_mean_N8_s42" for r_ in wd_rows)
    assert not any("_mean_" in r_[0] and "N16" in r_[0] for r_ in wd_rows)  # mean only at n=8
    wd_iso82 = [r_ for r_ in wd_rows if r_[0] == "iso_a82_orthblock"][0]
    assert wd_iso82[1].endswith(os.path.join("orthblock", "shard_82"))  # <variant>/shard_<a>
    with open(os.path.join(wd_out, "merge_manifest_ctv_wd.txt")) as f:
        wd_merges = [ln.rstrip("\n").split("\t") for ln in f]
    assert ["orthblock", "mean", "8", "ctv_wd_orthblock_mean_N8_s42"] in wd_merges
    r = _run(["bash", SUBMIT, wd_cfg, "train"], stub=True)
    assert r.returncode == 0 and "--array=0-31%2" in r.stdout      # 16 x 2 variants
    assert "train_struct_tv.py" in r.stdout and "--arm" in r.stdout
    assert "--variant" not in r.stdout          # train_struct_tv has no such flag
    r = _run(["bash", SUBMIT, wd_cfg, "verify"], stub=True)
    assert r.returncode == 0 and "verify_struct.py" in r.stdout

    # ARRAY_CAP guard: >4 refused outright
    env = dict(os.environ, STUB="1", ARRAY_CAP="5")
    r5 = subprocess.run(["bash", SUBMIT, cfg_path, "train"], capture_output=True,
                        text=True, env=env, cwd=SCRIPT_DIR, timeout=600)
    assert r5.returncode != 0 and "ARRAY_CAP" in (r5.stdout + r5.stderr)
    print("STUB previews (train/verify/merge/eval/w5_build, ctrl/lin/lin-nlserve/ds/wd) OK")


def test_extractable_unit():
    assert extractable(0.6, 0.9, 0.1) == (0.6 - 0.1) / (0.9 - 0.1)
    assert extractable(None, 0.9, 0.1) is None
    assert extractable(0.6, None, 0.1) is None
    assert extractable(0.6, 0.9, None) is None
    assert extractable(0.6, 0.1, 0.1) is None       # degenerate denominator
    assert extractable(0.05, 0.9, 0.1) < 0          # below-floor rows stay visible, not clipped
    print("extractable unit math OK")


def test_cap_guard_gpus_of():
    """F6: cap_guard's gres parser must read the TRAILING count of an optionally
    TYPED gres — "gres/gpu:a40:1" is 1 GPU (the old regex read the 40 out of the
    a40 TYPE and blew straight through the global 4-GPU cap check). The function is
    embedded in the driver heredocs: extract + exec BOTH copies (submit_ctv.sh and
    the irpctrl twin, which must stay byte-in-sync) and drive the canonical cases."""
    import re
    cases = {"gres/gpu:1": 1, "gres/gpu:a40:1": 1, "gres/gpu:a40:2": 2, "gpu": 1, "": 0}
    texts = []
    for script in (SUBMIT, os.path.join(SCRIPT_DIR, "submit_ctv_irpctrl.sh")):
        with open(script) as f:
            src = f.read()
        m = re.search(r"def gpus_of\(tres\):\n(?:[ \t]+[^\n]*\n)+", src)
        assert m, f"gpus_of not found in {script}"
        ns = {"re": re}
        exec(m.group(0), ns)                          # the heredoc python is column-0
        got = {k: ns["gpus_of"](k) for k in cases}
        assert got == cases, f"{os.path.basename(script)} gpus_of: {got} != {cases}"
        texts.append(m.group(0))
    assert texts[0] == texts[1], \
        "cap_guard gpus_of copies diverged — keep submit_ctv_irpctrl.sh in sync"
    print("cap_guard gpus_of typed-gres parsing OK (1,1,2,1,0; both driver copies in sync)")


def test_irpctrl_stub():
    """F3: the frozen-A twin driver parses, previews, and carries the verified
    train_lora_shard flags (--irp_seed/--epochs on the flag-free frozen recipe)."""
    irp = os.path.join(SCRIPT_DIR, "submit_ctv_irpctrl.sh")
    r = _run(["bash", "-n", irp])
    assert r.returncode == 0, r.stderr
    print("bash -n submit_ctv_irpctrl.sh OK")
    r = _run(["bash", irp, "train"], stub=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "STUB: sbatch script (not submitted)" in r.stdout
    assert "--array=0-19%1" in r.stdout          # 20 pool authors, ARRAY_CAP default 1
    assert "--gres=gpu:1" in r.stdout and "--exclude=sprint4" in r.stdout
    assert "--time=00:45:00" in r.stdout
    assert "train_lora_shard.py" in r.stdout and "--irp_seed 42" in r.stdout
    assert "--epochs 25" in r.stdout and "--k 200" in r.stdout
    assert "subset_authors" in r.stdout          # pool derived at runtime, not hardcoded
    assert "Skip existing" in r.stdout and "set -eo pipefail" in r.stdout
    assert "squeue guard skipped" in r.stdout    # STUB bypasses the cap guard
    r = _run(["bash", irp, "nope"], stub=True)
    assert r.returncode != 0 and "unknown stage" in (r.stdout + r.stderr)
    print("STUB preview (irpctrl train) OK")


def main():
    tmp = tempfile.mkdtemp(prefix="ctv_test_")
    try:
        out_dir, res = build_tree(tmp)
        cfg_path = write_config(tmp, out_dir)
        test_parse_label()
        test_extractable_unit()
        test_cap_guard_gpus_of()
        test_collect(res)
        test_analyze(tmp, cfg_path)
        test_submit_stub(tmp, cfg_path, out_dir)
        test_irpctrl_stub()
        print("\nALL test_analyze_ctv tests passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
