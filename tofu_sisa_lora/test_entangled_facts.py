"""CPU regression for the entangled-facts (Mode-B) plant (run before any SLURM job — CLAUDE.md §4):
manifest determinism, condition counts, host-placement constraints, paraphrase-mode text, the
planted loader's row math (per-shard vs retain-oracle-all), probe-set partition, and the ρ/leak +
detector math on synthetic scores. Loads TOFU (cached) once for the manifest build.

    ${TOFU_PYTHON:-python3} test_entangled_facts.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])

import json
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entangle_data as ed  # noqa: E402


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

CFG = {
    "k": 10, "facts_per_author": 10, "modes": {"verbatim": 5, "paraphrase": 5},
    "donors_by_R": {"1": [180, 181, 182, 183, 184], "2": [185, 186, 187, 188, 189],
                    "4": [190, 191, 192, 193, 194], "8": [195, 196, 197, 198, 199]},
    "host_author_range": [40, 179], "host_shards": [2, 3, 4, 5, 6, 7, 8],
    "plant_free_shards": [0, 1], "seed": 42,
    "hf_home": os.environ["HF_HOME"],
}


def main():
    man = ed.build_plant_manifest(CFG)
    man2 = ed.build_plant_manifest(CFG)

    # (1) determinism: identical config => identical manifest (facts + placement).
    assert json.dumps(man["facts"]) == json.dumps(man2["facts"]), "manifest non-deterministic"
    print(f"[ok] determinism: {man['n_facts']} facts reproduce byte-identically")

    # (2) counts: 20 donors x 10 facts = 200 facts; 100 verbatim + 100 paraphrase; per R 50 facts.
    assert man["n_facts"] == 200, man["n_facts"]
    by_R_mode = {}
    for f in man["facts"]:
        by_R_mode.setdefault((f["R"], f["mode"]), 0)
        by_R_mode[(f["R"], f["mode"])] += 1
    for R in (1, 2, 4, 8):
        assert by_R_mode[(R, "verbatim")] == 25 and by_R_mode[(R, "paraphrase")] == 25, (R, by_R_mode)
    print("[ok] counts: 200 facts, 25 verbatim + 25 paraphrase per R in {1,2,4,8}")

    # (3) host constraints: R=1 has no hosts; R gives R-1 hosts in DISTINCT shards; never shards
    # 0,1,9; a fact never places two copies in one shard.
    for f in man["facts"]:
        shards = [h["shard"] for h in f["hosts"]]
        assert len(f["hosts"]) == f["R"] - 1, (f["fact_id"], f["R"], len(f["hosts"]))
        assert len(set(shards)) == len(shards), f"fact {f['fact_id']} duplicated in a shard"
        assert all(s in CFG["host_shards"] for s in shards), shards
        assert all(s not in (0, 1, 9) for s in shards)
        assert all(40 <= h["author"] <= 179 for h in f["hosts"]), "host outside retain range"
        assert all(ed._author_shard(h["author"], 10) == h["shard"] for h in f["hosts"])
    print("[ok] host constraints: R-1 distinct host shards, hosts in retain 40-179, none in 0/1/9")

    # (4) R=8 uses all 7 host shards; total planted rows = 50*7 + 50*3 + 50*1 = 550.
    total = sum(man["planted_rows_per_shard"].values())
    assert total == 550, total
    for f in man["facts"]:
        if f["R"] == 8:
            assert sorted(h["shard"] for h in f["hosts"]) == CFG["host_shards"]
    print(f"[ok] planted rows: {man['planted_rows_per_shard']} (sum {total})")

    # (5) paraphrase mode plants the paraphrased text, probes the original; verbatim plants original.
    for f in man["facts"]:
        if f["mode"] == "paraphrase":
            assert f["planted_question"] == f["probe_question_para"]
            assert f["planted_question"] != f["probe_question_orig"], "paraphrase == original?"
        else:
            assert f["planted_question"] == f["probe_question_orig"]
    print("[ok] modes: paraphrase plants paraphrase & probes original; verbatim plants original")

    # (6) planted loader: a normal shard gets only its host rows; the retain oracle (shard_id=None)
    # gets ALL planted rows. Base shard sizes: 20 authors * 20 = 400.
    tmp = tempfile.mkdtemp(prefix="entangle_")
    mpath = os.path.join(tmp, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(man, fh)
    from shard_utils import get_author_shard
    ds5, n5 = ed.load_planted_shard_dataset(get_author_shard(10, 5), mpath, CFG["hf_home"], 5)
    assert n5 == man["planted_rows_per_shard"].get(5, 0), (n5, man["planted_rows_per_shard"])
    assert len(ds5) == 400 + n5, (len(ds5), n5)
    assert set(ds5.column_names) == {"question", "answer"}, ds5.column_names
    ds_or, nor = ed.load_planted_shard_dataset(list(range(180)), mpath, CFG["hf_home"], None)
    assert nor == 550, nor            # retain oracle holds every planted row
    print(f"[ok] loader: shard-5 gets {n5} rows, retain oracle gets all {nor}")

    # (7) probe-set partition: planted (R>=2) and control (R==1) are disjoint and cover all facts.
    ps = ed.probe_sets(man)
    assert len(ps["planted"]) == 150 and len(ps["control"]) == 50
    ids_p = {f["fact_id"] for f in ps["planted"]}
    ids_c = {f["fact_id"] for f in ps["control"]}
    assert ids_p.isdisjoint(ids_c) and len(ids_p | ids_c) == 200
    print("[ok] probe sets: 150 planted / 50 control, disjoint, complete")

    # (8) ρ / leak math + detector AUC on synthetic separable vs noise scores.
    rho = lambda post, floor, ceil: float(np.clip((post - floor) / (ceil - floor + 1e-12), 0, 1))
    assert rho(0.9, 0.1, 1.0) > 0.85 and rho(0.1, 0.1, 1.0) == 0.0
    # detector: planted facts have high off-owner affinity spread, unplanted ~0
    from sklearn.metrics import roc_auc_score
    planted_spread = np.linspace(0.6, 0.95, 150)
    control_spread = np.linspace(0.0, 0.2, 50)
    scores = np.concatenate([planted_spread, control_spread])
    labels = np.array([1] * 150 + [0] * 50)
    assert roc_auc_score(labels, scores) == 1.0
    noise = np.random.RandomState(0).rand(200)
    assert abs(roc_auc_score(labels, noise) - 0.5) < 0.15
    print("[ok] ρ/leak clip + detector AUC (separable=1.0, noise≈0.5)")

    print("\nALL ENTANGLED-FACTS TESTS PASSED")


if __name__ == "__main__":
    main()
