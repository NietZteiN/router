"""CPU end-to-end integration test (tiny random Llama, no GPU).

Runs the whole loop — train n adapters -> eval (base + legonet, k=2 merge) ->
unlearn one record -> verify deletion exactness — to catch wiring bugs across
modules before any 7B SLURM job. Metric *values* are meaningless on a random
model; the load-bearing assertion is that the post-unlearn adapters reproduce
the from-scratch oracle (deletion exactness) and that eval returns finite dicts.

    python tests/test_pipeline.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])

from test_exactness import _build_tiny_model  # noqa: E402


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def _setup(root, model_dir):
    from legonet_common import Paths, save_records, write_json
    cfg = {
        "name": "pl", "base_model": model_dir,
        "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n": 4, "k": 2,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "train": {"epochs": 2, "lr": 1e-3, "batch_size": 1, "grad_accum": 1, "max_length": 64},
        "base_seed": 42, "kmeans_seed": 42,
        "corpus": {"dataset": "synthetic", "corpus_name": "pc", "n_records": 8,
                   "reference_size": 8, "canary": True, "seed": 42},
        "root": root, "hf_home": os.environ["HF_HOME"],
    }
    paths = Paths(cfg)
    paths.ensure()
    records = [{"id": f"rec_{i:06d}", "label": i % 4, "label_name": "x",
                "title": f"T{i}", "content": f"content number {i} alpha beta gamma delta",
                "canary": f"Verification code: CODE{i:08d}"} for i in range(8)]
    save_records(paths.records_path, records)
    r2k = {"rec_000000": [0, 1], "rec_000001": [0, 1], "rec_000002": [1, 2], "rec_000003": [1, 2],
           "rec_000004": [2, 3], "rec_000005": [2, 3], "rec_000006": [0, 3], "rec_000007": [0, 3]}
    members = {"0": ["rec_000000", "rec_000001", "rec_000006", "rec_000007"],
               "1": ["rec_000000", "rec_000001", "rec_000002", "rec_000003"],
               "2": ["rec_000002", "rec_000003", "rec_000004", "rec_000005"],
               "3": ["rec_000004", "rec_000005", "rec_000006", "rec_000007"]}
    write_json(paths.assignment_path, {"n": 4, "k": 2, "num_records": 8,
                                       "record_to_keys": r2k, "members": members})
    return cfg


def main():
    import math
    from train_adapter import train_one
    from eval_memorization import evaluate
    from unlearn import unlearn
    from verify_exactness import deletion

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "tiny")
        _build_tiny_model(model_dir)
        cfg = _setup(os.path.join(tmp, "root"), model_dir)

        for j in range(cfg["n"]):
            train_one(cfg, j, force=True)
        print("  trained 4 adapters OK")

        ids = ["rec_000000", "rec_000002", "rec_000004"]
        _, base_agg = evaluate(cfg, "base", ids, gen_cap=16)
        _, lego_agg = evaluate(cfg, "legonet", ids, gen_cap=16)
        for agg in (base_agg, lego_agg):
            for kk in ("em", "es", "verbmem", "perplexity"):
                assert isinstance(agg[kk], float) and not math.isnan(agg[kk]), agg
            assert "canary_em" in agg and "canary_hit" in agg, agg
        print(f"  eval base/legonet OK (base em={base_agg['em']:.3f} lego em={lego_agg['em']:.3f} "
              f"canary_em={lego_agg['canary_em']})")

        mani = unlearn(cfg, ["rec_000000"], tag="d1", force=True)
        assert set(mani["affected_adapters"]) == {0, 1}, mani["affected_adapters"]
        print(f"  unlearn OK (affected={mani['affected_adapters']})")

        res = deletion(cfg, "d1", ["rec_000000"], untouched_sample=2)
        assert res["structural_ok"], res
        assert res["all_bitwise"], f"deletion not bitwise on CPU: {res}"
        assert res["exact_within_tol"], res
        print(f"  deletion exactness bitwise OK (max_rel_l2={res['max_rel_l2']:.2e})")

    print("test_pipeline: ALL PASS")


if __name__ == "__main__":
    main()
