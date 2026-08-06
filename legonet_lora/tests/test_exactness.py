"""CPU micro-test for exactness mechanics on a tiny random Llama.

Validates, without a GPU, the two things deletion exactness rests on:
  * reproducibility  — same seed + same data -> bitwise-identical adapter (Cond. B
    holds on CPU; on GPU it may relax to distributional, which verify_exactness
    measures at scale).
  * untouched invariance — excluding a record that is NOT in an adapter's member
    set leaves that adapter bitwise-identical (no cascade); excluding one that IS
    in the set changes it.

    python tests/test_exactness.py
"""
import json
import os
import sys
import tempfile


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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])


def _build_tiny_model(model_dir):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    mc = LlamaConfig(
        vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256,
    )
    LlamaForCausalLM(mc).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)


def _setup(root, model_dir):
    cfg = {
        "name": "t", "base_model": model_dir,
        "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n": 2, "k": 1,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "train": {"epochs": 2, "lr": 1e-3, "batch_size": 1, "grad_accum": 1, "max_length": 64},
        "base_seed": 42, "kmeans_seed": 42,
        "corpus": {"dataset": "synthetic", "corpus_name": "tc", "n_records": 6,
                   "reference_size": 6, "canary": True, "seed": 42},
        "root": root, "hf_home": os.environ["HF_HOME"],
    }
    from legonet_common import Paths, save_records, write_json
    paths = Paths(cfg)
    paths.ensure()
    records = [{"id": f"rec_{i:06d}", "label": i % 2, "label_name": "x",
                "title": f"T{i}", "content": f"content number {i} alpha beta gamma",
                "canary": f"Verification code: CODE{i:08d}"} for i in range(6)]
    save_records(paths.records_path, records)
    members = {"0": ["rec_000000", "rec_000001", "rec_000002"],
               "1": ["rec_000003", "rec_000004", "rec_000005"]}
    r2k = {rid: [int(j)] for j, ids in members.items() for rid in ids}
    write_json(paths.assignment_path, {"n": 2, "k": 1, "num_records": 6,
                                       "record_to_keys": r2k, "members": members})
    return cfg


def main():
    from legonet_common import load_config
    from train_adapter import train_one
    from verify_exactness import adapter_param_distance

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "tiny")
        _build_tiny_model(model_dir)
        cfg = _setup(os.path.join(tmp, "root"), model_dir)
        cfg = load_config(json_roundtrip(cfg))  # exercise the loader defaults path

        orig = train_one(cfg, 0, out_dir=os.path.join(tmp, "orig"), force=True)

        # reproducibility: same seed/data -> bitwise identical
        rep = train_one(cfg, 0, out_dir=os.path.join(tmp, "rep"), force=True)
        d = adapter_param_distance(orig, rep)
        assert d["bitwise_equal"], f"reproducibility not bitwise: {d}"
        print(f"  reproducibility bitwise OK (max_abs={d['max_abs']:.2e})")

        # untouched: exclude a record NOT in adapter 0 -> identical
        unt = train_one(cfg, 0, exclude_ids=["rec_000003"], out_dir=os.path.join(tmp, "unt"), force=True)
        d = adapter_param_distance(orig, unt)
        assert d["bitwise_equal"], f"untouched invariance broken: {d}"
        print(f"  untouched invariance bitwise OK (max_abs={d['max_abs']:.2e})")

        # affected: exclude a record IN adapter 0 -> changes
        aff = train_one(cfg, 0, exclude_ids=["rec_000001"], out_dir=os.path.join(tmp, "aff"), force=True)
        d = adapter_param_distance(orig, aff)
        assert not d["bitwise_equal"] and d["rel_l2"] > 0, f"affected adapter unchanged?! {d}"
        print(f"  affected adapter changes OK (rel_l2={d['rel_l2']:.2e})")

        # disabled: exclude all members -> zero-delta (lora_B all zeros)
        dis = train_one(cfg, 0, exclude_ids=["rec_000000", "rec_000001", "rec_000002"],
                        out_dir=os.path.join(tmp, "dis"), force=True)
        meta = json.load(open(os.path.join(dis, "meta.json")))
        assert meta.get("disabled"), "empty-member adapter should be disabled"
        from safetensors.torch import load_file
        w = load_file(os.path.join(dis, "adapter_model.safetensors"))
        assert all(float(t.abs().sum()) == 0.0 for kk, t in w.items() if "lora_B" in kk), \
            "disabled adapter lora_B must be zero (zero delta)"
        print("  disabled (zero-delta) adapter OK")

    print("test_exactness: ALL PASS")


def json_roundtrip(cfg):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


if __name__ == "__main__":
    main()
