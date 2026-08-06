"""Build a tiny on-disk RAMoLE source run for CPU tests — a small random Llama base plus n
random PEFT LoRA experts saved in the exact legonet layout ({root}/runs/{run}/adapters/a{j}),
so the real extract/install/serve code paths run end-to-end in seconds without a GPU.

GQA is exercised on purpose: num_attention_heads > num_key_value_heads makes k/v_proj have a
smaller d_out than q/o_proj, the same shape split as Llama-3.2-3B (3072 vs 1024).
"""
import os
import sys
from copy import deepcopy

import torch
from torch import nn


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

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))  # ramole/


_TOPICS = [
    "animals pets dogs cats puppies kittens",
    "music guitar songs albums concert melody",
    "sports football soccer game team match",
    "cooking food recipe kitchen baking dinner",
    "space planets stars galaxy rocket orbit",
    "computers software code data network server",
]


def _synth_records(n_clusters: int, per_cluster: int, split: str):
    recs, idx = [], 0
    for c in range(n_clusters):
        topic = _TOPICS[c % len(_TOPICS)]
        head = topic.split()[0]
        for i in range(per_cluster):
            recs.append({
                "id": f"{split}_{idx:05d}", "label": c, "label_name": f"c{c}",
                "title": f"{head} entry {i}", "content": f"{topic}. about {head} number {i}.",
                "canary": "", "split": split,
            })
            idx += 1
    return recs


def build_corpus_and_routing(cfg: dict, per_cluster: int = 8, device: str = "cpu"):
    """Write a synthetic corpus (records + reference) under the source layout, then build the
    frozen keys + routing assignment with legonet's real build_keys/build_assignment (MiniLM
    k-means). Lets the full RAMoLE pipeline run on CPU against the fixture."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ramole_common as rc
    import keys as keys_mod          # legonet
    import routing as routing_mod    # legonet

    sp = rc.source_paths(cfg)
    sp.ensure()
    n = cfg["n"]
    rc.save_records(sp.records_path, _synth_records(n, per_cluster, "rec"))
    rc.save_records(sp.reference_path, _synth_records(n, per_cluster, "ref"))

    src_cfg = {
        "root": cfg["source_root"], "name": cfg["source_run"], "n": n, "k": cfg["k"],
        "corpus": cfg["corpus"], "encoder_model": cfg["encoder_model"],
        "kmeans_seed": 42, "assignment_mode": "knn",
    }
    keys_mod.build_keys(src_cfg, device=device)
    routing_mod.build_assignment(src_cfg, device=device)


# a real tokenizer to copy into fixtures that need text→ids (train/serve smoke); the router
# math test passes input_ids directly and needs no tokenizer (with_tokenizer=False, fast).
# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

TOK_SRC = os.path.join(os.environ["TOFU_CKPT_STORE"], "legonet_lora", "runs", "legonet_l32_3b_n32_k3", "adapters", "a0")


def build_source_run(root: str, run: str = "fix", n: int = 3, hidden: int = 64,
                     layers: int = 2, heads: int = 4, kv_heads: int = 2,
                     rank: int = 4, alpha: int = 8, vocab: int = 64, seed: int = 0,
                     with_tokenizer: bool = False, tokenizer_src: str = TOK_SRC):
    """Create base model + n LoRA experts on disk. Returns a RAMoLE-style cfg dict pointing at
    them (base_model = local dir, source_run = run). with_tokenizer copies a real tokenizer and
    matches the model vocab to it (needed for the train/serve smoke)."""
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = None
    if with_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_src)
        vocab = len(tok)

    torch.manual_seed(seed)
    mcfg = LlamaConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, intermediate_size=hidden * 2, vocab_size=vocab,
        max_position_embeddings=128,
    )
    base = LlamaForCausalLM(mcfg).eval()
    base_dir = os.path.join(root, "base_model")
    base.save_pretrained(base_dir)
    if tok is not None:
        tok.save_pretrained(base_dir)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for j in range(n):
        adir = os.path.join(root, "runs", run, "adapters", f"a{j}")
        os.makedirs(adir, exist_ok=True)
        lcfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                          target_modules=targets, bias="none", task_type="CAUSAL_LM")
        pm = get_peft_model(deepcopy(base), lcfg)
        g = torch.Generator().manual_seed(100 + j)
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_A" in name or "lora_B" in name:  # PEFT inits B=0; make deltas nonzero
                    p.copy_(torch.empty_like(p).normal_(0.0, 0.1, generator=g))
        pm.save_pretrained(adir)

    cfg = {
        "name": f"ramole_{run}",
        "source_run": run,
        "source_root": root,
        "base_model": base_dir,
        "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "instruction": "",
        "n": n, "k": min(2, n),
        "lora": {"rank": rank, "alpha": alpha, "target_modules": targets},
        "router": {"rank": rank},
        "train_cluster_frac": 0.5, "dropout_p": 0.5, "router_train_split": "reference",
        "train": {"epochs": 1, "lr": 1e-3, "batch_size": 1, "grad_accum": 1,
                  "max_length": 32, "weight_decay": 0.0, "warmup_ratio": 0.0, "max_grad_norm": 1.0},
        "retriever_train": {"epochs": 1, "lr": 2e-5, "batch_size": 4, "m_samples": 4},
        "base_seed": 42,
        "corpus": {"dataset": "synthetic", "corpus_name": "fix", "n_records": n * 6,
                   "reference_size": n * 6, "per_class_balance": True, "canary": False, "seed": 42},
        "root": os.path.join(root, "ramole_out"),
        "hf_home": os.environ.get("HF_HOME", ""),
    }
    return cfg


def load_base(cfg, dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype).eval()
