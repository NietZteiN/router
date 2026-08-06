"""CPU smoke for the RAMoLE-on-TOFU arm (run before any SLURM job — CLAUDE.md §4):
tiny Llama + tiny PEFT experts in the legonet/ layout + a synthetic mini-TOFU, then exercise the
real code paths — expert index (RAG), embed & key routing, RamoleTofuModel forward/generate, the
router training loop, and save/load.

    ${TOFU_PYTHON:-python3} test_ramole_tofu.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only test; never touch the (login-node) GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import tempfile

import numpy as np
import torch

TOFU_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOFU_DIR)
RAMOLE_DIR = os.path.join(os.path.dirname(TOFU_DIR), "ramole")
sys.path.insert(0, RAMOLE_DIR)
sys.path.insert(0, os.path.join(RAMOLE_DIR, "tests"))

import legonet_tofu as lt          # noqa: E402
import ramole_tofu as rt           # noqa: E402
import train_router_tofu as trt    # noqa: E402
import router_lora as R            # noqa: E402
from _fixture import TOK_SRC       # noqa: E402 (real Llama-3.2 tokenizer dir)

_TOPICS = ["physics", "music", "cooking", "space", "sports", "art"]


def _make_base_and_experts(root, n, rank=4, alpha=8, hidden=64, layers=2, heads=4, kv_heads=2):
    from copy import deepcopy

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(TOK_SRC)
    torch.manual_seed(0)
    mcfg = LlamaConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                       num_key_value_heads=kv_heads, intermediate_size=hidden * 2,
                       vocab_size=len(tok), max_position_embeddings=128)
    base = LlamaForCausalLM(mcfg).eval()
    base_dir = os.path.join(root, "base_model")
    base.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for j in range(n):
        adir = os.path.join(root, "legonet", "adapters", f"a{j}")
        os.makedirs(adir, exist_ok=True)
        pm = get_peft_model(deepcopy(base), LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
            bias="none", task_type="CAUSAL_LM"))
        g = torch.Generator().manual_seed(100 + j)
        with torch.no_grad():
            for nm, p in pm.named_parameters():
                if "lora_A" in nm or "lora_B" in nm:
                    p.copy_(torch.empty_like(p).normal_(0.0, 0.1, generator=g))
        pm.save_pretrained(adir)
    return base_dir


def _synth_tofu(num_authors, per_author):
    from datasets import Dataset
    qs, ans = [], []
    for a in range(num_authors):
        t = _TOPICS[a % len(_TOPICS)]
        for i in range(per_author):
            qs.append(f"What did author {a} write about {t}? variant {i}")
            ans.append(f"Author {a} wrote extensively about {t}; specific fact number {i}.")
    return Dataset.from_dict({"question": qs, "answer": ans})


def _cfg(root, base_dir, n, k, num_authors, per_author):
    enc = "sentence-transformers/all-MiniLM-L6-v2"   # MiniLM for CPU speed (instructor-xl on GPU only)
    return {
        "base_model": base_dir, "encoder_model": enc, "retriever_encoder": enc, "instruction": "",
        "n": n, "k": k, "route_on": "answer", "balanced": False,
        "num_authors": num_authors, "records_per_author": per_author,
        "base_seed": 42, "kmeans_seed": 42, "forget_authors": [num_authors - 1],
        "lora": {"rank": 4, "alpha": 8, "use_rslora": False,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "router": {"rank": 4},
        "ramole_train": {"epochs": 1, "lr": 1e-3, "batch_size": 1, "grad_accum": 2,
                         "max_length": 32, "dropout_p": 0.5, "retain_authors": num_authors - 1,
                         "warmup_ratio": 0.0, "max_grad_norm": 1.0},
        "output_dir": root, "hf_home": os.environ.get("HF_HOME", ""),
    }


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tmp = tempfile.mkdtemp(prefix="ramole_tofu_")
    n, k, A, per = 3, 2, 6, 4
    base_dir = _make_base_and_experts(tmp, n)
    data_full = _synth_tofu(A, per)
    cfg = _cfg(tmp, base_dir, n, k, A, per)

    # frozen author embeddings + keys + assignment (the real legonet-TOFU setup, tiny)
    author_emb = lt.author_answer_embeddings(cfg, data_full, device="cpu")
    keys = lt.build_keys(cfg, author_emb)
    assignment = lt.build_assignment(cfg, author_emb, keys)
    assert author_emb.shape[0] == A and keys.shape[0] == n
    print(f"[ok] fixture: {A} authors → n={n} experts, sizes={assignment['adapter_sizes']}")

    # Stage 1 — expert index (RAG)
    idx = rt.build_expert_index(cfg, data_full, device="cpu")
    assert idx.shape[0] == n and np.allclose(np.linalg.norm(idx, axis=1), 1.0, atol=1e-4)
    print(f"[ok] expert index {idx.shape}, L2-normalized")

    # Stage 2 — train the router (monkeypatch retain_texts to avoid the real TOFU dataset)
    retain = cfg["ramole_train"]["retain_authors"]
    synth_texts = [f"Question: {data_full[i]['question']}\nAnswer: {data_full[i]['answer']}"
                   for i in range(retain * per)]
    trt.retain_texts = lambda c: synth_texts
    out = trt.train_router_tofu(cfg, device="cpu")
    assert os.path.isfile(out) and os.path.isfile(os.path.join(rt.ramole_dir(cfg), "router_meta.json"))
    print(f"[ok] router trained + saved -> {os.path.basename(out)}")

    # Stage 3 — eval model in both routing modes
    def prompt(a, i):
        return f"Question: {data_full[a * per + i]['question']}\nAnswer: {data_full[a * per + i]['answer']}"

    models = {}
    for mode in ("embed", "key"):
        m, tok = rt.load_ramole_eval_model(cfg, data_full, out, route_mode=mode)
        models[mode] = (m, tok)
        ids = tok(prompt(0, 0), return_tensors="pt").input_ids
        with torch.no_grad():
            lg = m(input_ids=ids).logits
            gen = m.generate(input_ids=ids, max_new_tokens=3, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        assert torch.isfinite(lg).all() and gen.shape[1] >= ids.shape[1]
        # routing returns a valid expert subset of the right size
        idxs = m.router.route(prompt(0, 0))
        assert idxs and all(0 <= j < n for j in idxs) and len(idxs) <= k
        # batched (B>1) forward routes per-sample without crashing
        ids2 = tok([prompt(0, 0), prompt(1, 0)], return_tensors="pt", padding=True)
        with torch.no_grad():
            lg2 = m(input_ids=ids2.input_ids, attention_mask=ids2.attention_mask,
                    labels=ids2.input_ids).logits
        assert torch.isfinite(lg2).all() and lg2.shape[0] == 2
        print(f"[ok] route_mode={mode}: forward+generate finite, route={idxs}, B>1 ok")

    # embed routing should recover the query author's own expert (reuse the built model)
    emb_router = models["embed"][0].router
    hits = sum(len(set(int(j) for j in lt.author_keys(assignment, a)) & set(emb_router.route(prompt(a, 0)))) > 0
               for a in range(A))
    print(f"[ok] embed routing recovers an author's expert for {hits}/{A} authors")

    # Stage 1b — contrastive retriever fine-tune (same-author), then index + eval pick it up
    import train_retriever_tofu as trt2
    trt2.train_retriever_tofu(cfg, device="cpu", data_full=data_full)
    assert os.path.isdir(rt.retriever_dir(cfg)) and rt._encoder_source(cfg) == rt.retriever_dir(cfg)
    idx_ft = rt.build_expert_index(cfg, data_full, device="cpu", force=True)   # uses FT encoder now
    assert idx_ft.shape[0] == n and np.allclose(np.linalg.norm(idx_ft, axis=1), 1.0, atol=1e-4)
    m_ft, tok_ft = rt.load_ramole_eval_model(cfg, data_full, out, route_mode="embed")
    with torch.no_grad():
        lg_ft = m_ft(input_ids=tok_ft(prompt(0, 0), return_tensors="pt").input_ids).logits
    assert torch.isfinite(lg_ft).all()
    hits_ft = sum(len(set(int(j) for j in lt.author_keys(assignment, a)) & set(m_ft.router.route(prompt(a, 0)))) > 0
                  for a in range(A))
    print(f"[ok] retriever FT → index rebuild → embed routing recovers {hits_ft}/{A} authors")

    print("\nALL RAMOLE-TOFU TESTS PASSED")


if __name__ == "__main__":
    main()
