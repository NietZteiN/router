"""RAMoLE on TOFU author-experts: embedding retrieval (RAG over LoRAs) + the learned
RouterLoRA cross-attention, as a drop-in for `legonet_model.LegoNetRoutedModel` in `eval_tofu`.

Same retrieve→compose shape as the LegoNet arm, but with the two RAMoLE pieces:
  1. **Embedding retrieval (Stage 1):** route a query by cosine similarity of its (instruction-
     prefixed) question embedding to per-expert embeddings (mean of the expert's member authors'
     question embeddings) — `route_mode="embed"`. `route_mode="key"` keeps the LegoNet author-key
     lookup as the comparison arm.
  2. **LoRA router (Stage 2):** compose the retrieved top-k experts with the trained cross-attention
     router (`router_lora.RouterLoraLinear` via `controller.set_active`) instead of the uniform 1/k.

Reuses `ramole/router_lora.py` (RouterLoraLinear/extract/build/load) and `ramole/ramole_common.py`
(instruction encoder); reuses `legonet_tofu.py` for paths/assignment/q2author/KNN. The experts are
the frozen TOFU pool; `legonet_full`/`legonet_unlearn` differ only in which adapter dirs are served
(`adapter_dir_fn`), the router is never retrained.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

import legonet_tofu as lt

# import the RAMoLE package (sibling of tofu_sisa_lora)
RAMOLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ramole")
if RAMOLE_DIR not in sys.path:
    sys.path.insert(0, RAMOLE_DIR)
import ramole_common as rc       # noqa: E402
import router_lora as R          # noqa: E402


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

_DEFAULT_INSTR = "Represent the sentence for similar task retrieval"
_DEFAULT_ENCODER = "hkunlp/instructor-xl"


# ── RAMoLE-on-TOFU artifact paths ({output_dir}/legonet/ramole/) ───────────────

def ramole_dir(cfg: dict) -> str:
    return os.path.join(lt.legonet_dir(cfg), "ramole")


def expert_index_path(cfg: dict, suffix: str = "") -> str:
    """suffix distinguishes rebuilt indices (e.g. '_exforget10') from the stale default.
    Distinct filenames are load-bearing: on-disk results depend on the stale file."""
    return os.path.join(ramole_dir(cfg), f"expert_index_n{cfg['n']}{suffix}.npy")


def router_path(cfg: dict) -> str:
    return os.path.join(ramole_dir(cfg), "router.safetensors")


def retriever_dir(cfg: dict) -> str:
    return os.path.join(ramole_dir(cfg), "retriever")


def _instr(cfg: dict) -> str:
    return cfg.get("instruction", _DEFAULT_INSTR)


def _encoder_name(cfg: dict) -> str:
    return cfg.get("retriever_encoder", _DEFAULT_ENCODER)


def _encoder_source(cfg: dict) -> str:
    """Resolve the retriever encoder. cfg['encoder_pin'] makes the choice explicit:
      'base' -> always the base encoder id (true off-the-shelf);
      'ft'   -> the fine-tuned retriever dir (error if absent);
      unset/'auto' -> FT dir if present else base (legacy behavior).
    E0 footgun this fixes: under 'auto', once a FT retriever exists every 'off-the-shelf' arm
    silently becomes the FT encoder — always record the resolved source in output JSONs."""
    pin = cfg.get("encoder_pin", "auto")
    d = retriever_dir(cfg)
    if pin == "base":
        return _encoder_name(cfg)
    if pin == "ft":
        if not os.path.isdir(d):
            raise RuntimeError(f"encoder_pin=ft but no fine-tuned retriever at {d}")
        return d
    return d if os.path.isdir(d) else _encoder_name(cfg)


# ── Stage 1: the LoRA-retriever index over TOFU author-experts ─────────────────

def build_expert_index(cfg: dict, data_full, device: str = "cpu", encoder=None,
                       force: bool = False, exclude_authors=None) -> np.ndarray:
    """(n, D) L2-normalized expert embeddings: expert j = mean of its member authors' question
    embeddings (instruction-prefixed). Default: ALL authors (the frozen-keys analogue), cached to
    the stale path. `exclude_authors` (E3 rebuilt-retain-only policy) drops those authors from
    every expert's member mean and caches to a DISTINCT `_ex{...}` file — never overwrites the
    stale index that existing results depend on.

    The cache filename must also encode the ENCODER pin: the stale `expert_index_n{n}.npy` was
    rebuilt in place with the FT retriever on 2026-06-29, so an `encoder_pin:"base"` run that
    hit that cache would silently return FT embeddings (the E0 footgun at the index layer, not
    just the query layer). `encoder_pin:"base"` therefore caches to `..._encbase[...].npy`;
    'ft'/'auto' keep the legacy names their on-disk results depend on."""
    excl = sorted(int(a) for a in (exclude_authors or []))
    enc_tag = "_encbase" if cfg.get("encoder_pin") == "base" else ""
    suffix = enc_tag + (f"_ex{excl[0]}-{excl[-1]}n{len(excl)}" if excl else "")
    path = expert_index_path(cfg, suffix)
    if os.path.exists(path) and not force:
        return np.load(path)
    with open(lt.assignment_path(cfg)) as f:
        assignment = json.load(f)
    embed = rc.make_embed_fn(_encoder_source(cfg), instruction=_instr(cfg), device=device, encoder=encoder)
    A, per = cfg["num_authors"], cfg["records_per_author"]
    questions = [data_full[i]["question"] for i in range(A * per)]
    qv = embed(questions)                                   # (A*per, D) normalized
    author_v = np.stack([qv[a * per:(a + 1) * per].mean(0) for a in range(A)])
    n = cfg["n"]
    excl_set = set(excl)
    idx = np.zeros((n, qv.shape[1]), dtype="float32")
    for j in range(n):
        members = [int(a) for a in assignment["members"][str(j)] if int(a) not in excl_set]
        if not members and excl:
            raise RuntimeError(f"expert {j}: exclude_authors empties its member set "
                               "(a zero row would silently outrank negative sims)")
        if members:
            idx[j] = author_v[members].mean(0)
    idx = idx / (np.linalg.norm(idx, axis=1, keepdims=True) + 1e-12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, idx.astype("float32"))
    print(f"[ramole_tofu] expert index {idx.shape} (excl={len(excl)}) -> {path}", flush=True)
    return idx.astype("float32")


# ── Routing (embed = RAG, key = LegoNet author lookup) ─────────────────────────

class RamoleRouter:
    """Per-query → tuple of top-k expert ids. `embed`: cosine over the expert index; `key`: the
    LegoNet q2author-exact lookup + KNN-over-frozen-keys fallback (comparison arm)."""

    def __init__(self, mode, k, loaded, *, index=None, qembed=None,
                 keys=None, assignment=None, q2author=None, embed_fn=None):
        self.mode = mode
        self.k = int(k)
        self.loaded = set(loaded)
        self.cache = {}
        if mode == "embed":
            self.index = np.asarray(index, dtype="float32")     # (n, D) normalized
            self.qembed = qembed                                # embed(list[str]) -> (N,D) normalized
        elif mode == "key":
            self._knn = lt.KNNRouter(np.asarray(keys, dtype="float32"), k)
            self.assignment = assignment
            self.q2author = q2author
            self.embed_fn = embed_fn
        else:
            raise ValueError(f"route mode must be embed|key, got {mode!r}")

    def route(self, text: str) -> tuple:
        cache_key = lt._norm(text)
        hit = self.cache.get(cache_key)
        if hit is not None:
            return hit
        q = lt.parse_question(text)
        if self.mode == "embed":
            v = self.qembed([q if q is not None else text])[0]   # (D,) normalized
            sims = self.index @ v
            idxs = [int(j) for j in np.argsort(-sims)[: self.k]]
        else:  # key
            author = self.q2author.get(lt._norm(q)) if q is not None else None
            if author is not None:
                idxs = lt.author_keys(self.assignment, int(author))
            else:
                idxs = self._knn.route_one(self.embed_fn(q if q is not None else text))
        idxs = tuple(sorted(int(j) for j in idxs if j in self.loaded))
        if not idxs:
            idxs = (sorted(self.loaded)[0],)
        self.cache[cache_key] = idxs
        return idxs


# ── Drop-in routed eval model (mirrors LegoNetRoutedModel) ─────────────────────

class RamoleTofuModel(nn.Module):
    def __init__(self, model, tokenizer, controller, router: RamoleRouter):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.controller = controller
        self.router = router

    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):
        pass  # routing happens inside forward/generate via the controller

    def _route_text(self, ids_1d) -> str:
        return self.tokenizer.decode(ids_1d, skip_special_tokens=True) if self.tokenizer else ""

    def _activate(self, idxs):
        self.controller.set_active(list(idxs))   # RouterLoRA composes these experts

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B = input_ids.shape[0]
        if B == 1:
            self._activate(self.router.route(self._route_text(input_ids[0])))
            return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        all_logits = []
        total_loss_sum, total_tokens = 0.0, 0
        for i in range(B):
            inp_i = input_ids[i:i + 1]
            mask_i = attention_mask[i:i + 1] if attention_mask is not None else None
            lab_i = labels[i:i + 1] if labels is not None else None
            self._activate(self.router.route(self._route_text(inp_i[0])))
            out_i = self.model(inp_i, attention_mask=mask_i, labels=lab_i, **kwargs)
            all_logits.append(out_i.logits)
            if out_i.loss is not None and lab_i is not None:
                n_tok = (lab_i != -100).sum().item()
                total_loss_sum += out_i.loss.item() * n_tok
                total_tokens += n_tok
        logits = torch.cat(all_logits, dim=0)
        loss = (torch.tensor(total_loss_sum / total_tokens, device=input_ids.device)
                if total_tokens > 0 else None)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids, **kwargs):
        self._activate(self.router.route(self._route_text(input_ids[0])))
        return self.model.generate(input_ids, **kwargs)


# ── Build the eval model (mirrors legonet_model.load_legonet_eval_model) ───────

def _ramole_build_cfg(cfg_l: dict) -> dict:
    """The minimal cfg that router_lora.build_ramole_model needs from a legonet-TOFU config."""
    return {
        "base_model": cfg_l["base_model"],
        "base_seed": cfg_l.get("base_seed", 42),
        "n": cfg_l["n"],
        "router": {"rank": cfg_l.get("router", {}).get("rank", cfg_l["lora"]["rank"])},
    }


def _full_adapter_dir_fn(cfg_l):
    return lambda j: lt.adapter_dir(cfg_l, j)


def _unlearn_adapter_dir_fn(cfg_l, tag):
    with open(lt.unlearn_manifest_path(cfg_l, tag)) as f:
        manifest = json.load(f)
    retr = {int(j): d for j, d in manifest["retrained_dirs"].items()}
    return lambda j: retr.get(j, lt.adapter_dir(cfg_l, j))


def load_ramole_eval_model(cfg_l, data_full, router_ckpt, route_mode="embed", unlearn_tag=None,
                           index_policy="stale"):
    """(RamoleTofuModel, tokenizer). Frozen base + spliced RouterLoraLinear (experts from the TOFU
    pool, or its post-deletion mix for unlearn_tag) + the trained router; routing via embed (RAG) or
    key. The router checkpoint is loaded UNCHANGED for both full and unlearn.
    index_policy (embed route only): 'stale' = the as-built index (includes forget authors — the
    encoder-centroid-leak arm); 'rebuilt' = member means excluding cfg_l['forget_authors']."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_dir_fn = (_unlearn_adapter_dir_fn(cfg_l, unlearn_tag) if unlearn_tag
                      else _full_adapter_dir_fn(cfg_l))
    model, tok, controller, meta, _ = R.build_ramole_model(
        _ramole_build_cfg(cfg_l), device=dev, load_router_weights=False, adapter_dir_fn=adapter_dir_fn)
    R.load_router(model, router_ckpt)
    model.eval()
    model.config.use_cache = True
    loaded = set(range(cfg_l["n"]))

    if route_mode == "key":
        from sentence_transformers import SentenceTransformer
        keys = np.load(lt.keys_path(cfg_l))
        with open(lt.assignment_path(cfg_l)) as f:
            assignment = json.load(f)
        q2author = lt.build_q2author(data_full, cfg_l["num_authors"], cfg_l["records_per_author"])
        sb = SentenceTransformer(cfg_l["encoder_model"], device="cpu")   # MiniLM, matches the keys
        embed_fn = lambda t: sb.encode(t, normalize_embeddings=True).astype("float32")
        router = RamoleRouter("key", cfg_l["k"], loaded, keys=keys, assignment=assignment,
                              q2author=q2author, embed_fn=embed_fn)
    else:  # embed — the RAG path
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(_encoder_source(cfg_l), device=dev)    # per encoder_pin (E0)
        excl = cfg_l["forget_authors"] if index_policy == "rebuilt" else None
        index = build_expert_index(cfg_l, data_full, device=dev, encoder=enc, exclude_authors=excl)
        qembed = rc.make_embed_fn(_encoder_name(cfg_l), instruction=_instr(cfg_l), device=dev, encoder=enc)
        router = RamoleRouter("embed", cfg_l["k"], loaded, index=index, qembed=qembed)

    return RamoleTofuModel(model, tok, controller, router), tok


def main():
    """Build (and cache) the expert index — run once before the embed-routing evals so the
    parallel eval tasks don't each rebuild it / race on the cache."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--build_index", action="store_true")
    args = ap.parse_args()
    cfg = lt.load_config(args.config)
    os.environ["HF_HOME"] = cfg["hf_home"]
    if args.build_index:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
        data_full = load_dataset("locuslab/TOFU", "full")["train"]
        enc = SentenceTransformer(_encoder_source(cfg), device=args.device)   # fine-tuned dir if present
        build_expert_index(cfg, data_full, device=args.device, encoder=enc, force=True)


if __name__ == "__main__":
    main()
