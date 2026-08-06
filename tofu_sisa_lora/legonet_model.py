"""LegoNet eval model: per-query top-k adapter activation + 1/k delta-average.

`LegoNetRoutedModel` is a drop-in for the PeftModel that `eval_tofu.evaluate_model`
passes around: it mirrors `router.RoutedModel`'s forward/generate/config/eval seam,
but instead of selecting one shard it activates the **delta-average of the query's
top-k adapters** via PEFT `add_weighted_adapter(combination_type="linear", w=1/k)`
(ported from legonet_lora/combine.py). Records routing to the same adapter-set share
one cached merge.

Routing (author-level, frozen):
  - TOFU-author queries (forget/retain): resolve the question -> author id ->
    the author's frozen top-k assignment. Consistent before and after unlearning
    (frozen keys => cascade-free).
  - OOD queries (real_authors / world_facts): nearest-cluster of the question's own
    MiniLM embedding (heuristic fallback; affects only utility's OOD components).

`legonet_full` and `legonet_unlearn` differ only in WHICH adapter dirs are loaded
(originals vs the unlearn/{tag} retrains for affected adapters) — the routing and
combine rule are identical.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import legonet_tofu as lt

_MERGE_PREFIX = "_lego"


# ── Loading ───────────────────────────────────────────────────────────────────

def _full_adapter_dir_fn(cfg):
    return lambda j: lt.adapter_dir(cfg, j)


def _unlearn_adapter_dir_fn(cfg, tag):
    """adapter_dir_fn(j) -> retrained unlearn dir for affected adapters, original else."""
    with open(lt.unlearn_manifest_path(cfg, tag)) as f:
        manifest = json.load(f)
    retr = {int(j): d for j, d in manifest["retrained_dirs"].items()}
    return lambda j: retr.get(j, lt.adapter_dir(cfg, j))


def load_legonet_adapters(cfg, adapter_dir_fn):
    """Load the frozen base + adapters a0..a{n-1} into one PeftModel.

    Skips any adapter dir that is missing on disk (the loader returns the set of
    indices actually resident). Adapter weights are cast to fp32 by PEFT.
    """
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=dtype,
        device_map="auto" if use_cuda else None, trust_remote_code=True,
    )
    model = None
    loaded = set()
    for j in range(cfg["n"]):
        d = adapter_dir_fn(j)
        if not os.path.isdir(d):
            print(f"[legonet] adapter a{j} dir missing ({d}); skipping", flush=True)
            continue
        name = f"a{j}"
        if model is None:
            model = PeftModel.from_pretrained(base, d, adapter_name=name)
        else:
            model.load_adapter(d, adapter_name=name)
        loaded.add(j)
    if model is None:
        raise RuntimeError("legonet: no adapters loaded")
    model.eval()
    return model, tok, loaded


def load_legonet_eval_model(cfg, data_full, unlearn_tag=None, merge_cap=96):
    """Build the (LegoNetRoutedModel, tokenizer) used by eval_tofu.

    unlearn_tag=None -> legonet_full (originals); else legonet_unlearn (mixed dirs).
    """
    keys = np.load(lt.keys_path(cfg))
    with open(lt.assignment_path(cfg)) as f:
        assignment = json.load(f)
    q2author = lt.build_q2author(data_full, cfg["num_authors"], cfg["records_per_author"])

    adapter_dir_fn = (_unlearn_adapter_dir_fn(cfg, unlearn_tag) if unlearn_tag
                      else _full_adapter_dir_fn(cfg))
    model, tok, loaded = load_legonet_adapters(cfg, adapter_dir_fn)

    # OOD encoder on CPU so it never competes with the LLM for GPU memory.
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer(cfg["encoder_model"], device="cpu")

    def embed_fn(text):
        return _sbert.encode(text, normalize_embeddings=True).astype("float32")

    wrapper = LegoNetRoutedModel(
        model, tok, keys=keys, assignment=assignment, q2author=q2author,
        k=int(cfg["k"]), embed_fn=embed_fn, loaded=loaded, merge_cap=merge_cap,
    )
    return wrapper, tok


# ── Routed eval model ─────────────────────────────────────────────────────────

class LegoNetRoutedModel(nn.Module):
    """Per-sample top-k delta-average; drop-in for the PeftModel in eval_tofu."""

    def __init__(self, model, tokenizer, keys, assignment, q2author, k, embed_fn,
                 loaded, merge_cap=96):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.keys = np.asarray(keys, dtype="float32")
        self.assignment = assignment
        self.q2author = q2author
        self.k = k
        self.embed_fn = embed_fn
        self.loaded = set(loaded)
        self.merge_cap = merge_cap
        self._knn = lt.KNNRouter(self.keys, k)
        self._route_cache = {}     # normalized text -> tuple(sorted idxs)
        self._merge_cache = {}     # tuple(idxs) -> merged adapter name
        self._merge_order = []     # FIFO for eviction

    # -- model surface eval_tofu relies on --
    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):
        pass  # adapter selection happens inside forward/generate

    # -- routing --
    def _route(self, text: str) -> tuple:
        key = lt._norm(text)
        cached = self._route_cache.get(key)
        if cached is not None:
            return cached
        q = lt.parse_question(text)
        author = self.q2author.get(lt._norm(q)) if q is not None else None
        if author is not None:
            idxs = lt.author_keys(self.assignment, int(author))
        else:
            vec = self.embed_fn(q if q is not None else text)
            idxs = self._knn.route_one(vec)
        idxs = tuple(sorted(j for j in idxs if j in self.loaded))
        if not idxs:  # extreme fallback: any loaded adapter
            idxs = (sorted(self.loaded)[0],)
        self._route_cache[key] = idxs
        return idxs

    def _activate(self, idxs: tuple):
        # Single adapter -> select directly (no merge, no extra fp32 copy).
        if len(idxs) == 1:
            self.model.set_adapter(f"a{idxs[0]}")
            return
        name = self._merge_cache.get(idxs)
        if name is None:
            if len(self._merge_cache) >= self.merge_cap:
                old = self._merge_order.pop(0)
                old_name = self._merge_cache.pop(old)
                try:
                    self.model.delete_adapter(old_name)
                except Exception:
                    pass
            name = f"{_MERGE_PREFIX}_" + "_".join(str(j) for j in idxs)
            w = [1.0 / len(idxs)] * len(idxs)
            self.model.add_weighted_adapter(
                adapters=[f"a{j}" for j in idxs], weights=w,
                adapter_name=name, combination_type="linear",
            )
            self._merge_cache[idxs] = name
            self._merge_order.append(idxs)
        self.model.set_adapter(name)

    def _route_text(self, input_ids_1d) -> str:
        return (self.tokenizer.decode(input_ids_1d, skip_special_tokens=True)
                if self.tokenizer is not None else "")

    # -- forward / generate (mirror router.RoutedModel) --
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B = input_ids.shape[0]
        if B == 1:
            self._activate(self._route(self._route_text(input_ids[0])))
            return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        all_logits = []
        total_loss_sum, total_tokens = 0.0, 0
        for i in range(B):
            inp_i = input_ids[i:i + 1]
            mask_i = attention_mask[i:i + 1] if attention_mask is not None else None
            lab_i = labels[i:i + 1] if labels is not None else None
            self._activate(self._route(self._route_text(inp_i[0])))
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
        self._activate(self._route(self._route_text(input_ids[0])))
        return self.model.generate(input_ids, **kwargs)
