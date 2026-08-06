"""SIFT-Masks eval model: serve `θ0 + (τ̄⊙m_a)/T` per query, on the repo's OU metrics.

`SiftMasksModel` is a drop-in for the PeftModel that `eval_tofu.evaluate_model` passes
around — it mirrors `legonet_model.LegoNetRoutedModel`'s forward/generate/config/eval
seam. SIFT-Masks is NOT a PEFT adapter (it's a full-model delta), so instead of
selecting an adapter we apply the task-specific masked delta to the base weights
in place, cached by the current task.

Routing (oracle, exactly the paper's "task identity is known at serve time"):
  * TOFU-author query  -> author id via legonet_tofu.build_q2author -> that author's mask.
  * OOD query (real_authors / world_facts, no task) -> base θ0 (documented choice).
  * Unlearned forget-author query (mask dropped) -> the maskless merged model
    θ0 + τ̄_tag/T' (paper Fig 8 held-out rule; H8 fix 2026-07-02 — was base θ0).

Labels (set by eval_tofu via the served condition):
  sift_full / sift_unlearn   : masked serve   θ0 + (τ̄⊙m_a)/T          (the method)
  merge_full / merge_unlearn : FT+Merge no-mask θ0 + τ̄/T  for served authors,
                               exposing the at-scale collapse the mask fixes.

`*_full` vs `*_unlearn` differ only in WHICH τ̄ + author set is loaded (τ̄ vs
τ̄_<tag>; all 200 vs retain-only). Reuses the existing retain90 KS reference for
forget_quality — method-independent.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import legonet_tofu as lt
import sift_masks as sm
from train_sift_masks import sift_dir

import sys

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


# ── loading ─────────────────────────────────────────────────────────────────────

def load_sift_eval_model(cfg, data_full, unlearn_tag: Optional[str] = None,
                         baseline: bool = False):
    """Build (SiftMasksModel, tokenizer) for eval_tofu.

    unlearn_tag=None -> full (all 200 authors, τ̄); else post-deletion (retain authors,
    τ̄_<tag>). baseline=True serves the FT+Merge no-mask model for served authors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["HF_HOME"] = cfg["hf_home"]

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    # fp32 to match training and keep the masked-delta application exact.
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32, trust_remote_code=True).to(device)
    base.eval()

    names = sm.trainable_names(base, tuple(cfg.get("frozen_substr", sm.GPT2_FROZEN_SUBSTR)))
    theta0 = sm.snapshot_params(base, names)            # on device

    sdir = sift_dir(cfg)
    if unlearn_tag:
        manifest = json.load(open(os.path.join(sdir, f"unlearn_{unlearn_tag}.json")))
        forgotten = set(manifest["forgotten_authors"])
        T = manifest["num_authors_after"]
        tau_path = os.path.join(sdir, f"tau_bar_{unlearn_tag}.pt")
    else:
        forgotten = set()
        T = cfg["num_authors"]
        tau_path = os.path.join(sdir, "tau_bar.pt")
    tau_bar = {n: v.to(device) for n, v in torch.load(tau_path).items()}
    served = set(range(cfg["num_authors"])) - forgotten      # authors that get a delta

    q2author = lt.build_q2author(data_full, cfg["num_authors"], cfg["records_per_author"])
    wrapper = SiftMasksModel(
        base, tok, names=names, theta0=theta0, tau_bar=tau_bar, T=T,
        served=served, q2author=q2author, sift_dir=sdir, baseline=baseline)
    return wrapper, tok


# ── served eval model ────────────────────────────────────────────────────────────

class SiftMasksModel(nn.Module):
    """Per-query masked-delta serving; drop-in for the PeftModel in eval_tofu."""

    def __init__(self, model, tokenizer, *, names, theta0, tau_bar, T, served,
                 q2author, sift_dir, baseline=False):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.names = names
        self.theta0 = theta0
        self.tau_bar = tau_bar
        self.T = int(T)
        self.served = set(int(a) for a in served)
        self.q2author = q2author
        self.sift_dir = sift_dir
        self.baseline = baseline
        self._applied = None            # cache key of the currently-applied weights
        self._mask_cache = {}           # author -> unpacked mask dict (LRU of 1)

    # -- surface eval_tofu relies on --
    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):        # selection happens in forward/generate
        pass

    # -- routing --
    def _route(self, text: str):
        q = lt.parse_question(text)
        if q is None:
            return None
        return self.q2author.get(lt._norm(q))      # author id or None (OOD)

    def _mask_for(self, author: int):
        m = self._mask_cache.get(author)
        if m is None:
            self._mask_cache = {}                  # keep only one unpacked mask resident
            packed = torch.load(mask_path_for(self.sift_dir, author))
            m = sm.unpack_mask(packed, self.names)
            self._mask_cache[author] = m
        return m

    def _apply(self, author):
        """Switch the base weights to serve `author` (cached; ~one apply per author)."""
        if author is None:                          # OOD -> base θ0 (documented choice)
            if self._applied != ("base",):
                sm.serve_base_(self.model, self.theta0, self.names)
                self._applied = ("base",)
        elif author in self.served and not self.baseline:
            key = ("mask", author)
            if key != self._applied:
                sm.serve_task_(self.model, self.theta0, self.tau_bar,
                               self._mask_for(author), self.names, self.T)
                self._applied = key
        else:
            # baseline (FT+Merge serves its one model for every task) or FORGOTTEN
            # author. Paper Fig 8: "To evaluate a task which has already been unlearned,
            # SIFT-Masks applies the merged model without any mask" -> θ0 + τ̄/T.
            # (Pre-H8 this branch served forgotten authors base θ0; at extended caps the
            # n=120 KS distinguishes raw-base style from the retain-finetuned oracle,
            # tanking forget_quality to 0.0045 vs legonet's 0.89 — a reference artifact,
            # see log/sift_masks/2026-07-02_extended-caps.md.)
            key = ("merge",)
            if key != self._applied:
                sm.serve_merged_(self.model, self.theta0, self.tau_bar, self.names, self.T)
                self._applied = key

    def _route_text(self, input_ids_1d) -> str:
        return (self.tokenizer.decode(input_ids_1d, skip_special_tokens=True)
                if self.tokenizer is not None else "")

    # -- forward / generate (mirror LegoNetRoutedModel) --
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B = input_ids.shape[0]
        if B == 1:
            self._apply(self._route(self._route_text(input_ids[0])))
            return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        all_logits, loss_sum, tok_total = [], 0.0, 0
        for i in range(B):
            inp = input_ids[i:i + 1]
            am = attention_mask[i:i + 1] if attention_mask is not None else None
            lab = labels[i:i + 1] if labels is not None else None
            self._apply(self._route(self._route_text(inp[0])))
            out = self.model(inp, attention_mask=am, labels=lab, **kwargs)
            all_logits.append(out.logits)
            if out.loss is not None and lab is not None:
                n = (lab != -100).sum().item()
                loss_sum += out.loss.item() * n
                tok_total += n
        loss = (torch.tensor(loss_sum / tok_total, device=input_ids.device)
                if tok_total > 0 else None)
        return CausalLMOutputWithPast(loss=loss, logits=torch.cat(all_logits, dim=0))

    def generate(self, input_ids, **kwargs):
        self._apply(self._route(self._route_text(input_ids[0])))
        return self.model.generate(input_ids, **kwargs)


def mask_path_for(sdir: str, author: int) -> str:
    return os.path.join(sdir, "masks", f"m_{author}.pt")
