"""ClAMU eval model: serve `θ0 + (m_c ⊙ τ̄)/T` per query (cluster-level masks).

`ClamuModel` is a drop-in for the PeftModel that `eval_tofu.evaluate_model` passes
around — same forward/generate/config seam as `sift_masks_model.SiftMasksModel` and
`legonet_model.LegoNetRoutedModel`. ClAMU is a full-model delta (not a PEFT adapter):
we apply the cluster-specific masked delta to the base weights in place, cached by the
currently-served cluster.

Routing (oracle, the paper's "task identity known at serve time"):
  * TOFU-author query -> author id (legonet_tofu.build_q2author) -> its cluster -> that
    cluster's mask.
  * OOD query (real_authors / world_facts) -> base θ0.
  * Unlearned forget-author query (not in the retain clustering) -> base θ0 (forgotten).

Labels (the localization ladder, set by eval_tofu via --label):
  clamu_full / clamu_unlearn : optimized cluster mask        (the method)
  emr_full   / emr_unlearn   : EMR sign-agreement mask       (heuristic baseline)
  tall_full  / tall_unlearn  : TALL threshold mask           (heuristic baseline)
  merge_full / merge_unlearn : no mask, θ0 + τ̄/T             (Global-merge baseline)

`*_full` vs `*_unlearn` differ only in WHICH τ̄ + assignment + masks are loaded (full
vs the post-deletion retain-only `_<tag>` artifacts). Reuses the SISA retain90 KS
reference for forget_quality — method-independent.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import clamu as cl
import legonet_tofu as lt
import sift_masks as sm

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


def _mask_kind(label: str) -> Optional[str]:
    """Map a label prefix to its on-disk mask family; 'merge' -> None (no mask)."""
    prefix = label.split("_", 1)[0]
    if prefix == "merge":
        return None
    if prefix in cl.MASK_KINDS:
        return prefix
    raise SystemExit(f"clamu: label '{label}' must start with one of "
                     f"{cl.MASK_KINDS + ('merge',)}_  (e.g. clamu_full, merge_unlearn)")


def load_clamu_eval_model(cfg, data_full, label: str, unlearn_tag: Optional[str] = None):
    """Build (ClamuModel, tokenizer) for eval_tofu.

    unlearn_tag=None -> full (all authors, τ̄, masks/); else post-deletion (retain
    authors, τ̄_<tag>, masks_<tag>/, assignment_<tag>). The mask family is chosen from
    the label prefix (clamu/emr/tall/merge).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["HF_HOME"] = cfg["hf_home"]
    kind = _mask_kind(label)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(            # fp32, match training
        cfg["model_name"], torch_dtype=torch.float32, trust_remote_code=True).to(device)
    base.eval()

    names = sm.trainable_names(base, tuple(cfg.get("frozen_substr", sm.GPT2_FROZEN_SUBSTR)))
    theta0 = sm.snapshot_params(base, names)

    with open(cl.assignment_path(cfg, unlearn_tag)) as f:
        assignment = json.load(f)
    author_to_cluster = {int(a): int(c) for a, c in assignment["author_to_cluster"].items()}
    served = set(author_to_cluster.keys())                 # authors with a delta
    T = len(assignment["authors"])                         # /T uses the served-author count
    tau_bar = {n: v.to(device) for n, v in torch.load(cl.tau_bar_path(cfg, unlearn_tag)).items()}

    q2author = lt.build_q2author(data_full, cfg["num_authors"], cfg["records_per_author"])
    wrapper = ClamuModel(
        base, tok, names=names, theta0=theta0, tau_bar=tau_bar, T=T,
        author_to_cluster=author_to_cluster, served=served, q2author=q2author,
        mask_dir=cl.mask_dir(cfg, unlearn_tag), mask_kind=kind,
        forgotten_serve=cfg.get("forgotten_serve", "base"))
    return wrapper, tok


class ClamuModel(nn.Module):
    """Per-query cluster-masked-delta serving; drop-in for the PeftModel in eval_tofu."""

    def __init__(self, model, tokenizer, *, names, theta0, tau_bar, T,
                 author_to_cluster, served, q2author, mask_dir, mask_kind,
                 forgotten_serve="base"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.names = names
        self.theta0 = theta0
        self.tau_bar = tau_bar
        self.T = int(T)
        self.author_to_cluster = author_to_cluster
        self.served = set(int(a) for a in served)
        self.q2author = q2author
        self.mask_dir = mask_dir
        self.mask_kind = mask_kind                 # None -> no-mask (Global merge) baseline
        # "base": forgotten authors -> raw θ0 (07-02 headline behavior).
        # "merged": forgotten authors -> maskless retain merge θ0 + τ̄_<tag>/T — the papers'
        # Fig-8 protocol (H8-align): still exact (τ̄_<tag> is a function of retain data only),
        # but its truth-ratio distribution is oracle-comparable, so extended-cap
        # forget_quality stops measuring the base-vs-oracle style artifact.
        assert forgotten_serve in ("base", "merged")
        self.forgotten_serve = forgotten_serve
        self._applied = None                       # cache key of currently-applied weights
        self._mask_cache = {}                      # cluster -> unpacked mask (LRU of 1)

    # -- surface eval_tofu relies on --
    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):                   # selection happens in forward/generate
        pass

    # -- routing --
    def _route(self, text: str):
        q = lt.parse_question(text)
        if q is None:
            return None
        return self.q2author.get(lt._norm(q))      # author id or None (OOD)

    def _mask_for(self, cluster: int):
        m = self._mask_cache.get(cluster)
        if m is None:
            self._mask_cache = {}                  # keep only one unpacked mask resident
            packed = torch.load(os.path.join(self.mask_dir, f"{self.mask_kind}_{cluster}.pt"))
            m = sm.unpack_mask(packed, self.names)
            self._mask_cache[cluster] = m
        return m

    def _apply(self, author):
        """Switch the base weights to serve `author` (cached; ~one apply per cluster)."""
        if author is None or author not in self.served:
            # Forgotten author (known id, no delta) under Fig-8 serving -> retain merge.
            # OOD (author is None) always stays base — changing it would shift the
            # real/world mu components and break comparability with every prior row.
            if author is not None and self.forgotten_serve == "merged":
                if self._applied != ("merge",):
                    sm.serve_merged_(self.model, self.theta0, self.tau_bar, self.names, self.T)
                    self._applied = ("merge",)
                return
            if self._applied != ("base",):        # OOD or forgotten -> base θ0
                sm.serve_base_(self.model, self.theta0, self.names)
                self._applied = ("base",)
            return
        if self.mask_kind is None:                 # Global-merge baseline (no mask)
            if self._applied != ("merge",):
                sm.serve_merged_(self.model, self.theta0, self.tau_bar, self.names, self.T)
                self._applied = ("merge",)
            return
        cluster = self.author_to_cluster[author]
        key = ("mask", cluster)
        if key != self._applied:
            sm.serve_task_(self.model, self.theta0, self.tau_bar,
                           self._mask_for(cluster), self.names, self.T)
            self._applied = key

    def _route_text(self, input_ids_1d) -> str:
        return (self.tokenizer.decode(input_ids_1d, skip_special_tokens=True)
                if self.tokenizer is not None else "")

    # -- forward / generate (mirror SiftMasksModel) --
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
