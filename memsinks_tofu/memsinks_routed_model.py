"""MemSinks routed-mask serving: per-query author mask, drop-in for the PeftModel
in eval_tofu (mirrors sift_masks_model.SiftMasksModel on the eval seam).

Routing (oracle, same "task identity known at serve time" assumption as SIFT):
  * TOFU-author query, not deleted -> gen + that author's sink slice (the
    TRAINING condition — the only mask configuration the model ever saw).
  * Deleted author (unlearn_tag -> cfg["forget_authors"]) -> gen-only: the
    author's slice is never applied. By the bake≡hook identity this is the
    routed view of the baked deletion.
  * OOD query (real_authors / world_facts; q2author miss) -> gen-only
    (documented choice, symmetric to SIFT's OOD -> θ0; gen-only is the
    "no author selected" state of this substrate).

Serving switches are MaskState.set_fixed() assignments — O(1), no weight
surgery; the per-neuron vector applies identically at every KV-cached decode
step (gate-tested in test_memsinks.py).

Loaded by eval_tofu.py --memsinks_config (and attack_mia.py, same flags); the
eval_tofu branch sys.path-inserts this project dir from the config path.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))
import legonet_tofu as lt

from memsinks_model import (
    MaskState,
    author_serve_vector,
    build_scale_vector,
    install_sink_hooks,
    load_masks,
)


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


def load_memsinks_eval_model(cfg, data_full, unlearn_tag: Optional[str] = None):
    """Build (MemSinksRoutedModel, tokenizer) for eval_tofu.

    cfg = the memsinks training config dict; run dir = cfg["output_dir"]
    (must contain trained/ with the adapter + sink_masks.pt).
    unlearn_tag "forget10" -> cfg["forget_authors"] served gen-only.
    """
    os.environ["HF_HOME"] = cfg.get("hf_home", os.environ.get("HF_HOME", ""))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    trained = os.path.join(cfg["output_dir"], "trained")
    model = PeftModel.from_pretrained(base, trained)
    model.eval()

    mask_table, num_gen = load_masks(trained)
    state = MaskState(mask_table, num_gen)
    install_sink_hooks(model, state, cfg["sink_modules"], model.config.num_hidden_layers)
    state.to(next(model.parameters()).device)

    if unlearn_tag is not None:
        if unlearn_tag != "forget10":
            raise ValueError(f"unknown unlearn_tag {unlearn_tag!r} (only 'forget10')")
        deleted = set(cfg["forget_authors"])
    else:
        deleted = set()

    q2author = lt.build_q2author(data_full, cfg["num_authors"], cfg["records_per_author"])
    return MemSinksRoutedModel(model, tok, state=state, q2author=q2author,
                               deleted=deleted), tok


class MemSinksRoutedModel(nn.Module):
    """Per-query routed-mask serving; drop-in for the PeftModel in eval_tofu."""

    def __init__(self, model, tokenizer, *, state: MaskState, q2author, deleted):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.state = state
        self.q2author = q2author
        self.deleted = set(int(a) for a in deleted)
        self._gen_only = build_scale_vector(state.mask_table.cpu(), state.num_gen, "dropall")
        self._vec_cache = {}            # author -> serve vector (CPU)
        self._applied = None

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

    def _apply(self, author):
        key = "gen" if (author is None or author in self.deleted) else int(author)
        if key == self._applied:
            return
        if key == "gen":
            v = self._gen_only
        else:
            v = self._vec_cache.get(key)
            if v is None:
                v = author_serve_vector(self.state.mask_table.cpu(), self.state.num_gen, key)
                self._vec_cache[key] = v
        self.state.set_fixed(v)
        self._applied = key

    def _route_text(self, input_ids_1d) -> str:
        return self.tokenizer.decode(input_ids_1d, skip_special_tokens=True)

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
