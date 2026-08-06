"""SEA-on-TOFU inference: frozen 4-bit base + per-author deletable proxy.

Minimal-faithful SEA: the only user-specific component is the personal LoRA proxy. The
expert/routing/steering layers have no TOFU analog (SEA_on_TOFU.md §3.1), so they are
omitted here; sea/sea_model.py keeps the full pipeline for the optional ablation.

Key invariant (the guide's #1 pitfall): never let adapters accumulate. SeaProxyModel keeps
at most one author's adapter resident and deletes the previous one on each attach, so
loading author A then author B can never leak A into B's outputs.

"Omission mode" == proxy not active == structurally identical to post-deletion, so deletion
behavior is verifiable before the irreversible rm.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def load_base(model_name="meta-llama/Llama-2-7B-chat-hf", hf_home=None, device_map="auto"):
    """Load the frozen 4-bit NF4 base (matches sea/sea_model + SEA paper). Returns (model, tok)."""
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map=device_map, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # base is frozen, always
    return model, tok


class SeaProxyModel:
    """Wraps the frozen base and swaps a single author proxy in/out.

    Usage:
        sea = SeaProxyModel(*load_base(...))
        sea.attach(author_id, lora_dir)          # personalized: that author's proxy active
        get_rouge(sea.model, sea.tokenizer, ...)
        with sea.omission() as m:                # base-only (== deleted)
            get_rouge(m, sea.tokenizer, ...)
    """

    def __init__(self, base, tokenizer):
        from peft import PeftModel  # local import keeps base load light
        self._PeftModel = PeftModel
        self.base = base
        self.tokenizer = tokenizer
        self.peft = None          # PeftModel once a proxy is first attached
        self.active = None        # currently active adapter name
        self._resident = set()    # adapter names held in memory

    @property
    def model(self):
        """The object metric primitives consume; PeftModel with active proxy, else base."""
        return self.peft if self.peft is not None else self.base

    def _adapter_name(self, author_id):
        return f"author_{author_id:03d}"

    def attach(self, author_id, lora_dir, keep_only_active=True):
        """Make author_id's proxy the active adapter. Drops the previous one by default."""
        name = self._adapter_name(author_id)
        if self.peft is None:
            self.peft = self._PeftModel.from_pretrained(self.base, lora_dir, adapter_name=name)
            self._resident = {name}
        elif name not in self._resident:
            self.peft.load_adapter(lora_dir, adapter_name=name)
            self._resident.add(name)
        self.peft.set_adapter(name)
        self.active = name
        if keep_only_active:
            for other in list(self._resident):
                if other != name:
                    self._delete(other)
        return self.peft

    def _delete(self, name):
        try:
            self.peft.delete_adapter(name)
        except Exception:
            pass
        self._resident.discard(name)

    @contextmanager
    def omission(self):
        """Base-only behavior (proxy not loaded). Identical to post-deletion output."""
        if self.peft is None:
            yield self.base
        else:
            with self.peft.disable_adapter():
                yield self.peft


@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=200):
    """Greedy generation in the TOFU eval prompt format (Question:/Answer:)."""
    prompt = f"Question: {question}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
