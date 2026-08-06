"""Prefix-concatenation serving for the peft_compose bake-off.

Composition rule for the prefix-tuning arm: instead of averaging weights, CONCATENATE every
shard's trained KV prefix in the cache — the base model then attends over all shards' virtual
tokens at once and routes by attention (no trained router). Deletion = drop a shard's segment
from the concatenation: byte-exact by construction, O(1).

`PrefixConcatModel` is a drop-in for the PeftModel in eval_tofu (same contract as RoutedModel /
EnsembleModel): forward(input_ids, attention_mask, labels) -> output with .loss/.logits, greedy
batch-1 generate(), .config, parameters()/eval() via nn.Module. The prefix cache is rebuilt per
call (get_prompt is one tiny MLP/embedding forward per shard) and the real tokens' positions
start after the concatenated prefix — the same convention peft's own single-prefix serving uses.

CLI smoke (loads the pool, one forward, prints loss):
    python prefix_concat.py --model_name M --pool_dir DIR --k 10 [--exclude 9]
"""
from __future__ import annotations

import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from peft import PeftModel


def _legacy_prompt_kv(peft_model, adapter_name: str, batch_size: int):
    """This adapter's prefix past-KV in legacy layout: list over layers of (k, v),
    each (B, n_kv_heads, num_virtual_tokens, head_dim)."""
    peft_model.set_adapter(adapter_name)
    pkv = peft_model.get_prompt(batch_size=batch_size)
    if isinstance(pkv, Cache):
        pkv = pkv.to_legacy_cache()
    # Entries are (k, v) tuples or stacked (2, B, H, S, D) tensors; index uniformly.
    return [(layer[0], layer[1]) for layer in pkv]


class PrefixConcatModel(nn.Module):
    """Frozen base + N per-shard KV prefixes concatenated in the cache."""

    def __init__(self, peft_model, shard_names: list, tokenizer=None):
        super().__init__()
        self.model = peft_model                      # PeftModel with all shard prefix adapters
        self.shards = list(shard_names)              # serving order (fixed, ascending)
        self.tokenizer = tokenizer
        self.base = peft_model.get_base_model()      # prefix adapters never touch base weights
        self.base_dtype = next(self.base.parameters()).dtype

    @property
    def config(self):
        return self.base.config

    def set_adapter(self, name: str) -> None:
        pass  # no-op: composition is fixed at construction (like RoutedModel)

    def _concat_prefix(self, batch_size: int):
        """(DynamicCache over all shards' prefixes in shard order, total prefix length)."""
        merged = None
        for name in self.shards:
            kv = _legacy_prompt_kv(self.model, name, batch_size)
            kv = [(k.to(self.base_dtype), v.to(self.base_dtype)) for k, v in kv]
            if merged is None:
                merged = kv
            else:
                merged = [
                    (torch.cat([mk, k], dim=2), torch.cat([mv, v], dim=2))
                    for (mk, mv), (k, v) in zip(merged, kv)
                ]
        total = merged[0][0].shape[2]
        return DynamicCache.from_legacy_cache(tuple(merged)), total

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B = input_ids.shape[0]
        cache, P = self._concat_prefix(B)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        full_mask = torch.cat(
            [torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device),
             attention_mask], dim=1)
        return self.base(input_ids=input_ids, attention_mask=full_mask, labels=labels,
                         past_key_values=cache, use_cache=True, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """Greedy batch-1 loop with the concatenated prefix as the initial cache
        (mirrors EnsembleModel._generate_sequential / eval_tofu's contract)."""
        assert input_ids is not None and input_ids.shape[0] == 1, "batch size 1 only"
        assert not kwargs.get("do_sample", False), "greedy decoding only"
        max_new_tokens = kwargs.get("max_new_tokens", 100)
        eos_id = kwargs.get("eos_token_id")
        if eos_id is None:
            eos_id = getattr(self.base.generation_config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_ids = set(int(e) for e in eos_id)
        elif eos_id is None:
            eos_ids = set()
        else:
            eos_ids = {int(eos_id)}

        self.base.eval()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            cache, P = self._concat_prefix(1)
            mask = torch.cat(
                [torch.ones(1, P, dtype=attention_mask.dtype, device=attention_mask.device),
                 attention_mask], dim=1)
            out = self.base(input_ids=input_ids, attention_mask=mask,
                            past_key_values=cache, use_cache=True)
            generated = []
            for _ in range(max_new_tokens):
                tok = out.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
                generated.append(tok)
                if int(tok.item()) in eos_ids:
                    break
                mask = torch.cat([mask, torch.ones_like(tok)], dim=1)
                out = self.base(input_ids=tok, attention_mask=mask,
                                past_key_values=out.past_key_values, use_cache=True)
        if generated:
            return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)
        return input_ids


def load_prefix_concat_model(model_name, pool_dir, k=10, exclude=(), device_map="auto",
                             torch_dtype=torch.bfloat16):
    """Base + every shard prefix adapter in pool_dir, minus `exclude` shard ids.
    Returns (PrefixConcatModel, tokenizer)."""
    exclude = set(exclude)
    shard_ids = [i for i in range(k)
                 if i not in exclude and os.path.isdir(os.path.join(pool_dir, f"shard_{i}"))]
    if not shard_ids:
        raise ValueError(f"no shard dirs found in {pool_dir}")
    missing = [i for i in range(k) if i not in exclude and i not in shard_ids]
    if missing:
        raise ValueError(f"prefix pool {pool_dir} is missing shards {missing} "
                         f"(silent skips would corrupt the composition)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True)
    first = shard_ids[0]
    model = PeftModel.from_pretrained(
        base, os.path.join(pool_dir, f"shard_{first}"), adapter_name=f"shard_{first}")
    for i in shard_ids[1:]:
        model.load_adapter(os.path.join(pool_dir, f"shard_{i}"), adapter_name=f"shard_{i}")
    wrapper = PrefixConcatModel(model, [f"shard_{i}" for i in shard_ids], tokenizer=tokenizer)
    return wrapper, tokenizer


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    args = ap.parse_args()
    m, tok = load_prefix_concat_model(args.model_name, args.pool_dir, k=args.k,
                                      exclude=args.exclude)
    enc = tok("Question: Who wrote it?\nAnswer: Nobody.", return_tensors="pt")
    enc = {k: v.to(next(m.parameters()).device) for k, v in enc.items()}
    out = m(**enc, labels=enc["input_ids"])
    print(f"[prefix_concat] shards={m.shards} loss={out.loss.item():.4f}")
