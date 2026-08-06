"""Inference: delta-average the k activated adapters over the frozen base.

Fixed combination rule (must be identical between the deployed and any reference
model): merge the k activated LoRA adapters with equal weight 1/k via PEFT
`add_weighted_adapter(combination_type="linear")` — one forward pass (plan §2,
§6). Exactness is a property of *training*, not of this rule.

Records that route to the same set of k adapters share one merge, so eval groups
by adapter-set and merges once per group.
"""
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from legonet_common import Paths

_MERGED = "_merged"


def route_groups(assignment: dict, record_ids: list[str]) -> dict:
    """{tuple(sorted k key idx): [record_id, ...]} for the given records."""
    groups = defaultdict(list)
    for rid in record_ids:
        key = tuple(sorted(assignment["record_to_keys"][rid]))
        groups[key].append(rid)
    return groups


class LegoNetModel:
    """Frozen base + a pool of loaded LoRA adapters; merges k on demand."""

    def __init__(self, model, tokenizer, loaded: set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.loaded = loaded

    @classmethod
    def from_config(cls, cfg: dict, adapter_idxs=None,
                    adapter_dir_fn=None, device_map: str | None = None):
        """Load base + the requested adapters (default: all n).

        `adapter_dir_fn(j) -> path` overrides where adapter j is read from (used
        to assemble a post-unlearn model from mixed original/retrained dirs).
        """
        paths = Paths(cfg)
        adapter_dir_fn = adapter_dir_fn or paths.adapter_dir
        idxs = list(range(cfg["n"])) if adapter_idxs is None else list(adapter_idxs)

        use_cuda = torch.cuda.is_available()
        if device_map is None:
            device_map = "auto" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32

        tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=dtype,
            device_map=device_map, trust_remote_code=True,
        )

        model = None
        loaded = set()
        for j in idxs:
            d = adapter_dir_fn(j)
            if not os.path.isdir(d):
                continue
            name = f"a{j}"
            if model is None:
                model = PeftModel.from_pretrained(base, d, adapter_name=name)
            else:
                model.load_adapter(d, adapter_name=name)
            loaded.add(j)
        if model is None:
            raise RuntimeError("no adapters loaded")
        model.eval()
        return cls(model, tok, loaded)

    @contextmanager
    def activated(self, idxs):
        """Activate the delta-average of adapters `idxs` (weights 1/k)."""
        idxs = [j for j in idxs if j in self.loaded]
        if not idxs:
            raise RuntimeError("none of the requested adapters are loaded")
        names = [f"a{j}" for j in idxs]
        w = [1.0 / len(names)] * len(names)
        self.model.add_weighted_adapter(
            adapters=names, weights=w, adapter_name=_MERGED,
            combination_type="linear",
        )
        self.model.set_adapter(_MERGED)
        try:
            yield self.model
        finally:
            self.model.delete_adapter(_MERGED)
