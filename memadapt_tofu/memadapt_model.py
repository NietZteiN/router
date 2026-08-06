"""Adapter installation, sparse checkpoints, and the eval-loadable model class.

The same ProductKeyMemory module file backs training (test-env, torch 2.5.1,
transformers 4.48.3) and evaluation inside open-unlearning (unlearning env,
torch 2.4.1, transformers 4.51.3) — routing code is never reimplemented, so
fp32 routing decisions are identical in both. Imports: torch + transformers
+ stdlib only.
"""

import json
import os
import time
from typing import Optional

import torch
from torch import nn
from transformers import LlamaForCausalLM

from memadapt_common import assignment_sha, save_json
from memory_layer import ProductKeyMemory

CHECKPOINT_NAME = "memadapt.pt"


class MemAdaptMLP(nn.Module):
    """MemAdapt(x) = MLP(x) + Memory(x) on one decoder layer's MLP.

    Batch state (training source ids, per-row block-lists) is set via explicit
    methods rather than forward kwargs — HF's decoder layer calls mlp(x)
    positionally and would drop anything else.
    """

    def __init__(self, mlp: nn.Module, memory: ProductKeyMemory):
        super().__init__()
        self.mlp = mlp
        self.memory = memory
        self._source_ids: Optional[torch.Tensor] = None
        self._blocked_sources: Optional[torch.Tensor] = None

    def set_batch_sources(self, source_ids: Optional[torch.Tensor]):
        self._source_ids = source_ids

    def set_batch_blocked_sources(self, blocked_sources: Optional[torch.Tensor]):
        """(B, num_sources) bool — per-query block-lists within one batch."""
        self._blocked_sources = blocked_sources

    def clear(self):
        self._source_ids = None
        self._blocked_sources = None

    def forward(self, x):
        out = self.mlp(x)
        if self.memory.values is not None:
            out = out + self.memory(
                x, source_ids=self._source_ids,
                blocked_sources=self._blocked_sources,
            )
        return out


def install_adapter(model, memory: ProductKeyMemory, layer_idx: int) -> MemAdaptMLP:
    layer = model.model.layers[layer_idx]
    assert not isinstance(layer.mlp, MemAdaptMLP), "adapter already installed"
    wrapper = MemAdaptMLP(layer.mlp, memory)
    layer.mlp = wrapper
    return wrapper


def get_adapter(model) -> MemAdaptMLP:
    for module in model.modules():
        if isinstance(module, MemAdaptMLP):
            return module
    raise ValueError("no MemAdaptMLP installed on this model")


def freeze_base(model, memory: ProductKeyMemory):
    """Freeze everything except the memory values (router is already frozen)."""
    for p in model.parameters():
        p.requires_grad_(False)
    memory.values.requires_grad_(True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 1 and trainable[0].endswith("memory.values"), trainable
    return trainable


# ---------------------------------------------------------------------------
# Sparse checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(memory: ProductKeyMemory, adapter_cfg: dict, run_dir: str,
                    extra_meta: Optional[dict] = None):
    """Row-sparse checkpoint: compact values + assignment + router (~446 MB)."""
    os.makedirs(run_dir, exist_ok=True)
    assigned_mask = memory.owner_full >= 0
    assigned_idx = assigned_mask.nonzero(as_tuple=True)[0].cpu()
    owner = memory.owner_full[assigned_mask].cpu()
    values = memory.values.detach().float().cpu()
    assert torch.isfinite(values).all(), "non-finite values in checkpoint"
    assert values[memory.pad_row].abs().sum() == 0, "pad row must be zero"
    # Store compact rows in sorted-assigned_idx order so the loader's
    # load_assignment(assigned_idx, owner) (which assigns remap = arange in
    # that order) reconstructs the identical table even if the original
    # assignment was loaded unsorted.
    perm = memory.remap[assigned_idx.to(memory.remap.device)].cpu()
    values_out = torch.zeros(
        assigned_idx.numel() + 1, memory.value_dim, dtype=torch.float32
    )
    values_out[: assigned_idx.numel()] = values[perm]

    payload = {
        "values": values_out,
        "assigned_idx": assigned_idx,
        "owner": owner,
        "router": {
            "w_q": memory.w_q.detach().cpu(),
            "k1": memory.k1.detach().cpu(),
            "k2": memory.k2.detach().cpu(),
        },
        "router_seed": memory.router_seed,
        "adapter_cfg": adapter_cfg,
        "assignment_sha": assignment_sha(assigned_idx, owner),
    }
    torch.save(payload, os.path.join(run_dir, CHECKPOINT_NAME))
    if extra_meta is not None:
        save_json(extra_meta, os.path.join(run_dir, "meta.json"))


def load_memory_from_checkpoint(path: str) -> ProductKeyMemory:
    """Rebuild a ProductKeyMemory (bit-exact routing + values) from memadapt.pt."""
    if os.path.isdir(path):
        path = os.path.join(path, CHECKPOINT_NAME)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["adapter_cfg"]
    memory = ProductKeyMemory(
        hidden=cfg["hidden"],
        n_sqrt=cfg["mem_size_sqrt"],
        key_dim=cfg["key_dim"],
        topk=cfg["topk"],
        half_topk=cfg["half_topk"],
        value_dim=cfg["value_dim"],
        router_seed=payload["router_seed"],
        key_scale=cfg.get("key_scale", 1.0),
        router_tensors=payload["router"],  # saved tensors are authoritative
    )
    memory.load_assignment(payload["assigned_idx"], payload["owner"],
                           values=payload["values"])
    got = assignment_sha(payload["assigned_idx"], payload["owner"])
    assert got == payload["assignment_sha"], "assignment sha mismatch in checkpoint"
    memory._assignment_sha = got
    memory._adapter_cfg = cfg
    return memory


def apply_blocklist_file(memory: ProductKeyMemory, blocklist_path: str,
                         force_hard_zero: bool = False) -> dict:
    """Unlearn from a blocklists/<tag>.json produced by build_blocklist.py."""
    with open(blocklist_path) as f:
        spec = json.load(f)
    have = getattr(memory, "_assignment_sha", None)
    assert have is not None and spec["assignment_sha"] == have, (
        "block-list was built for a different entry assignment"
    )
    t0 = time.perf_counter()
    memory.set_blocklist(spec["entries"])
    if spec.get("hard_zero", False) or force_hard_zero:
        memory.hard_zero_blocked()
        spec["hard_zero"] = True
    spec["_apply_seconds"] = time.perf_counter() - t0
    return spec


# ---------------------------------------------------------------------------
# Eval entry point for open-unlearning (registered via MODEL_REGISTRY)
# ---------------------------------------------------------------------------

class MemAdaptLlamaForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM + memory adapter, loadable by OU's get_model.

    configs/model/MemAdapt-Llama-3.2-1B.yaml passes these through model_args:
        memadapt_checkpoint: run dir containing memadapt.pt
        blocklist:           optional blocklists/<tag>.json (None = FT row)
        hard_zero:           optional bool, zero blocked values too
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args,
                        memadapt_checkpoint: str = None,
                        blocklist: str = None,
                        hard_zero: bool = False,
                        **kwargs):
        assert memadapt_checkpoint, "memadapt_checkpoint model_arg is required"
        model = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        assert not (hard_zero and not blocklist), (
            "hard_zero requires a blocklist"
        )
        memory = load_memory_from_checkpoint(memadapt_checkpoint)
        install_adapter(model, memory, memory._adapter_cfg["layer_idx"])
        # Move to the base model's device but keep the memory path fp32.
        device = next(model.parameters()).device
        memory.to(device)
        if blocklist:
            spec = apply_blocklist_file(memory, blocklist,
                                        force_hard_zero=hard_zero)
            print(
                f"[memadapt] blocked {len(spec['entries'])} entries "
                f"(sources {spec.get('sources')}, hard_zero={spec['hard_zero']}) "
                f"in {spec['_apply_seconds']:.4f}s"
            )
        return model
