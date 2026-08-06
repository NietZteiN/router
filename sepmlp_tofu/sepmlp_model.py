"""SepMLP model integration: wrapper, install/freeze, checkpoint I/O,
droplist deletion, and the open-unlearning eval entry class.

Mirrors memadapt_tofu/memadapt_model.py. Imports torch + transformers + stdlib
only so it loads in BOTH environments (test-env for training/relearn,
`unlearning` for OU eval).
"""

import json
import os
import time
from typing import Dict, List

import torch
from torch import nn

from bank_layer import AuthorBank, BankState
from sepmlp_common import bank_sha, file_sha256, save_json, slurm_job_id


class SepMlpMLP(nn.Module):
    """Wraps a decoder layer's MLP: out = mlp(x) + bank(x, state).

    Batch state is carried by the shared BankState object (set via trainer /
    eval methods), never forward kwargs: HF's decoder layer calls mlp(x)
    positionally and would silently drop extra kwargs.
    """

    def __init__(self, mlp: nn.Module, bank: AuthorBank, state: BankState):
        super().__init__()
        self.mlp = mlp
        self.bank = bank
        self.state = state

    def forward(self, x):
        return self.mlp(x) + self.bank(x, self.state)


def install_banks(model, banks: Dict[int, AuthorBank], state: BankState) -> List[SepMlpMLP]:
    """Splice every configured decoder layer's mlp with a SepMlpMLP wrapper."""
    wrappers = []
    for layer_idx, bank in sorted(banks.items()):
        layer = model.model.layers[layer_idx]
        assert not isinstance(layer.mlp, SepMlpMLP), (
            f"bank already installed on layer {layer_idx}"
        )
        wrapper = SepMlpMLP(layer.mlp, bank, state)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    assert len(wrappers) == len(banks)
    return wrappers


def freeze_base(model, banks: Dict[int, AuthorBank]) -> List[str]:
    """Freeze everything except the bank tensors; assert the exact set."""
    for p in model.parameters():
        p.requires_grad_(False)
    for bank in banks.values():
        bank.W_gate.requires_grad_(True)
        bank.W_up.requires_grad_(True)
        bank.b_gate.requires_grad_(True)
        bank.W_down.requires_grad_(True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    expected = 4 * len(banks)
    assert len(trainable) == expected and all(
        n.rsplit(".", 1)[-1] in ("W_gate", "W_up", "b_gate", "W_down")
        for n in trainable
    ), trainable
    return trainable


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def compute_bank_sha(banks: Dict[int, AuthorBank]) -> str:
    # Covers EVERY bank tensor's shape (gate/up/bias/down) so adding or
    # resizing a tensor can never silently pair with a stale droplist.
    any_bank = next(iter(banks.values()))
    shapes = []
    for l, b in sorted(banks.items()):
        shapes += [tuple(b.W_gate.shape), tuple(b.W_up.shape),
                   tuple(b.b_gate.shape), tuple(b.W_down.shape)]
    return bank_sha(any_bank.author_ids, sorted(banks.keys()), shapes)


def save_checkpoint(banks: Dict[int, AuthorBank], adapter_cfg: dict,
                    run_dir: str, extra_meta: dict = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    any_bank = next(iter(banks.values()))
    payload = {
        "banks": {
            int(l): {
                "W_gate": b.W_gate.detach().cpu().float(),
                "W_up": b.W_up.detach().cpu().float(),
                "b_gate": b.b_gate.detach().cpu().float(),
                "W_down": b.W_down.detach().cpu().float(),
            }
            for l, b in sorted(banks.items())
        },
        "author_ids": any_bank.author_ids.detach().cpu(),
        "layers": sorted(int(l) for l in banks.keys()),
        "adapter_cfg": adapter_cfg,
        "bank_sha": compute_bank_sha(banks),
    }
    path = os.path.join(run_dir, "sepmlp.pt")
    torch.save(payload, path)
    meta = dict(extra_meta or {})
    meta["bank_sha"] = payload["bank_sha"]
    meta["checkpoint_sha256"] = file_sha256(path)
    save_json(meta, os.path.join(run_dir, "meta.json"))
    return path


def load_banks_from_checkpoint(run_dir: str):
    """Returns (banks dict layer->AuthorBank, adapter_cfg, state). The banks
    are reconstructed then overwritten with the stored tensors, so the seeded
    init only defines shapes; loaded values are authoritative."""
    path = run_dir if run_dir.endswith(".pt") else os.path.join(run_dir, "sepmlp.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["adapter_cfg"]
    author_ids = payload["author_ids"]
    state = BankState()
    banks = {}
    for l in payload["layers"]:
        bank = AuthorBank(
            hidden=cfg["hidden"], width=cfg["width"], author_ids=author_ids,
            layer_idx=int(l), init_seed=cfg["init_seed"],
            init_std=cfg.get("init_std"),
            penalty_form=cfg.get("penalty_form", "output_gram"),
            gate_act=cfg.get("gate_act", "relu"),
        )
        stored = payload["banks"][int(l)]
        with torch.no_grad():
            bank.W_gate.copy_(stored["W_gate"])
            bank.W_up.copy_(stored["W_up"])
            if "b_gate" in stored:  # extensible: pre-bias payloads load as 0
                bank.b_gate.copy_(stored["b_gate"])
            bank.W_down.copy_(stored["W_down"])
        banks[int(l)] = bank
    expected_sha = compute_bank_sha(banks)
    assert payload["bank_sha"] == expected_sha, (
        "bank_sha mismatch: checkpoint does not match its author/layer map"
    )
    return banks, cfg, state


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def apply_droplist_file(banks: Dict[int, AuthorBank], path: str,
                        mode: str = "remove") -> dict:
    """The O(1) unlearning op. Asserts bank_sha provenance, then deletes the
    listed authors from every layer by one of two value-identical mechanisms
    (remove == zero_wdown == active-mask == bake, pinned by test_deletion.py):
      - "remove"     (default): physically index-select survivors — shrinks the
                     grouped matrices, frees the dropped slices' parameters.
      - "zero_wdown" (paper-exact, MUSR §3.2): zero the dropped authors' W_down
                     columns in place at fixed shape — a static weight edit
                     readable back from the stored parameters.
    Both drive the dropped adapter's contribution to exactly 0; pick "remove"
    to reclaim storage, "zero_wdown" for the paper's stored-weight-verifiable
    framing."""
    assert mode in ("remove", "zero_wdown"), f"unknown deletion mode {mode!r}"
    with open(path) as f:
        spec = json.load(f)
    sha = compute_bank_sha(banks)
    assert spec["bank_sha"] == sha, (
        f"droplist bank_sha {spec['bank_sha'][:12]} != checkpoint {sha[:12]} — "
        "wrong droplist for this checkpoint"
    )
    t0 = time.perf_counter()
    dropped = None
    for bank in banks.values():
        if mode == "remove":
            n = bank.remove_authors(spec["authors"])
        else:
            n = bank.zero_wdown_authors(spec["authors"])
        assert dropped is None or n == dropped, "layers disagree on drop count"
        dropped = n
    spec["_apply_seconds"] = time.perf_counter() - t0
    spec["_dropped_per_layer"] = dropped
    spec["_deletion_mode"] = mode
    return spec


# ---------------------------------------------------------------------------
# OU eval entry
# ---------------------------------------------------------------------------

try:  # transformers is present in both envs; guard only for torch-only tools
    from transformers import LlamaForCausalLM

    class SepMlpLlamaForCausalLM(LlamaForCausalLM):
        """LlamaForCausalLM + per-author MLP banks, loadable by OU's get_model.

        configs/model/SepMlp-Llama-3.2-1B.yaml passes these through model_args:
            sepmlp_checkpoint: run dir containing sepmlp.pt
            droplist:          optional droplists/<tag>.json (None = FT row)
        """

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args,
                            sepmlp_checkpoint: str = None,
                            droplist: str = None,
                            **kwargs):
            assert sepmlp_checkpoint, "sepmlp_checkpoint model_arg is required"
            model = super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )
            banks, cfg, state = load_banks_from_checkpoint(sepmlp_checkpoint)
            if droplist:
                spec = apply_droplist_file(banks, droplist)
                print(
                    f"[sepmlp] dropped {len(spec['authors'])} authors "
                    f"({spec['_dropped_per_layer']} slices/layer) "
                    f"in {spec['_apply_seconds']:.4f}s"
                )
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            for bank in banks.values():
                bank.to(device=device, dtype=dtype)
            install_banks(model, banks, state)
            model._sepmlp_banks = banks
            model._sepmlp_state = state
            return model

except ImportError:  # pragma: no cover
    SepMlpLlamaForCausalLM = None


def build_meta(cfg: dict, extra: dict = None) -> dict:
    meta = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_config_path"),
        "script_sha256": None,  # caller fills with its own file
        "slurm_job_id": slurm_job_id(),
        "seed": cfg.get("seed"),
        "torch_version": torch.__version__,
    }
    meta.update(extra or {})
    return meta
