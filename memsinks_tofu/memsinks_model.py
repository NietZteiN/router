"""Model plumbing for MemSinks/SeqTD-on-TOFU: mask state, LoRA-delta hooks,
and fixed per-neuron scale vectors for deletion/serving.

Substrate = masked LoRA delta ("lora_delta"): forward hooks on
`...mlp.{gate_proj,up_proj}.lora_B.default` gate each MLP intermediate
neuron's LoRA contribution per author. peft 0.14.0's non-DoRA forward is
`result = base(x) + lora_B(lora_A(dropout(x))) * scaling`
(peft/tuners/lora/layer.py:596-640), so a hook on the `lora_B` Linear sees
exactly the pre-scaling delta; the scalar `scaling` commutes with the mask.
The base path is never touched: a masked sink neuron behaves exactly like the
pretrained neuron, not like a dead one.

For any FIXED per-neuron scale vector v (deletion/serving conditions),
hook-masking is identical to row-scaling lora_B's weight:
  v ⊙ (W_B (A x)) == (diag(v) W_B)(A x)
which is what bake_deletion.py materializes — every eval condition is a
bone-stock PEFT adapter dir (eval_tofu.py --preloaded_adapter, no wrapper).
The bake≡hook identity is unit-tested in test_memsinks.py.
"""
import json
import os

import torch


class MaskState:
    """Shared mutable state between the trainer and the sink hooks.

    mask_table: (num_authors, num_mem) bool — frozen, built once on CPU
    (masks.build_partition_and_table), moved to device once at train start.
    `current` is the per-batch mask (B, 1, intermediate_size) or None; author
    IDs are constant per example, so the mask is constant over the sequence
    dim and broadcasts over both full forwards (B,T,I) and KV-cached
    single-token steps (B,1,I).
    """

    def __init__(self, mask_table: torch.Tensor, num_gen: int):
        assert mask_table.dtype == torch.bool
        self.mask_table = mask_table
        self.num_gen = num_gen
        self.num_mem = mask_table.shape[1]
        self.intermediate_size = num_gen + self.num_mem
        self.current = None
        self.seen_authors = set()   # distinct-ID guard (silent all-masks-identical failure)

    def to(self, device):
        self.mask_table = self.mask_table.to(device)
        return self

    def set_batch(self, author_ids: torch.Tensor):
        """author_ids: (B,) long tensor of author INDICES 0..num_authors-1."""
        self.seen_authors.update(int(a) for a in author_ids.tolist())
        sink = self.mask_table[author_ids]                                  # (B, num_mem)
        gen = torch.ones(sink.shape[0], self.num_gen, dtype=torch.bool, device=sink.device)
        self.current = torch.cat([gen, sink], dim=-1).unsqueeze(1)          # (B, 1, I)

    def set_fixed(self, vector: torch.Tensor):
        """Serve/probe with one fixed per-neuron vector (bool or float, (I,))."""
        self.current = vector.reshape(1, 1, -1).to(self.mask_table.device)

    def clear(self):
        self.current = None


def _sink_hook(state: MaskState):
    def hook(module, inputs, output):
        if state.current is not None:
            return output * state.current.to(dtype=output.dtype, device=output.device)
        if module.training:
            raise RuntimeError(
                "MemSinks: sink-hooked module ran a training forward with no mask set "
                "(author_ids never reached MaskState.set_batch — the silent-failure guard)")
        return output
    return hook


def install_sink_hooks(peft_model, state: MaskState, sink_modules, n_layers):
    """Hook every `...mlp.<m>.lora_B.default` for m in sink_modules.

    Asserts the exact expected hook count — a partial match (e.g. gate_proj
    missing from target_modules) must fail loudly, not train half-masked.
    """
    suffixes = tuple(f".mlp.{m}.lora_B.default" for m in sink_modules)
    handles = []
    for name, mod in peft_model.named_modules():
        if name.endswith(suffixes):
            handles.append(mod.register_forward_hook(_sink_hook(state)))
    expected = n_layers * len(sink_modules)
    assert len(handles) == expected, (
        f"hooked {len(handles)} modules, expected {expected} "
        f"(layers={n_layers} x sink_modules={sink_modules})")
    return handles


def build_scale_vector(mask_table, num_gen, mode, forget_authors=None, sink_scale=1.0):
    """Fixed per-neuron scale vector (float32, (I,)) for a serving condition.

    full        : all ones (primary full-model serving; sinks at 1.0)
    delete      : 0 on the union of forget_authors' sink masks, 1 elsewhere
    dropall     : 0 on the whole sink pool (paper-"dropout" analogue; no gen
                  rescale — gen deltas trained at full strength every step)
    full_scaled : sinks at sink_scale (paper-"all" analogue; NOT the primary
                  serving mode for this fine-tuning port — see plan §5)
    """
    intermediate = num_gen + mask_table.shape[1]
    v = torch.ones(intermediate, dtype=torch.float32)
    if mode == "full":
        pass
    elif mode == "delete":
        assert forget_authors, "delete mode needs forget_authors"
        union = mask_table[list(forget_authors)].any(dim=0)
        v[num_gen:][union] = 0.0
    elif mode == "dropall":
        v[num_gen:] = 0.0
    elif mode == "full_scaled":
        v[num_gen:] = sink_scale
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return v


def author_serve_vector(mask_table, num_gen, author: int):
    """Author a's own TRAINING-condition vector (gen=1, own sinks=1, others=0).
    Used by the H4 memorization-gap probe, not by any headline serving mode."""
    intermediate = num_gen + mask_table.shape[1]
    v = torch.zeros(intermediate, dtype=torch.float32)
    v[:num_gen] = 1.0
    v[num_gen:][mask_table[author]] = 1.0
    return v


def author_delete_vector(mask_table, num_gen, author: int):
    """All deltas on EXCEPT author a's sinks (the single-author deletion)."""
    intermediate = num_gen + mask_table.shape[1]
    v = torch.ones(intermediate, dtype=torch.float32)
    v[num_gen:][mask_table[author]] = 0.0
    return v


def save_masks(run_dir, mask_table, num_gen, cfg, extra=None):
    """Persist the frozen mask table + partition next to the trained adapter."""
    import masks as M
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"mask_table": mask_table.cpu(), "num_gen": num_gen,
                "id_scheme": cfg["id_scheme"], "p_gen": cfg["p_gen"],
                "p_mem": cfg.get("p_mem")}, os.path.join(run_dir, "sink_masks.pt"))
    meta = {
        "mask_sha256": M.table_sha256(mask_table.cpu()),
        "num_gen": num_gen,
        "num_mem": int(mask_table.shape[1]),
        "num_authors": int(mask_table.shape[0]),
        "config": cfg,
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(run_dir, "memsinks_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_masks(run_dir):
    d = torch.load(os.path.join(run_dir, "sink_masks.pt"), weights_only=True)
    return d["mask_table"], int(d["num_gen"])
