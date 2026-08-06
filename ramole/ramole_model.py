"""Stage 3 — RamoleModel: the served model (frozen base + RouterLoRA + trained router).

Drop-in for `legonet_lora/combine.py::LegoNetModel` in eval: exposes `.config`, a no-op
`set_adapter`, `forward`, and `generate`, plus the underlying `.model` (the HF model with the
spliced RouterLoraLinear layers) that the reused `metrics_for_records` loop drives directly.

Two routing entry points:
  - set_active(ids)            — a single expert set shared by the whole forward (batch-uniform;
                                 used by grouped eval, like LegoNet groups records by adapter-set).
  - set_routing(batch_idlists) — heterogeneous batch: per-sample retrieved sets are deduped into a
                                 union and an additive (-inf) mask, so each row attends only to its
                                 own experts in one fused forward (paper §4 mapping matrix M).
"""
import torch
from torch import nn

import ramole_common as rc  # noqa: F401  (ensures legonet_lora is on sys.path for combine/eval imports)
import router_lora as R


class RamoleModel(nn.Module):
    def __init__(self, model, tokenizer, controller, meta):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.controller = controller
        self.meta = meta

    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):  # drop-in no-op (routing happens via the controller)
        pass

    @classmethod
    def from_config(cls, cfg, device: str = "cpu", load_router: bool = True,
                    adapter_dir_fn=None) -> "RamoleModel":
        model, tok, controller, meta, _ = R.build_ramole_model(
            cfg, device=device, load_router_weights=load_router, adapter_dir_fn=adapter_dir_fn)
        model.eval()
        model.config.use_cache = True   # serving wants the KV cache (training set it False)
        return cls(model, tok, controller, meta)

    # ── routing ────────────────────────────────────────────────────────────────
    def set_active(self, ids):
        """Compose exactly these experts for the next forward (no mask)."""
        self.controller.set_active([int(j) for j in ids])

    def set_routing(self, batch_idlists):
        """Per-sample routing for a heterogeneous batch. batch_idlists[i] = expert ids for row i."""
        union = sorted({int(j) for ids in batch_idlists for j in ids})
        pos = {j: t for t, j in enumerate(union)}
        b, m = len(batch_idlists), len(union)
        mask = torch.full((b, m), float("-inf"))
        for i, ids in enumerate(batch_idlists):
            for j in ids:
                mask[i, pos[int(j)]] = 0.0
        self.controller.set_routed(union, mask)

    # ── HF passthrough ───────────────────────────────────────────────────────────
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
