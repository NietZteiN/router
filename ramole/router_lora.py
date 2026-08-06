"""Stage 2 — the RouterLoRA cross-attention that composes multiple live LoRA experts.

This is RAMoLE's core novelty and the one thing the existing LegoNet code cannot do.
`legonet_lora/combine.py` composes the k retrieved experts with a uniform 1/k delta-average
(`add_weighted_adapter(linear)`), which fuses them into ONE delta and destroys the per-expert
structure. RAMoLE instead keeps every active expert's output `v_i = scaling·B_i A_i x`
separate and weights them with a learned, per-layer cross-attention gate computed from the
hidden state (paper Eqs. 6–11):

    q   = A_r x                       # router query   (A_r: r×d_in)
    k_i = B_r^T v_i                   # router key      (B_r: d_out×r)
    s_i = <q, k_i> / sqrt(r)          # score per expert
    alpha = softmax_i(s_i)            # attention over experts
    x'  = sum_i alpha_i v_i           # weighted expert output, replaces the summed LoRA term

Only the per-layer router params gamma = {A_r, B_r} are trainable; the base model and all
expert LoRAs are frozen. Because routing is computed from hidden states (not a fixed expert
table), the router generalizes zero-shot to experts it never saw in training.

We manage LoRA weights manually here (NOT through PEFT): PEFT's `set_adapter`/
`add_weighted_adapter`/`adapter_names` paths each either select or merge a *single* delta per
token and cannot express the attention sum over multiple experts.

Design notes grounded in the on-disk adapters (`legonet_l32_3b_n32_k3`):
  - 28 layers × {q,k,v,o}_proj = 112 target linears; GQA ⇒ q/o have d_out=3072, k/v d_out=1024,
    so A_r/B_r are sized PER LAYER (a single global router pair is shape-incompatible).
  - lora_A.weight is (r, d_in), lora_B.weight is (d_out, r); scaling = lora_alpha/r = 2.0
    (use_rslora=False — NOT the sqrt(r) convention used by the tofu rslora shards).
  - Router math runs in fp32 for a stable softmax even when base+experts are bf16.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager

import torch
from torch import nn

from ramole_common import Paths, source_paths


# ── Expert weight extraction ───────────────────────────────────────────────────

def extract_expert_weights(cfg: dict, dtype: torch.dtype = torch.float32, adapter_dir_fn=None):
    """Load every expert's per-layer (A, B) from its PEFT safetensors, grouped by the
    target-linear module path. Returns (experts, meta):

        experts[path] = {"A": (n, r, d_in), "B": (n, d_out, r)}  (CPU tensors, `dtype`)
        meta = {n, rank, alpha, scaling, disabled: [bool]*n, paths: [...]}

    `adapter_dir_fn(j) -> path` overrides where expert j is read from (default: the source run's
    `adapter_dir`); used to assemble a POST-DELETION pool from mixed original/retrained dirs (the
    unlearning demo). Asserts every expert contributes every path with matching rank/dims — raises
    rather than silently skipping (the recurring tofu footgun: a partial expert set corrupts the pool).
    """
    from safetensors import safe_open

    # source_paths(cfg) only resolves for the DBpedia layout; skip it when an explicit
    # adapter_dir_fn is given (e.g. the TOFU pool), so a TOFU cfg need not carry source_run.
    adapter_dir_fn = adapter_dir_fn or source_paths(cfg).adapter_dir
    n = cfg["n"]
    per_path: dict[str, dict[str, list]] = {}
    disabled = [False] * n
    rank = alpha = None

    for j in range(n):
        adir = adapter_dir_fn(j)
        acfg_path = os.path.join(adir, "adapter_config.json")
        st_path = os.path.join(adir, "adapter_model.safetensors")
        if not os.path.isfile(acfg_path) or not os.path.isfile(st_path):
            raise RuntimeError(f"expert {j}: missing adapter at {adir}")
        with open(acfg_path) as f:
            acfg = json.load(f)
        r_j, a_j = int(acfg["r"]), float(acfg["lora_alpha"])
        if rank is None:
            rank, alpha = r_j, a_j
        elif (r_j, a_j) != (rank, alpha):
            raise RuntimeError(
                f"expert {j} rank/alpha ({r_j},{a_j}) != pool ({rank},{alpha}); "
                "RAMoLE assumes a homogeneous expert pool")
        meta_path = os.path.join(adir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                disabled[j] = bool(json.load(f).get("disabled", False))

        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                mp = key.replace("base_model.model.", "").rsplit(".lora_", 1)[0]
                if key.endswith("lora_A.weight"):
                    which = "A"
                elif key.endswith("lora_B.weight"):
                    which = "B"
                else:
                    raise RuntimeError(f"expert {j}: unexpected adapter key {key}")
                slot = per_path.setdefault(mp, {"A": [None] * n, "B": [None] * n})
                slot[which][j] = f.get_tensor(key)

    experts = {}
    for mp, slot in per_path.items():
        for j in range(n):
            if slot["A"][j] is None or slot["B"][j] is None:
                raise RuntimeError(f"expert {j}: missing weights for {mp}")
        A = torch.stack(slot["A"], 0).to(dtype)   # (n, r, d_in)
        B = torch.stack(slot["B"], 0).to(dtype)   # (n, d_out, r)
        assert A.shape[1] == rank and B.shape[2] == rank, f"{mp}: rank mismatch"
        experts[mp] = {"A": A, "B": B}

    meta = {
        "n": n, "rank": rank, "alpha": alpha, "scaling": alpha / rank,
        "disabled": disabled, "num_disabled": int(sum(disabled)),
        "paths": sorted(experts), "num_paths": len(experts),
    }
    return experts, meta


# ── Active-expert controller ───────────────────────────────────────────────────

class RouterController:
    """Shared, mutable handle every RouterLoraLinear reads at forward time to learn which
    experts are live (analog of PEFT's `set_adapter`). Plain object — never an nn.Module, so
    it stays out of every state_dict.

      active_idx : LongTensor (m,) — indices into [0, n_pool) of the experts to compose
      logit_mask : (b, m) float 0/-inf added to scores before softmax (per-sample inference
                   routing); None during training (batch-uniform active set)
      train_pool : the full set of training-cluster experts, for dropout sampling
    """

    def __init__(self, n_pool: int):
        self.n_pool = n_pool
        self.active_idx = torch.arange(n_pool, dtype=torch.long)
        self.logit_mask = None
        self.train_pool = list(range(n_pool))
        # E2 diagnostics: opt-in capture of the per-layer attention weights alpha.
        # Default off => zero effect on training/serving. Callers must clear `captured`
        # between records (append-only) and only capture teacher-forced forwards (never
        # generate: KV-cache decode steps have l=1 and break position pooling).
        self.capture_alpha = False
        self.captured: dict[str, list] = {}

    def set_active(self, idx):
        self.active_idx = torch.as_tensor(list(idx), dtype=torch.long)
        self.logit_mask = None

    def set_pool(self, idx):
        """Define the training pool (e.g. the 40% training clusters) and activate all of it."""
        self.train_pool = [int(i) for i in idx]
        self.set_active(self.train_pool)

    def set_routed(self, union_idx, logit_mask: torch.Tensor):
        """Per-sample inference routing: union of the batch's retrieved experts + (b,m) mask."""
        self.active_idx = torch.as_tensor(list(union_idx), dtype=torch.long)
        self.logit_mask = logit_mask

    def sample_dropout(self, p: float, generator: torch.Generator):
        """Random LoRA Dropout (paper Eq.13): keep each training-pool expert w.p. (1-p), force
        >= min(2, |pool|) survivors to avoid an all-masked softmax. Uses `generator` so
        set_determinism makes it reproducible."""
        pool = self.train_pool
        m = len(pool)
        floor = min(2, m)
        while True:
            keep = torch.rand(m, generator=generator) >= p
            if int(keep.sum()) >= floor:
                break
        self.set_active([pool[i] for i in range(m) if bool(keep[i])])

    @contextmanager
    def temporarily(self, active=None, logit_mask=None):
        prev = (self.active_idx, self.logit_mask)
        if active is not None:
            self.active_idx = torch.as_tensor(list(active), dtype=torch.long)
        self.logit_mask = logit_mask
        try:
            yield
        finally:
            self.active_idx, self.logit_mask = prev


# ── The per-layer cross-attention module ───────────────────────────────────────

class RouterLoraLinear(nn.Module):
    """Replaces one target nn.Linear. Wraps the frozen base linear, holds all experts'
    (A_i, B_i) for this layer as frozen buffers, and the trainable router (A_r, B_r)."""

    def __init__(self, base: nn.Linear, expert_A: torch.Tensor, expert_B: torch.Tensor,
                 scaling: float, router_rank: int, controller: RouterController):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        # frozen, non-persistent → excluded from the router checkpoint
        self.register_buffer("expert_A", expert_A.to(dev), persistent=False)  # (n, r, d_in)
        self.register_buffer("expert_B", expert_B.to(dev), persistent=False)  # (n, d_out, r)
        n, r, d_in = expert_A.shape
        d_out = expert_B.shape[1]
        self.n_pool = n
        self.scaling = float(scaling)
        self.router_rank = int(router_rank)
        self._sqrt_r = math.sqrt(self.router_rank)
        self.controller = controller
        self._path_name = "?"   # module path; stamped by install_router (alpha-capture key)
        # Router params are ALWAYS fp32 (stable softmax/optimizer moments under bf16 base).
        self.A_r = nn.Parameter(torch.empty(self.router_rank, d_in, dtype=torch.float32, device=dev))
        self.B_r = nn.Parameter(torch.empty(d_out, self.router_rank, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.A_r, a=math.sqrt(5))
        # small B_r ⇒ scores start ≈0 ⇒ near-uniform attention (≈ the 1/k baseline) while still
        # giving A_r a nonzero gradient on step 1 (unlike a zero init).
        nn.init.normal_(self.B_r, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        ctrl = self.controller
        active = ctrl.active_idx.to(self.expert_A.device)
        if active.numel() == 0:
            return base_out
        assert x.dim() == 3, f"RouterLoraLinear expects (b,l,d) hidden states, got {tuple(x.shape)}"

        xf = x.float()                                                  # (b, l, d_in)
        A = self.expert_A.index_select(0, active).float()              # (m, r, d_in)
        B = self.expert_B.index_select(0, active).float()              # (m, d_out, r)
        Ax = torch.einsum("mrd,bld->mblr", A, xf)                      # (m, b, l, r)
        v = self.scaling * torch.einsum("mor,mblr->mblo", B, Ax)       # (m, b, l, d_out)
        q = torch.einsum("rd,bld->blr", self.A_r, xf)                  # (b, l, r)
        kk = torch.einsum("or,mblo->mblr", self.B_r, v)                # (m, b, l, r)
        s = torch.einsum("blr,mblr->mbl", q, kk) / self._sqrt_r        # (m, b, l)

        mask = ctrl.logit_mask
        if mask is not None:
            mask = mask.to(s.device)
            assert mask.shape == (s.shape[1], s.shape[0]), (
                f"logit_mask {tuple(mask.shape)} != (b={s.shape[1]}, m={s.shape[0]})")
            s = s + mask.transpose(0, 1).unsqueeze(-1)                 # (b,m)->(m,b,1)

        alpha = torch.softmax(s, dim=0)                                # over experts
        if ctrl.capture_alpha:  # E2 diagnostics (opt-in; see RouterController)
            ctrl.captured.setdefault(self._path_name, []).append(
                (active.detach().cpu(), alpha.detach().float().cpu()))
        xprime = torch.einsum("mbl,mblo->blo", alpha, v)               # (b, l, d_out)
        return base_out + xprime.to(base_out.dtype)


# ── Install / freeze / save / load ─────────────────────────────────────────────

def install_router(model, experts: dict, scaling: float, router_rank: int,
                   controller: RouterController) -> list[str]:
    """Splice a RouterLoraLinear over every target linear named in `experts`. Returns the
    installed module paths (sorted)."""
    installed = []
    for mp in sorted(experts):
        parent_path, child = mp.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        base_linear = getattr(parent, child)
        if not isinstance(base_linear, nn.Linear):
            raise RuntimeError(f"{mp}: target is {type(base_linear).__name__}, not nn.Linear")
        d = experts[mp]
        assert d["A"].shape[2] == base_linear.in_features, f"{mp}: d_in mismatch"
        assert d["B"].shape[1] == base_linear.out_features, f"{mp}: d_out mismatch"
        rl = RouterLoraLinear(base_linear, d["A"], d["B"], scaling, router_rank, controller)
        rl._path_name = mp
        setattr(parent, child, rl)
        installed.append(mp)
    return installed


def freeze_to_router(model) -> list[str]:
    """Freeze everything, then re-enable only the router params (A_r, B_r). Returns their
    parameter names (expected: 2 per installed layer)."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for name, p in model.named_parameters():
        if name.endswith(".A_r") or name.endswith(".B_r"):
            p.requires_grad_(True)
            names.append(name)
    return names


def router_parameters(model):
    return [p for _, p in model.named_parameters() if p.requires_grad]


def router_state_dict(model) -> dict:
    return {name: p.detach().float().cpu()
            for name, p in model.named_parameters() if p.requires_grad}


def save_router(model, path: str):
    """Save router-only params (A_r, B_r per layer) to a safetensors file. The caller writes
    the companion meta JSON (Paths.router_meta)."""
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(router_state_dict(model), path)


def load_router(model, path: str):
    """Load router-only weights; assert the key set exactly matches the model's router params."""
    from safetensors.torch import load_file
    sd = load_file(path)
    own = dict(model.named_parameters())
    router_keys = {name for name, p in own.items() if p.requires_grad}
    if set(sd) != router_keys:
        raise RuntimeError(
            f"router checkpoint keys != model router params "
            f"(missing {sorted(router_keys - set(sd))[:3]}, extra {sorted(set(sd) - router_keys)[:3]})")
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].dtype).to(own[k].device))


# ── Build the full RAMoLE model ────────────────────────────────────────────────

def build_ramole_model(cfg: dict, device: str = "cpu", dtype=None,
                       load_router_weights: bool = True, adapter_dir_fn=None):
    """Frozen base + spliced RouterLoraLinear over all targets + (optionally) trained router
    weights. `adapter_dir_fn` (see extract_expert_weights) serves a post-deletion expert pool.
    Returns (model, tokenizer, controller, meta, installed_paths)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ramole_common import set_determinism
    set_determinism(cfg["base_seed"])

    use_cuda = device != "cpu" and torch.cuda.is_available()
    if dtype is None:
        dtype = torch.bfloat16 if use_cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=dtype,
        device_map="auto" if use_cuda else None, trust_remote_code=True)
    if not use_cuda:
        model = model.to(device)
    model.config.use_cache = False

    experts, meta = extract_expert_weights(cfg, dtype=dtype, adapter_dir_fn=adapter_dir_fn)
    controller = RouterController(meta["n"])
    installed = install_router(model, experts, meta["scaling"], cfg["router"]["rank"], controller)
    meta["router_params"] = freeze_to_router(model)
    meta["installed"] = installed

    # Paths(cfg) only resolves for the DBpedia layout (needs name/root); for TOFU we pass
    # load_router_weights=False and load the router explicitly via load_router().
    if load_router_weights:
        rpath = Paths(cfg).router_path
        if os.path.isfile(rpath):
            load_router(model, rpath)
            meta["router_loaded_from"] = rpath
    return model, tok, controller, meta, installed
