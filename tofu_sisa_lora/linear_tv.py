"""Linearized (tangent-space) task vectors — composable_tv [lin] arm core lib (model-agnostic).

Serve the FIRST-ORDER TAYLOR EXPANSION of the base LM around theta0 instead of the finetuned
weights themselves:

    f_lin(x; tau) = f(x; theta0) + (J_theta f)(x; theta0) . tau
    tau           = sum_a w_a * scaling_a * B_a @ A_a        (target Linear weights only)

Training fits ONLY B_a against f_lin's loss (A_a is a frozen seeded projection — the IRP
convention from train_lora_shard.apply_irp_projections), so each author's knowledge lives in
an exactly-linear slot: composition is true logit-space superposition and deletion is
tau <- tau - tau_a — cheap, deterministic, O(1), with a closed-form guarantee the nonlinear
serve lacks (Ortiz-Jimenez et al. 2023, "Task Arithmetic in the Tangent Space").

Hard-won environment constraints (do not "simplify" these away):
  * torch.func.jvp COMPOSES with reverse-mode autograd through the tangent in torch 2.5.1:
    loss.backward() on f_lin reaches B through tau (verified in-env; the trainer rests on it).
  * functional_call/jvp is INCOMPATIBLE with gradient checkpointing (the known repo gotcha,
    cf. ClAMU's mask_batch_rows) — cut the batch instead of checkpointing.
  * attention must be EAGER and use_cache=False: forward-mode AD does not cover the SDPA /
    flash kernels; loaders pass attn_implementation="eager".
  * jvp tangents are deterministic WITHIN a grad mode, but no_grad vs grad-mode outputs can
    differ by ~1 ULP (kernel dispatch); eval runs entirely under no_grad and training
    entirely under grad — never assert byte-equality across the two modes.

Rejected design paths:
  * Serving via a trained-then-linearized PEFT adapter (linearize only at eval): the training
    loss would be the NONLINEAR model's — composition then has no exactness guarantee, which
    is the whole point of the arm. Train-time and serve-time function must be the same f_lin.
  * Routing/per-query adapter selection: composition is GLOBAL here (one tau for every
    query), so forward is batch-capable and needs no router — maximum-simplicity paradigm.
"""
from __future__ import annotations

import hashlib
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from jd_collection import _adapter_scaling, _read_adapter, _PREFIX

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

# Same six Linear families the SISA shards train (train_lora_shard target_modules);
# gate_proj deliberately excluded to match the frozen recipe.
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")

# factors: {param_name: (A, B, scaling)} — A frozen (rank, in), B trained leaf (out, rank).
FactorDict = Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]
TauDict = Dict[str, torch.Tensor]


# ── target selection ─────────────────────────────────────────────────────────────

def target_names(model) -> List[str]:
    """Ordered param names of the target Linear WEIGHTS: [q,k,v,o,up,down]_proj across all
    layers (96 for Llama-3.2-1B: 16 layers x 6). Derived from named_modules — never hardcode
    the layer count. Plain (non-PEFT) models only: a peft wrapper renames the Linears to
    *.base_layer, which this walk intentionally does not match."""
    names = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod_name.rsplit(".", 1)[-1] in TARGET_SUFFIXES:
            names.append(mod_name + ".weight")
    if not names:
        raise ValueError("target_names: no [q,k,v,o,up,down]_proj Linears found "
                         "(is this a plain causal-LM base model?)")
    return names


# ── deterministic seeded factors ─────────────────────────────────────────────────

def seeded_A(shape, irp_seed: int, author: int, layer_name: str,
             device=None, dtype=torch.float32) -> torch.Tensor:
    """Frozen A ~ N(0,1) from SHA-256(f"{irp_seed}:{author}:{layer_name}:default") — mirrors
    train_lora_shard.apply_irp_projections (author plays shard_id; adapter_key 'default'), so
    each (author, layer) pair is independent and iteration order is irrelevant.

    CUDA-safe idiom: draw on a CPU generator into a CPU tensor, THEN move — CPU RNG streams
    are device-stable, whereas a CUDA generator would tie the bytes to the GPU."""
    seed_bytes = hashlib.sha256(
        f"{irp_seed}:{author}:{layer_name}:default".encode()
    ).digest()
    seed_int = int.from_bytes(seed_bytes[:4], "little")
    gen = torch.Generator()
    gen.manual_seed(seed_int)
    a = torch.empty(tuple(shape), dtype=torch.float32)
    a.normal_(mean=0.0, std=1.0, generator=gen)
    return a.to(device=device, dtype=dtype)


def lora_scaling(rank: int, alpha: float, rslora: bool = True) -> float:
    """PEFT LoRA scaling: alpha/sqrt(r) under rslora, else alpha/r (== jd_collection's read)."""
    return alpha / math.sqrt(rank) if rslora else alpha / rank


def init_author_factors(model, names: List[str], *, rank: int, alpha: float, rslora: bool,
                        irp_seed: int, author: int, device=None,
                        dtype=torch.float32) -> FactorDict:
    """Fresh factors for one author: frozen seeded A + ZERO B (leaf, requires_grad) per target
    weight. B=0 makes the initial tangent 0 == serving the exact base model (PEFT's own
    B-zero-init convention)."""
    sd = dict(model.named_parameters())
    if device is None:
        device = next(model.parameters()).device
    s = lora_scaling(rank, alpha, rslora)
    factors: FactorDict = {}
    for n in names:
        out_f, in_f = sd[n].shape
        A = seeded_A((rank, in_f), irp_seed, author, n[: -len(".weight")],
                     device=device, dtype=dtype)
        B = torch.zeros(out_f, rank, device=device, dtype=dtype, requires_grad=True)
        factors[n] = (A, B, s)
    return factors


# ── tangent construction / composition ───────────────────────────────────────────

def build_tangent(factors: FactorDict, dtype=torch.float32) -> TauDict:
    """tau_n = scaling * B_n @ A_n, DIFFERENTIABLE through B (A enters as a constant — it is
    created requires_grad=False and never touched). fp32 matmul, cast at the end."""
    tau: TauDict = {}
    for n, (A, B, s) in factors.items():
        tau[n] = (s * (B.float() @ A.float())).to(dtype)
    return tau


def compose_tangents(adapter_dirs: List[str], weights: List[float],
                     dtype=torch.float32) -> TauDict:
    """Composed tangent sum_i w_i * scaling_i * B_i @ A_i per target param name.

    Deterministic: (dir, weight) pairs are SORTED by adapter dirname before the fp32
    accumulation — fp addition is non-associative, and the deletion contract
    (subtract == recompose, byte-stable across caller order) needs a canonical order.
    Subtraction = weight -1.0 entries. Reads standard PEFT dirs (jd_collection._read_adapter);
    slot names map to param names via + ".weight"."""
    assert len(adapter_dirs) == len(weights)
    pairs = sorted(
        zip(adapter_dirs, weights),
        key=lambda t: (os.path.basename(os.path.normpath(t[0])), os.path.abspath(t[0])),
    )
    tau: TauDict = {}
    ref_slots = None
    for d, w in pairs:
        slots, cfg = _read_adapter(d)
        if ref_slots is None:
            ref_slots = set(slots)
        elif set(slots) != ref_slots:
            raise ValueError(f"{d}: slot set differs from the first adapter's")
        s = _adapter_scaling(cfg)
        for slot, (A, B) in slots.items():
            name = slot + ".weight"
            delta = (float(w) * s) * (B.float() @ A.float())
            tau[name] = tau[name] + delta if name in tau else delta
    return {n: t.to(dtype) for n, t in tau.items()}


# ── the linearized forward ───────────────────────────────────────────────────────

def linearized_forward(model, names: List[str], tau: TauDict, input_ids,
                       attention_mask=None, labels=None) -> CausalLMOutput:
    """f_lin = f(theta0) + jvp: partial functional_call over ONLY `names`, with jvp of the
    function theta -> logits(x; theta) evaluated at theta0 in direction tau.

    logits = primal_logits + jvp_logits (bit-equal to the plain forward when tau == 0: the
    primal path runs the identical ops and a zero tangent propagates as exact zeros).
    Loss = shifted CE in fp32 computed HERE (ignore_index -100, mean over answer tokens —
    the HF convention) because the model's own loss path would only see the primal logits.
    Requires eager attention + use_cache=False; gradient checkpointing must be OFF
    (functional_call incompatibility — cut the batch instead). Batch-capable."""
    sd = dict(model.named_parameters())
    primals = {n: sd[n].detach() for n in names}
    tangents = {}
    for n in names:
        t = tau.get(n)
        if t is None:
            t = torch.zeros_like(sd[n])
        tangents[n] = t.to(dtype=sd[n].dtype) if t.dtype != sd[n].dtype else t

    call_kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        call_kwargs["attention_mask"] = attention_mask

    def _logits_fn(params):
        out = functional_call(model, params, args=(), kwargs=call_kwargs)
        return out.logits

    primal, tangent = jvp(_logits_fn, (primals,), (tangents,))
    logits = primal + tangent

    loss = None
    if labels is not None:
        lf = logits.float()
        shift_logits = lf[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(lf.device).long()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    return CausalLMOutput(loss=loss, logits=logits)


# ── nonlinear-debug helpers ──────────────────────────────────────────────────────

@contextmanager
def applied_tau(model, names: List[str], tau: TauDict):
    """Temporarily add tau into the target weights, restoring the EXACT original bytes on
    exit (snapshot + copy_, never add-then-subtract — fp cancellation is not exact)."""
    sd = dict(model.named_parameters())
    saved = {n: sd[n].detach().clone() for n in names}
    with torch.no_grad():
        for n in names:
            sd[n].data.add_(tau[n].to(device=sd[n].device, dtype=sd[n].dtype))
    try:
        yield model
    finally:
        with torch.no_grad():
            for n in names:
                sd[n].data.copy_(saved[n])


@torch.no_grad()
def disentanglement_error(base_model, names: List[str], tau_i: TauDict, tau_j: TauDict,
                          batch) -> float:
    """xi(i,j) = mean over batch examples of || f(t_i+t_j) - f(t_i) - f(t_j) + f(0) ||_2 in
    logit space (fp32, per-example flattened L2), with f = the NONLINEAR model served by
    DIRECT weight addition (applied_tau; exact byte-restore between conditions).

    Under the linear serve this is 0 BY CONSTRUCTION — jvp is linear in tau — so measuring it
    there is meaningless. It exists for the nonlinear-debug serve mode, where it is the
    H-lin-4 crosstalk metric: how far the real network is from its own tangent approximation
    on composed task vectors. `batch` = dict with input_ids (+ optional attention_mask)."""
    kwargs = {"input_ids": batch["input_ids"], "use_cache": False}
    if batch.get("attention_mask") is not None:
        kwargs["attention_mask"] = batch["attention_mask"]

    def _logits(tau: Optional[TauDict]):
        if tau is None:
            return base_model(**kwargs).logits.float()
        with applied_tau(base_model, names, tau):
            return base_model(**kwargs).logits.float()

    f0 = _logits(None)
    fi = _logits(tau_i)
    fj = _logits(tau_j)
    fij = _logits({n: tau_i[n] + tau_j[n] for n in names})
    per_example = (fij - fi - fj + f0).flatten(1).norm(dim=1)
    return float(per_example.mean().item())


# ── the eval seam (mirrors sift_masks_model.SiftMasksModel's contract) ───────────

class LinearTVModel(nn.Module):
    """Serve the linearized composition — drop-in for the PeftModel in eval_tofu/attack_mia.

    serve="linear"          : every forward/generate step goes through linearized_forward.
                              The composition is GLOBAL (one tau for all queries — no
                              routing), so batched forwards are safe.
    serve="nonlinear-debug" : theta0 + sum(tau) added DIRECTLY into the weights once at
                              construction (the 2x2 fallback arm, f(theta0 + tau)); forward
                              is then the plain model. NOTE: mutates the passed base model.
                              disentanglement_error is only a metric in THIS mode (it is
                              exactly 0 by construction under linear serve).
    """

    SERVE_MODES = ("linear", "nonlinear-debug")

    def __init__(self, model, tokenizer, *, names: List[str], tau: TauDict,
                 serve: str = "linear"):
        super().__init__()
        if serve not in self.SERVE_MODES:
            raise ValueError(f"serve={serve!r} not in {self.SERVE_MODES}")
        self.model = model
        self.tokenizer = tokenizer
        self.names = list(names)
        self.tau = tau
        self.serve_mode = serve
        if serve == "nonlinear-debug":
            sd = dict(model.named_parameters())
            with torch.no_grad():
                for n in self.names:
                    sd[n].data.add_(tau[n].to(device=sd[n].device, dtype=sd[n].dtype))

    # -- surface eval_tofu relies on --
    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):        # composition is global; nothing to select
        pass

    # -- forward / generate --
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("use_cache", None)   # forced off inside linearized_forward
        if self.serve_mode == "linear":
            return linearized_forward(self.model, self.names, self.tau, input_ids,
                                      attention_mask=attention_mask, labels=labels)
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels, use_cache=False, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=100,
                 do_sample=False, pad_token_id=None, eos_token_id=None, **kwargs):
        """Greedy batch-1 stepwise decode WITHOUT KV cache: jvp has no cache concept, so each
        step re-runs the full prefix through f_lin (O(T^2) — eval-scale only; budget eval
        wall time accordingly). Honors max_new_tokens + eos; returns prompt+generated ids
        (the HF generate slice convention eval_tofu relies on)."""
        assert input_ids is not None and input_ids.shape[0] == 1, \
            "LinearTVModel.generate is batch-1 (repo eval convention)"
        assert not do_sample, "greedy only"
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if eos is None:
            eos_set = set()
        elif isinstance(eos, (list, tuple)):
            eos_set = {int(e) for e in eos}
        else:
            eos_set = {int(eos)}
        ids = input_ids
        am = attention_mask if attention_mask is not None else torch.ones_like(ids)
        for _ in range(int(max_new_tokens)):
            logits = self.forward(input_ids=ids, attention_mask=am).logits
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            am = torch.cat([am, am.new_ones(1, 1)], dim=1)
            if int(nxt.item()) in eos_set:
                break
        return ids


# ── storage: standard PEFT dir + rung-4 B-only ──────────────────────────────────

def save_author_adapter(out_dir: str, factors: FactorDict, *, base_model_name: str,
                        rank: int, alpha: float, rslora: bool) -> str:
    """ALWAYS write a standard PEFT LoRA adapter dir (adapter_config.json +
    adapter_model.safetensors, base_model.model.<slot>.lora_{A,B}.weight keys) so the
    existing tooling — merge_subset._weighted_factor_cat, subspace_overlap, eval_tofu
    --preloaded_adapter — works on lin-arm shards unchanged."""
    from peft import LoraConfig
    from safetensors.torch import save_file

    tensors = {}
    for name, (A, B, _s) in factors.items():
        slot = name[: -len(".weight")]
        tensors[_PREFIX + slot + ".lora_A.weight"] = A.detach().cpu().contiguous()
        tensors[_PREFIX + slot + ".lora_B.weight"] = B.detach().cpu().contiguous()
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=list(TARGET_SUFFIXES),
        bias="none", task_type="CAUSAL_LM", use_rslora=rslora,
    )
    cfg.base_model_name_or_path = base_model_name
    os.makedirs(out_dir, exist_ok=True)
    cfg.save_pretrained(out_dir)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    return out_dir


B_ONLY_FILE = "b_only.pt"
_B_META_REQUIRED = ("author", "irp_seed", "rank", "alpha", "rslora", "in_features")


def save_b_only(out_dir: str, B_dict: Dict[str, torch.Tensor], meta: dict) -> str:
    """Rung-4 lean storage: ONLY the trained B matrices + the meta needed to re-derive A from
    its seed at load time (~half the bytes of the factor pair). meta must carry
    author/irp_seed/rank/alpha/rslora/in_features ({param_name: A.shape[1]}) — asserted here
    so a load can never silently mis-derive A."""
    missing = [k for k in _B_META_REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"save_b_only: meta missing {missing}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, B_ONLY_FILE)
    torch.save({"B": {n: B.detach().cpu() for n, B in B_dict.items()}, "meta": dict(meta)},
               path)
    return path


def load_author_factors(adapter_dir: str) -> Tuple[FactorDict, dict]:
    """Rung-4 load: read b_only.pt, re-derive each A from the recorded (irp_seed, author,
    module) seed — byte-identical to the training-time A — and return (factors, meta)."""
    blob = torch.load(os.path.join(adapter_dir, B_ONLY_FILE), map_location="cpu",
                      weights_only=True)   # tensors + plain-dict meta only
    meta = blob["meta"]
    s = lora_scaling(meta["rank"], meta["alpha"], meta["rslora"])
    factors: FactorDict = {}
    for name, B in blob["B"].items():
        A = seeded_A((meta["rank"], meta["in_features"][name]),
                     meta["irp_seed"], meta["author"], name[: -len(".weight")])
        factors[name] = (A, B.float(), s)
    return factors, meta


# ── eval_tofu loader ─────────────────────────────────────────────────────────────

def _id_list(v) -> List[int]:
    if v is None:
        return []
    if isinstance(v, str):
        return [int(x) for x in v.split(",") if x.strip()]
    return [int(x) for x in v]


def resolve_compose(cfg, *, authors=None, n=None, subtract=None):
    """Pure resolution of the --linear_tv_* selection into (adapter_dirs, weights).

    authors : comma string / list of author ids to compose.
    n       : compose the FIRST N of the config pool — merge_subset.subset_authors(
              cfg[pool_seed], N) — mutually exclusive with `authors`; capped at pool_size
              (only pool authors are trained). Pools derive at runtime, never hardcoded.
    subtract: ids composed at NEGATIVE weight (tangent subtraction = the O(1) unlearn op).

    Compose weights: +-1 sums by default; cfg["compose"]="mean" uses +-1/N_pos for EVERY
    entry (a FIXED lambda keeps drop-a-term exact — the merged_additive _s{lambda} lesson).
    Author dirs = {cfg[out_dir]}/shard_{a}. Split out from the loader so the CPU gate can
    cover the selection logic without downloading the base model.
    """
    from merge_subset import subset_authors

    pos = _id_list(authors)
    if n is not None:
        if pos:
            raise ValueError("pass either authors or n, not both")
        pool_size = int(cfg.get("pool_size", n))
        if not (1 <= int(n) <= pool_size):
            raise ValueError(f"n={n} outside [1, pool_size={pool_size}]")
        pos = subset_authors(int(cfg.get("pool_seed", 42)), int(n))
    if not pos:
        raise ValueError("linear_tv: no authors selected (pass authors or n)")
    neg = _id_list(subtract)

    mode = cfg.get("compose", "sum")
    if mode == "sum":
        lam = 1.0
    elif mode == "mean":
        lam = 1.0 / len(pos)
    else:
        raise ValueError(f"unknown compose mode {cfg.get('compose')!r}")
    dirs = [os.path.join(cfg["out_dir"], f"shard_{a}") for a in pos + neg]
    weights = [lam] * len(pos) + [-lam] * len(neg)
    return dirs, weights


def load_linear_tv_eval_model(cfg, *, authors=None, n=None, subtract=None,
                              serve="linear", device=None):
    """Build (LinearTVModel, tokenizer) for eval_tofu's --linear_tv_* flags.

    Selection semantics live in resolve_compose (see its docstring); serve = "linear"
    (the method) | "nonlinear-debug" (theta0+tau baked into the weights, the 2x2 fallback).
    Missing adapter dirs hard-fail (no silent skips — the load_prefix_concat_model rule).
    """
    dirs, weights = resolve_compose(cfg, authors=authors, n=n, subtract=subtract)
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"linear_tv adapter dir missing: {d}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if "hf_home" in cfg:
        os.environ["HF_HOME"] = cfg["hf_home"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.float32 if cfg.get("fp32", True) else torch.bfloat16
    # eager + no cache: fwAD does not cover SDPA kernels; jvp has no KV-cache concept.
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, attn_implementation="eager",
        trust_remote_code=True).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    base.config.use_cache = False

    names = target_names(base)
    tau = compose_tangents(dirs, weights)
    if set(tau) != set(names):
        raise ValueError(
            f"composed tangent covers {len(tau)} params but the model has {len(names)} "
            f"targets — adapter pool does not match this base model")
    tau = {k: v.to(device) for k, v in tau.items()}
    wrapper = LinearTVModel(base, tok, names=names, tau=tau, serve=serve)
    return wrapper, tok
