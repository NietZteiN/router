"""BlockTc model integration: wrappers + surgery, install/freeze, detector
init, checkpoint I/O, droplist deletion, and the open-unlearning eval entry.

Mirrors sepmlp_tofu/sepmlp_model.py (and hosts the detector-init pre-pass,
which sepmlp kept in its training script — here it sits next to the surgery
because install_tc/raw-mlp ordering is what it is coupled to). Imports torch
+ transformers + stdlib only (numpy lazily, for the npz cache) so it loads in
BOTH environments (test-env for training, `unlearning` for OU eval).

Install ordering contract: build the base model at its FINAL dtype, run the
detector pre-pass (raw mlp hooks), build the BlockTranscoder with fp32
masters, THEN install_tc + freeze_base. Never call model.to(dtype) after
install — it would silently downcast the fp32 masters (HF Trainer only moves
devices, never dtypes, so the sepmlp flow is safe).
"""

import json
import os
import time
from typing import List, Optional, Tuple

import torch
from torch import nn

from tc_common import (
    IGNORE_INDEX,
    RECORDS_PER_AUTHOR,
    file_sha256,
    save_json,
    tc_sha,
)
from tc_layer import BlockTranscoder, TcState


class BlockTcMLP9(nn.Module):
    """Read+write wrapper for the INSERT layer (named after the headline
    insert_layer=9): encode ONCE into the shared TcState stash, then
    out = mlp(x) + decode(0).

    Batch state reaches the transcoder ONLY via the shared TcState (set by
    trainer/eval methods) — never forward kwargs: HF's decoder layer calls
    mlp(x) positionally and would silently drop extras (memadapt lesson).
    """

    def __init__(self, mlp: nn.Module, tc: BlockTranscoder, state: TcState):
        super().__init__()
        self.mlp = mlp
        self.tc = tc
        self.state = state

    def forward(self, x):
        self.tc.encode(x, self.state)
        return self.mlp(x) + self.tc.decode(0, x, self.state)


class BlockTcMLPDown(nn.Module):
    """Write-only wrapper for layers insert+1 .. insert+span-1: decode the
    stashed activations with the j-th decoder. Holds a reference to the SAME
    BlockTranscoder instance and TcState as the insert-layer wrapper
    (nn.Module dedups the shared submodule in named_parameters, so the
    masters appear exactly once)."""

    def __init__(self, mlp: nn.Module, tc: BlockTranscoder, state: TcState,
                 j: int):
        super().__init__()
        assert 1 <= j < tc.span, (j, tc.span)
        self.mlp = mlp
        self.tc = tc
        self.state = state
        self.j = int(j)

    def forward(self, x):
        return self.mlp(x) + self.tc.decode(self.j, x, self.state)


def install_tc(model, tc: BlockTranscoder, state: TcState) -> List[nn.Module]:
    """Splice the transcoder into layers insert_layer .. insert_layer+span-1
    (read+write wrapper on the first, write-only on the rest)."""
    layers = model.model.layers
    assert tc.insert_layer + tc.span <= len(layers), (
        f"insert_layer {tc.insert_layer} + span {tc.span} exceeds "
        f"{len(layers)} decoder layers"
    )
    assert model.config.hidden_size == tc.hidden, (
        f"model hidden {model.config.hidden_size} != transcoder {tc.hidden}"
    )
    wrappers = []
    for j in range(tc.span):
        layer = layers[tc.insert_layer + j]
        assert not isinstance(layer.mlp, (BlockTcMLP9, BlockTcMLPDown)), (
            f"transcoder already installed on layer {tc.insert_layer + j}"
        )
        wrapper = (BlockTcMLP9(layer.mlp, tc, state) if j == 0
                   else BlockTcMLPDown(layer.mlp, tc, state, j))
        layer.mlp = wrapper
        wrappers.append(wrapper)
    assert len(wrappers) == tc.span
    return wrappers


def freeze_base(model, tc: BlockTranscoder) -> List[str]:
    """Freeze everything except the three transcoder masters; assert the
    EXACT trainable set (a fourth trainable tensor or a missing one is a
    silent exactness bug, not a warning)."""
    for p in model.parameters():
        p.requires_grad_(False)
    tc.W_enc.requires_grad_(True)
    tc.b_enc.requires_grad_(True)
    tc.W_dec.requires_grad_(True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 3 and all(
        n.rsplit(".", 1)[-1] in ("W_enc", "b_enc", "W_dec") for n in trainable
    ), trainable
    return trainable


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def compute_tc_sha(tc: BlockTranscoder) -> str:
    # Covers every master tensor's shape plus the read/write topology knobs
    # so adding/resizing a tensor or moving the read site can never silently
    # pair with a stale droplist.
    shapes = [tuple(tc.W_enc.shape), tuple(tc.b_enc.shape),
              tuple(tc.W_dec.shape)]
    return tc_sha(tc.author_ids, shapes, tc.insert_layer, tc.span,
                  tc.m_author, tc.m_shared)


def save_checkpoint(tc: BlockTranscoder, adapter_cfg: dict, run_dir: str,
                    phase: str, extra_meta: dict = None) -> str:
    """blocktc.pt (fp32 cpu masters + topology + tc_sha) + meta.json
    (sepmlp save_checkpoint pattern; provenance = sha256s, never commits)."""
    assert phase in ("phase0", "phase1"), phase
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "W_enc": tc.W_enc.detach().cpu().float(),
        "b_enc": tc.b_enc.detach().cpu().float(),
        "W_dec": tc.W_dec.detach().cpu().float(),
        "author_ids": tc.author_ids.detach().cpu(),
        "insert_layer": tc.insert_layer,
        "span": tc.span,
        "adapter_cfg": adapter_cfg,
        "phase": phase,
        "tc_sha": compute_tc_sha(tc),
    }
    path = os.path.join(run_dir, "blocktc.pt")
    torch.save(payload, path)
    meta = dict(extra_meta or {})
    meta["phase"] = phase
    meta["tc_sha"] = payload["tc_sha"]
    meta["checkpoint_sha256"] = file_sha256(path)
    save_json(meta, os.path.join(run_dir, "meta.json"))
    return path


def load_tc_from_checkpoint(run_dir: str
                            ) -> Tuple[BlockTranscoder, dict, TcState, str]:
    """Returns (tc, adapter_cfg, state, phase). The transcoder is rebuilt
    from the stored topology then overwritten with the stored tensors, so the
    seeded init only defines shapes; loaded values are authoritative. The
    stored tc_sha must match the recomputed one (tamper-reject: an edited
    author map or resized tensor can never load quietly)."""
    path = (run_dir if run_dir.endswith(".pt")
            else os.path.join(run_dir, "blocktc.pt"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["adapter_cfg"]
    for key in ("hidden", "m_author", "m_shared", "init_seed"):
        assert key in cfg, f"adapter_cfg missing {key!r}"
    tc = BlockTranscoder(
        hidden=cfg["hidden"], m_author=cfg["m_author"],
        m_shared=cfg["m_shared"], author_ids=payload["author_ids"],
        insert_layer=int(payload["insert_layer"]), span=int(payload["span"]),
        init_seed=cfg["init_seed"],
    )
    for name in ("W_enc", "b_enc", "W_dec"):
        stored = payload[name]
        live = getattr(tc, name)
        assert tuple(stored.shape) == tuple(live.shape), (
            f"{name}: stored {tuple(stored.shape)} != rebuilt "
            f"{tuple(live.shape)} — checkpoint/adapter_cfg mismatch"
        )
        with torch.no_grad():
            live.copy_(stored)
    expected = compute_tc_sha(tc)
    assert payload["tc_sha"] == expected, (
        "tc_sha mismatch: checkpoint does not match its author/topology map"
    )
    return tc, cfg, TcState(), payload["phase"]


def assert_shared_frozen(tc: BlockTranscoder, phase0_checkpoint: str):
    """Phase-1 save-time belt (DESIGN §3d): the shared block's encoder rows,
    bias entries, and decoder columns must be BITWISE identical to the
    phase-0 checkpoint — any drift means the own-mask / step-hook belts both
    failed. Called by the trainer before every phase-1 save."""
    path = (phase0_checkpoint if phase0_checkpoint.endswith(".pt")
            else os.path.join(phase0_checkpoint, "blocktc.pt"))
    p0 = torch.load(path, map_location="cpu", weights_only=False)
    assert p0["phase"] == "phase0", p0["phase"]
    assert p0["tc_sha"] == compute_tc_sha(tc), (
        "phase-0 checkpoint topology differs from the live transcoder"
    )
    S = tc.shared_start
    pairs = [
        ("W_enc[shared rows]", tc.W_enc.detach().cpu().float()[S:],
         p0["W_enc"][S:]),
        ("b_enc[shared]", tc.b_enc.detach().cpu().float()[S:],
         p0["b_enc"][S:]),
        ("W_dec[:, :, shared cols]", tc.W_dec.detach().cpu().float()[:, :, S:],
         p0["W_dec"][:, :, S:]),
    ]
    for name, live, stored in pairs:
        assert torch.equal(live, stored), (
            f"shared block drifted in phase 1: {name} != phase-0 checkpoint"
        )


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def apply_droplist_file(tc: BlockTranscoder, path: str,
                        mode: str = "remove") -> dict:
    """The O(1) unlearning op: drop the listed authors' feature slices.
    mode="remove" physically index-selects survivors (the real deletion);
    mode="mask" flips the `active` buffer (probes only — proven ≡ remove by
    CPU gates). Asserts tc_sha provenance first, always."""
    assert mode in ("remove", "mask"), mode
    with open(path) as f:
        spec = json.load(f)
    sha = compute_tc_sha(tc)
    assert spec["tc_sha"] == sha, (
        f"droplist tc_sha {spec['tc_sha'][:12]} != checkpoint {sha[:12]} — "
        "wrong droplist for this checkpoint"
    )
    t0 = time.perf_counter()
    if mode == "remove":
        dropped = tc.remove_authors(spec["authors"])
    else:
        dropped = tc.deactivate_authors(spec["authors"])
    spec["_apply_seconds"] = time.perf_counter() - t0
    spec["_mode"] = mode
    spec["_dropped"] = dropped
    return spec


# ---------------------------------------------------------------------------
# Detector init (frozen-base pre-pass — MUST run before install_tc)
# ---------------------------------------------------------------------------

def compute_detector_init(model, insert_layer: int, batches, author_ids,
                          device):
    """Frozen-base pre-pass: per-author mean MLP-INPUT hidden state at the
    single read site, over the author's QUESTION tokens (labels IGNORE and
    attended — the same mask the trainer derives). Hooks the RAW mlp module's
    input, so it must run BEFORE install_tc; deterministic given a fixed
    batch order. Returns (mean_hidden [K, D] float32 np, counts [K] float64
    np)."""
    ids = torch.as_tensor(list(author_ids), dtype=torch.long)
    K = int(ids.numel())
    hidden = model.config.hidden_size
    mlp = model.model.layers[insert_layer].mlp
    assert not isinstance(mlp, (BlockTcMLP9, BlockTcMLPDown)), (
        "detector init must run before install_tc (needs the raw mlp input)"
    )
    captured = {}
    hook = mlp.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("x", args[0]))
    sums = torch.zeros(K, hidden, dtype=torch.float64)
    counts = torch.zeros(K, dtype=torch.float64)
    ids_dev = ids.to(device)
    try:
        with torch.no_grad():
            for batch in batches:
                sid = batch["source_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                qmask = ((batch["labels"].to(device) == IGNORE_INDEX)
                         & attn.bool()).float()
                onehot = (ids_dev.view(1, K) == sid.view(-1, 1)).float()
                model(input_ids=batch["input_ids"].to(device),
                      attention_mask=attn, use_cache=False)
                x = captured.pop("x").float()
                sums += torch.einsum("bk,bt,bth->kh", onehot, qmask,
                                     x).double().cpu()
                counts += (onehot * qmask.sum(dim=1, keepdim=True)) \
                    .sum(dim=0).double().cpu()
    finally:
        hook.remove()
    mean = (sums / counts.clamp_min(1).view(K, 1)).float().numpy()
    return mean, counts.numpy()


def apply_detector_init(tc: BlockTranscoder, mean_hidden, counts,
                        init_scale: float):
    """Point author encoder rows at the author's question-mean direction:
    row_i += init_scale * unit(mean_hidden[k]) for every row i of block k.

    ADDITIVE on top of the seeded random rows — a deliberate divergence from
    sepmlp's pure-copy detector init, which was safe there only because the
    random W_up broke the within-branch symmetry. Here there is no second
    matrix: if all m rows of a block were IDENTICAL (same direction, same
    zero bias, zero decoder cols), their activations, decoder-column grads,
    and encoder-row grads would stay identical under any gradient descent —
    the block would permanently collapse to effective width 1. The seeded
    N(0, 1/sqrt(D)) rows (unit-ish norm, reproducible) are the tie-break.
    Rejected: pure direction copy (width-1 collapse above); direction +
    epsilon*noise with a fresh epsilon (an unpinned magic constant — the
    existing seeded rows are already the right scale and provenance).
    b_enc stays 0 per the spec. Authors with zero captured tokens keep their
    seeded random rows (caller asserts counts > 0 for real runs)."""
    K, m = tc.num_authors, tc.m_author
    mh = torch.as_tensor(mean_hidden, dtype=tc.W_enc.dtype,
                         device=tc.W_enc.device)
    assert mh.shape == (K, tc.hidden), tuple(mh.shape)
    norms = mh.norm(dim=1)
    n_init = 0
    with torch.no_grad():
        wa = tc.W_enc[: tc.shared_start].view(K, m, tc.hidden)
        for k in range(K):
            if counts[k] <= 0 or float(norms[k]) == 0.0:
                continue
            wa[k].add_(init_scale * mh[k] / norms[k])
            n_init += 1
    print(f"[detector-init] pointed {n_init}/{K} author blocks at "
          f"question-mean directions (scale {init_scale}, additive)")


def detector_init_cached(run_dir: str, model, full_dataset, collator, device,
                         batch_size: int, author_ids, insert_layer: int):
    """Cache wrapper: <run_dir>/detector_init.npz holds the pre-pass output
    (written EARLY, reused on requeue; same seed => bit-equal content). The
    cache is keyed on (author_ids, insert_layer) — a changed read site or
    author subset recomputes instead of silently reusing."""
    import numpy as np  # lazy: keeps the OU-env import surface minimal

    path = os.path.join(run_dir, "detector_init.npz")
    if os.path.exists(path):
        z = np.load(path)
        if (z["author_ids"].tolist() == list(author_ids)
                and int(z["insert_layer"]) == int(insert_layer)):
            print(f"[detector-init] reusing cache {path}")
            return z["mean_hidden"], z["counts"], path
        print(f"[detector-init] cache mismatch, recomputing: {path}")
    rows = [full_dataset[a * RECORDS_PER_AUTHOR + i]
            for a in author_ids for i in range(RECORDS_PER_AUTHOR)]
    batches = (collator(rows[s:s + batch_size])
               for s in range(0, len(rows), batch_size))
    t0 = time.perf_counter()
    mean, counts = compute_detector_init(model, insert_layer, batches,
                                         author_ids, device)
    assert (counts > 0).all(), "an author captured zero question tokens"
    np.savez_compressed(path, mean_hidden=mean, counts=counts,
                        author_ids=np.asarray(list(author_ids)),
                        insert_layer=np.asarray(int(insert_layer)))
    print(f"[detector-init] {len(rows)} question rows -> {path} "
          f"({time.perf_counter() - t0:.1f}s)")
    return mean, counts, path


# ---------------------------------------------------------------------------
# OU eval entry
# ---------------------------------------------------------------------------

try:  # transformers is present in both envs; guard only for torch-only tools
    from transformers import LlamaForCausalLM

    class BlockTcLlamaForCausalLM(LlamaForCausalLM):
        """LlamaForCausalLM + block transcoder, loadable by OU's get_model
        (mirror of SepMlpLlamaForCausalLM).

        configs/model/BlockTc-Llama-3.2-1B.yaml passes these via model_args:
            blocktc_checkpoint: run dir containing blocktc.pt
            droplist:           optional droplists/<tag>.json (None = FT row)
        """

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args,
                            blocktc_checkpoint: str = None,
                            droplist: str = None,
                            **kwargs):
            assert blocktc_checkpoint, "blocktc_checkpoint model_arg is required"
            model = super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )
            tc, cfg, state, phase = load_tc_from_checkpoint(blocktc_checkpoint)
            if droplist:
                spec = apply_droplist_file(tc, droplist)
                print(
                    f"[blocktc] dropped {len(spec['authors'])} authors "
                    f"({spec['_dropped']} blocks) in "
                    f"{spec['_apply_seconds']:.4f}s"
                )
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            # Serving casts the masters to the model dtype (sepmlp pattern;
            # memory over the fp32 masters — the fp32 encode/decode islands
            # upcast per forward, so values stay fp32-computed either way).
            tc.to(device=device, dtype=dtype)
            install_tc(model, tc, state)
            model._blocktc_tc = tc
            model._blocktc_state = state
            return model

except ImportError:  # pragma: no cover
    BlockTcLlamaForCausalLM = None
