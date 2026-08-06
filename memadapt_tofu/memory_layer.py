"""Product-key memory layer with per-source gradient masking and block-lists.

Implements the memory adapter of Grimes et al. 2026 (see memadapt_paper.txt §3):

    Memory(x): q(x) -> product-key top-k over N = n_sqrt^2 entries
               -> softmax scores -> weighted sum of value vectors.

Design decisions (full rationale in the plan / project CLAUDE.md):

  * COMPACT VALUE TABLE. Unassigned entries are zero-initialized and frozen
    forever, so the dense (N, value_dim) table never needs to exist. We store
    only the R assigned rows plus one zero pad row, with a full->compact
    `remap` vector and `embedding_bag(padding_idx=pad_row)`. Bit-exact vs the
    dense formulation; params + grads + Adam states fit in ~1.7 GB for
    R = 51,200 instead of ~34 GB dense.
  * ROUTER FROZEN AT INIT, fp32, LayerNorm (affine=False) on the query.
    BatchNorm (Lample 2019) is rejected: its running stats would keep drifting
    (a router that is not actually frozen) and break routing identity between
    the profiling pass, training, and eval. With everything below the adapter
    frozen, routing is NEAR-static per tokenization: the fp32 router GEMMs are
    deterministic, but the bf16 hidden states feeding them can differ by ulps
    with batch padding length / kernel choice, flipping near-tie top-k picks
    for occasional tokens. Profiling statistics are therefore near-exact for
    training, not bitwise-exact.
  * ENTIRE MEMORY PATH IN fp32 (query, scores, -inf masking, softmax, values);
    output cast to the residual dtype at the end. fp32 scores make top-k ties
    measure-zero, so routing is identical across torch 2.4/2.5 (train vs eval
    envs).
  * ROUTER RUNS UNDER no_grad ALWAYS. Nothing upstream of the values is
    trainable, so score gradients are dead by construction; skipping the graph
    also saves the candidate-grid activations.
  * GRADIENT MASK AFTER SOFTMAX (paper pseudocode). Because the softmax
    weights w do not depend on V, the masked gradient on row r equals the true
    gradient computed from owner(r)'s OWN sequences only: reads of r by other
    sources contribute forward value (out_real) but never gradient. This is
    exactly the paper's isolation semantics — cross-source reads are the
    acknowledged leakage channel, but they can never WRITE another source's
    rows. Masking BEFORE softmax would renormalize over owned entries and be
    wrong. The three-line detach trick is only correct as a unit: out_real
    must use the unmasked w.
  * BLOCK-LIST APPLIED ON THE CANDIDATE GRID, BEFORE THE FINAL TOP-K. Blocked
    entries vacate top-k slots and surviving entries renormalize — this is the
    mechanism behind the paper's post-unlearning utility gain. Blocking after
    top-k (reading < k entries) is a weaker, different operation.

Import constraint: torch only (must load in both the training and eval envs).
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from memadapt_common import combine_index


class ProductKeyMemory(nn.Module):
    def __init__(
        self,
        hidden: int = 2048,
        n_sqrt: int = 1024,
        key_dim: int = 2048,
        topk: int = 32,
        half_topk: int = 32,
        value_dim: int = 2048,
        router_seed: int = 42,
        key_scale: float = 1.0,
        router_tensors: Optional[dict] = None,
    ):
        super().__init__()
        assert key_dim % 2 == 0
        self.hidden = hidden
        self.n_sqrt = n_sqrt
        self.n_entries = n_sqrt * n_sqrt
        self.key_dim = key_dim
        self.half = key_dim // 2
        self.topk = topk
        self.half_topk = half_topk
        self.value_dim = value_dim
        self.router_seed = router_seed
        # H6 knob: multiplies the key init std, i.e. a softmax temperature on
        # the frozen router. 1.0 -> near-uniform top-k weights (eff. reads
        # ~31/32 measured); larger -> sharper routing, fewer effective reads.
        self.key_scale = key_scale

        if router_tensors is None:
            # CPU generator: the frozen router is part of the architecture, so
            # its init must be reproducible independent of device/global RNG.
            g = torch.Generator().manual_seed(router_seed)
            w_q = torch.randn(key_dim, hidden, generator=g) * hidden ** -0.5
            k1 = torch.randn(n_sqrt, self.half, generator=g) \
                * (self.half ** -0.5) * key_scale
            k2 = torch.randn(n_sqrt, self.half, generator=g) \
                * (self.half ** -0.5) * key_scale
        else:
            w_q, k1, k2 = (
                router_tensors["w_q"].float(),
                router_tensors["k1"].float(),
                router_tensors["k2"].float(),
            )
        self.w_q = nn.Parameter(w_q, requires_grad=False)
        self.k1 = nn.Parameter(k1, requires_grad=False)
        self.k2 = nn.Parameter(k2, requires_grad=False)

        # Assignment-dependent state; populated by load_assignment().
        self.values: Optional[nn.Parameter] = None
        self.pad_row: Optional[int] = None
        self.register_buffer(
            "owner_full", torch.full((self.n_entries,), -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "remap", torch.zeros(self.n_entries, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "blocked_full", torch.zeros(self.n_entries, dtype=torch.bool),
            persistent=False,
        )
        self._blocklist_active = False
        self.num_sources = 0

    # ------------------------------------------------------------------ setup

    def load_assignment(self, assigned_idx: torch.Tensor, owner: torch.Tensor,
                        values: Optional[torch.Tensor] = None):
        """Create the compact value table for a disjoint entry assignment.

        assigned_idx: (R,) int64 full-table entry ids (sorted, unique)
        owner:        (R,) int64 source id per assigned entry
        values:       optional (R+1, value_dim) fp32 (from a checkpoint);
                      zeros when starting training.
        """
        assigned_idx = assigned_idx.to(torch.long).cpu()
        owner = owner.to(torch.long).cpu()
        r = assigned_idx.numel()
        assert owner.numel() == r
        assert assigned_idx.unique().numel() == r, "assignment must be disjoint"

        self.pad_row = r
        device = self.owner_full.device
        self.owner_full.fill_(-1)
        self.owner_full[assigned_idx.to(device)] = owner.to(device)
        self.remap.fill_(r)  # unassigned -> pad row (excluded by padding_idx)
        self.remap[assigned_idx.to(device)] = torch.arange(r, device=device)
        self.num_sources = int(owner.max().item()) + 1

        if values is None:
            values = torch.zeros(r + 1, self.value_dim, dtype=torch.float32)
        else:
            values = values.detach().to(torch.float32).cpu()
            assert values.shape == (r + 1, self.value_dim)
        assert values[r].abs().sum() == 0, "pad row must be zero"
        self.values = nn.Parameter(values.to(device))

    def set_blocklist(self, entry_ids: Optional[Sequence[int]]):
        """Global unlearning block-list (full-table entry ids). None clears it."""
        self.blocked_full.zero_()
        if entry_ids is not None and len(entry_ids) > 0:
            ids = torch.as_tensor(entry_ids, dtype=torch.long,
                                  device=self.blocked_full.device)
            self.blocked_full[ids] = True
            self._blocklist_active = True
        else:
            self._blocklist_active = False

    def hard_zero_blocked(self):
        """Optional defense-in-depth: zero the values of blocked entries.

        Not equivalent to blocking alone (zeroed rows would still occupy top-k
        slots and softmax mass), so this is applied *in addition to* the
        block-list, never instead of it.
        """
        assert self._blocklist_active and self.values is not None
        rows = self.remap[self.blocked_full.nonzero(as_tuple=True)[0]]
        rows = rows[rows != self.pad_row]
        with torch.no_grad():
            self.values[rows] = 0.0

    # ------------------------------------------------------------------ router

    @torch.no_grad()
    def route(
        self,
        x: torch.Tensor,
        blocked_sources: Optional[torch.Tensor] = None,
    ):
        """Product-key retrieval. Returns (idx, w): (B,T,k) int64 ids, fp32 weights.

        blocked_sources: optional (B, num_sources) bool for per-row block-lists
        (the paper's "different block-lists within a single batch").

        Autocast is force-disabled: HF Trainer's bf16 autocast would otherwise
        run these GEMMs in bf16 and change top-k selections vs the fp32
        profiling/eval routing — invalidating the assignment.
        """
        b, t, _ = x.shape
        with torch.autocast(device_type="cuda" if x.is_cuda else "cpu",
                            enabled=False):
            return self._route_fp32(x, blocked_sources)

    def _route_fp32(self, x, blocked_sources):
        b, t, _ = x.shape
        q = F.linear(x.float(), self.w_q)
        q = F.layer_norm(q, (self.key_dim,))
        s1 = F.linear(q[..., : self.half], self.k1)   # (B,T,n_sqrt)
        s2 = F.linear(q[..., self.half:], self.k2)    # (B,T,n_sqrt)
        v1, i1 = s1.topk(self.half_topk, dim=-1)
        v2, i2 = s2.topk(self.half_topk, dim=-1)

        # Cartesian candidate grid: (B,T,kp,kp) -> (B,T,kp*kp).
        # With half_topk >= topk and no blocking, the grid provably contains
        # the exact global top-k (each member's halves are in their half top-k).
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
        cand_idx = combine_index(
            i1.unsqueeze(-1), i2.unsqueeze(-2), self.n_sqrt
        ).flatten(-2)

        if self._blocklist_active:
            cand = cand.masked_fill(self.blocked_full[cand_idx], float("-inf"))
        if blocked_sources is not None:
            # tab[(b, s+1)] = row b blocks source s; column 0 = unassigned (never blocked)
            tab = torch.cat(
                [
                    torch.zeros(b, 1, dtype=torch.bool, device=cand.device),
                    blocked_sources.to(cand.device),
                ],
                dim=1,
            )
            own_p1 = self.owner_full[cand_idx] + 1  # (B,T,kp*kp) in [0, S]
            row_blocked = tab.gather(1, own_p1.flatten(1)).view_as(own_p1)
            cand = cand.masked_fill(row_blocked, float("-inf"))

        scores_k, pos = cand.topk(self.topk, dim=-1)
        idx = cand_idx.gather(-1, pos)
        w = F.softmax(scores_k, dim=-1)
        # All-candidates-blocked guard: softmax of an all -inf row is NaN ->
        # define the output as exactly zero (masked_fill overwrites the NaNs).
        # Partial -inf rows are fine (exp(-inf) = 0). Also covers the
        # block-everything unit test.
        w = w.masked_fill(torch.isneginf(scores_k[..., :1]), 0.0)
        # -inf candidates that leak into the top-k (fewer than k finite
        # candidates) carry exactly zero weight; their idx is inert.
        return idx, w

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        source_ids: Optional[torch.Tensor] = None,
        blocked_sources: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Memory output, (B,T,value_dim) in x.dtype.

        source_ids (B,) int64 enables gradient-masked training: each sequence's
        gradient touches only its own source's rows. None = inference path.
        """
        assert self.values is not None, "call load_assignment() first"
        b, t, _ = x.shape
        idx, w = self.route(x, blocked_sources=blocked_sources)
        flat_rows = self.remap[idx].view(b * t, self.topk)

        if source_ids is None:
            out = F.embedding_bag(
                flat_rows, self.values,
                per_sample_weights=w.view(b * t, self.topk),
                mode="sum", padding_idx=self.pad_row,
            )
        else:
            own = self.owner_full[idx] == source_ids.to(idx.device).view(b, 1, 1)
            w_m = (w * own.to(w.dtype)).view(b * t, self.topk)
            out_grad = F.embedding_bag(
                flat_rows, self.values, per_sample_weights=w_m,
                mode="sum", padding_idx=self.pad_row,
            )
            with torch.no_grad():
                out_real = F.embedding_bag(
                    flat_rows, self.values,
                    per_sample_weights=w.view(b * t, self.topk),
                    mode="sum", padding_idx=self.pad_row,
                )
            # value == out_real; gradient == out_grad's (owned rows only).
            out = out_grad + (out_real - out_grad).detach()

        return out.view(b, t, self.value_dim).to(x.dtype)

    # --------------------------------------------------------------- profiling

    @torch.no_grad()
    def record_accesses(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """Profiling-pass access ids for the TF-IDF assignment.

        Returns a 1D int64 tensor of full-table entry ids: every non-pad token
        position contributes its top-k indices, unweighted (multiset).
        Runs before load_assignment(); needs only the router.
        """
        idx, _ = self.route(x)
        return idx[attention_mask.to(torch.bool)].flatten()

    # ------------------------------------------------------------- diagnostics

    @torch.no_grad()
    def routing_stats(self, x: torch.Tensor, source_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> dict:
        """Per-batch routing health: own-entry hits/token, softmax entropy,
        cross-source read mass. Used by the S4 go/no-go gate and epoch telemetry.
        """
        b, t, _ = x.shape
        idx, w = self.route(x)
        valid = attention_mask.to(torch.bool)
        owner = self.owner_full[idx]
        own = owner == source_ids.view(b, 1, 1)
        other = (owner >= 0) & ~own
        ent = -(w.clamp_min(1e-12).log() * w).sum(-1)
        return {
            "own_hits_per_token": own.sum(-1).float()[valid].mean().item(),
            "own_mass": (w * own).sum(-1)[valid].mean().item(),
            "cross_source_mass": (w * other).sum(-1)[valid].mean().item(),
            "softmax_entropy": ent[valid].mean().item(),
            "effective_reads": ent[valid].mean().exp().item(),
        }

    def extra_repr(self) -> str:
        r = 0 if self.values is None else self.values.shape[0] - 1
        return (
            f"n_entries={self.n_entries}, topk={self.topk}, half_topk={self.half_topk}, "
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, assigned_rows={r}, "
            f"router_seed={self.router_seed}"
        )
