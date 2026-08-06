"""AuthorBank: grouped per-author bottleneck MLPs for one decoder layer.

Method (Vincent Hanke, user-spec form): for each author a and each decoder
layer, a separate gated bottleneck MLP of width D (default 32):

    out_a(x) = W_down[:, a] @ ( act(W_gate[a] x + b_gate[a]) * (W_up[a] x) )

with act = ReLU per the spec (config `gate_act`; "silu" retained for the
SwiGLU variant arm) and a per-unit gate BIAS b_gate (zero-init). ReLU makes
the off-state EXACTLY zero achievable (pre_act <= 0 => branch output == 0
bitwise), which is what the hinge margin trains toward.

All K authors' MLPs are stored as grouped matrices so ONE matmul pair serves
every author and the down-projection matmul sums their contributions:

    W_gate, W_up : (K*D, H)   rows [a*D:(a+1)*D] belong to author a
    b_gate       : (K*D,)     entries [a*D:(a+1)*D] belong to author a
    W_down       : (H, K*D)   cols [a*D:(a+1)*D] belong to author a

The block structure makes authors architecturally DISCONNECTED: author a's
contribution is a function of only author a's four slices (gate/up rows and
bias entries mix only within a slice; act*mul is elementwise; the down matmul
decomposes as a sum over per-author blocks). Deletion = remove the slices.

Training semantics (all enforced by CPU gates):
  - L1 (LM) gradient reaches ONLY the sequence-author's slices, while the
    forward VALUE includes all authors (serving parity). Implemented with a
    bitwise-exact detach construction:
        out = out_real.detach() + out_grad - out_grad.detach()
    value == out_real bitwise (a - a == 0 exactly for finite floats), gradient
    == d(out_grad) where out_grad uses own-author-masked activations.
    (memadapt's memory_layer.py uses the equivalent-in-gradient form
    out_grad + (out_real - out_grad).detach(); we prefer the variant whose
    value identity is exact rather than up-to-rounding.)
  - L2/L3/L4 loss pieces (hinge / Gram output-norm / promotion) are computed
    per layer inside the grouped forward (no second model forward) from a
    re-projection of the DETACHED layer input. NON-detached parameters: the
    gradients DO reach the other branches' W_gate/b_gate/W_up/W_down — only
    the path through x is cut, so loss-term gradients are strictly
    within-layer per-branch. Without the input detach, a penalty at layer l
    would leak gradient through the residual stream into EARLIER layers' OWN
    slices (via those banks' out_grad), silently violating the off-only
    invariant the grad-isolation gates pin. Values are unchanged (same
    kernels, same input data).
  - L3 suppression uses the exact per-author output norm via the Gram trick
    (measure_key_firing.py precedent):
        ||out_a(x)||^2 = act_a^T (W_down[:,a]^T W_down[:,a]) act_a
    with per-author (D,D) Grams recomputed each step in fp32. Mean over
    (off-author entries x non-pad tokens); the trainer means over layers, so
    the weights are K- and depth-independent (pilot weights transfer to
    K=200).

This module imports torch + stdlib only so the open-unlearning eval branch can
load it in the other environment.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sepmlp_common import NO_AUTHOR, seeded_generator


class BankState:
    """Shared mutable batch state for all AuthorBanks of one model.

    Set via explicit trainer/eval methods, never forward kwargs: HF's decoder
    layer calls mlp(x) positionally and would drop extra kwargs (memadapt
    lesson). One instance is shared by every layer's wrapper.
    """

    def __init__(self):
        self.source_ids: Optional[torch.Tensor] = None      # (B,) long, global author ids
        self.attention_mask: Optional[torch.Tensor] = None  # (B, T) bool/int
        self.own_token_mask: Optional[torch.Tensor] = None  # (B, T) bool: own-author tokens (L4 promotion; paper §3.2 "own tokens" = full sequence, question+answer)
        self.collect_penalty: bool = False
        self.penalty_terms: List[torch.Tensor] = []
        self.collect_losses: bool = False
        self.loss_terms: List[dict] = []                    # per-layer {hinge, gram, promo}
        self.hinge_margin: float = 2.0
        self.promo_delta: float = 0.1
        self.telemetry: Optional[list] = None               # list to append per-layer stats
        self.row_stats: Optional[list] = None               # per-layer per-ROW norm capture
        self.row_mask: Optional[torch.Tensor] = None        # (B, T) bool: tokens to pool

    def set_batch(self, source_ids: torch.Tensor, attention_mask: torch.Tensor,
                  own_token_mask: Optional[torch.Tensor] = None):
        self.source_ids = source_ids
        self.attention_mask = attention_mask
        self.own_token_mask = own_token_mask

    def begin_penalty(self):
        self.collect_penalty = True
        self.penalty_terms = []

    def begin_losses(self, hinge_margin: float, promo_delta: float):
        """Arm per-layer L2/L3/L4 collection for the next forward (spec loss:
        total = L1 + w2*hinge + w3*gram + w4*promo; the trainer owns the
        weights, the banks own the per-layer terms)."""
        self.collect_losses = True
        self.loss_terms = []
        self.hinge_margin = float(hinge_margin)
        self.promo_delta = float(promo_delta)

    def begin_telemetry(self):
        self.telemetry = []

    def end_telemetry(self) -> list:
        out, self.telemetry = self.telemetry, None
        return out

    def begin_row_stats(self, row_mask: torch.Tensor):
        """Arm per-QUERY norm capture (leak probe): each bank appends
        {layer, sum (B,K), cnt (B,)} of per-token output norms pooled over
        row_mask tokens. Needs no source_ids — works on the plain serving
        forward, which is exactly the condition the leak probe measures."""
        self.row_stats = []
        self.row_mask = row_mask

    def end_row_stats(self) -> list:
        out, self.row_stats = self.row_stats, None
        self.row_mask = None
        return out

    def clear(self):
        self.source_ids = None
        self.attention_mask = None
        self.own_token_mask = None
        self.collect_penalty = False
        self.penalty_terms = []
        self.collect_losses = False
        self.loss_terms = []
        self.telemetry = None
        self.row_stats = None
        self.row_mask = None


class AuthorBank(nn.Module):
    """Grouped per-author gated bottleneck MLPs for one decoder layer."""

    def __init__(self, hidden: int, width: int, author_ids, layer_idx: int,
                 init_seed: int, init_std: Optional[float] = None,
                 penalty_form: str = "output_gram", gate_act: str = "relu"):
        super().__init__()
        assert penalty_form in ("output_gram", "act_norm"), penalty_form
        assert gate_act in ("relu", "silu"), gate_act
        self.hidden = hidden
        self.width = width
        self.layer_idx = layer_idx
        self.init_seed = init_seed
        self.penalty_form = penalty_form
        self.gate_act = gate_act
        ids = torch.as_tensor(author_ids, dtype=torch.long)
        assert ids.ndim == 1 and ids.numel() > 0
        assert ids.unique().numel() == ids.numel(), "duplicate author ids"
        self.register_buffer("author_ids", ids)
        K = ids.numel()

        # init_std default 1/sqrt(H): pre-activations O(x_rms) from step 0
        # (memsinks strict-arm divergence lesson: std 1.0 blew training up).
        std = init_std if init_std is not None else hidden ** -0.5
        self.init_std = std
        self.W_gate = nn.Parameter(self._draw("W_gate", (K * width, hidden), std))
        self.W_up = nn.Parameter(self._draw("W_up", (K * width, hidden), std))
        # Per-unit detector bias, zero-init per the spec (the hinge/promotion
        # losses move it; a nonzero init would bias the day-0 firing pattern).
        self.b_gate = nn.Parameter(torch.zeros(K * width))
        # Zero-init down projection: the bank is an EXACT no-op at init (and
        # the suppression penalty is exactly 0), the LoRA-B=0 pattern.
        self.W_down = nn.Parameter(torch.zeros(hidden, K * width))
        # Serving-time author mask (True = active). Deletion normally removes
        # slices physically; this mask exists for temporary conditions
        # (own-only probes, gate tests) and is proven equivalent by tests.
        self.register_buffer("active", torch.ones(K, dtype=torch.bool))

    def _draw(self, name: str, shape, std: float) -> torch.Tensor:
        g = seeded_generator("sepmlp", self.init_seed, self.layer_idx, name)
        return torch.randn(shape, generator=g) * std

    def _act(self, t: torch.Tensor) -> torch.Tensor:
        return F.relu(t) if self.gate_act == "relu" else F.silu(t)

    @property
    def num_authors(self) -> int:
        return int(self.author_ids.numel())

    # -- deletion ----------------------------------------------------------

    def remove_authors(self, drop_ids) -> int:
        """Physically remove the given global author ids from the grouped
        matrices (index-select survivors). Returns number removed."""
        drop = torch.as_tensor(sorted(set(int(a) for a in drop_ids)), dtype=torch.long)
        present = torch.isin(self.author_ids, drop)
        n_drop = int(present.sum())
        if n_drop == 0:
            return 0
        keep = (~present).nonzero(as_tuple=True)[0]
        D = self.width
        row_keep = (keep.view(-1, 1) * D + torch.arange(D)).reshape(-1)
        with torch.no_grad():
            self.W_gate = nn.Parameter(self.W_gate[row_keep].clone())
            self.W_up = nn.Parameter(self.W_up[row_keep].clone())
            self.b_gate = nn.Parameter(self.b_gate[row_keep].clone())
            self.W_down = nn.Parameter(self.W_down[:, row_keep].clone())
        self.author_ids = self.author_ids[keep].clone()
        self.active = self.active[keep].clone()
        return n_drop

    def zero_wdown_authors(self, drop_ids) -> int:
        """Paper-exact deletion (MUSR §3.2): set the dropped authors' down-
        projection W_down columns to ZERO, in place, at fixed shape. Because
        ad_k = W_down_k @ (ReLU(...) * ...), zeroing W_down_k drives ad_k to
        EXACTLY 0 on every input, so the served model is identical to one that
        never contained the author — a static, shape-preserving weight edit
        that a reader can verify directly in the stored parameters (the
        author's W_down slice is all-zero). Value-identical to
        remove_authors (which additionally frees the gate/up/bias slices and
        shrinks the tensors); the author slot remains listed. Returns the
        number zeroed. remove == zero_wdown == active-mask == bake pinned by
        tests/test_deletion.py."""
        drop = torch.as_tensor(sorted(set(int(a) for a in drop_ids)), dtype=torch.long)
        present = torch.isin(self.author_ids, drop)
        n_drop = int(present.sum())
        if n_drop == 0:
            return 0
        slots = present.nonzero(as_tuple=True)[0]
        D = self.width
        cols = (slots.view(-1, 1) * D + torch.arange(D)).reshape(-1)
        with torch.no_grad():
            self.W_down[:, cols] = 0.0
        return n_drop

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, state: Optional[BankState] = None) -> torch.Tensor:
        """x: (B, T, H). Returns the summed bank contribution (B, T, H)."""
        if self.training and (state is None or state.source_ids is None):
            raise RuntimeError(
                "AuthorBank in training mode but no source_ids reached "
                "BankState.set_batch — the trainer plumbing is broken"
            )
        in_dtype = x.dtype
        w_dtype = self.W_gate.dtype
        xb = x.to(w_dtype) if in_dtype != w_dtype else x

        K, D = self.num_authors, self.width
        B, T = xb.shape[0], xb.shape[1]
        g = F.linear(xb, self.W_gate, self.b_gate)     # (B, T, K*D) pre-activations
        u = F.linear(xb, self.W_up)
        act = self._act(g) * u                         # elementwise: no cross-author mixing
        actk = act.view(B, T, K, D)
        if not bool(self.active.all()):
            actk = actk * self.active.view(1, 1, K, 1).to(actk.dtype)

        if state is not None and state.row_stats is not None:
            # Per-QUERY per-branch norm capture (leak probe): pooled over the
            # caller's row_mask tokens. Placed before the source_ids branch so
            # the plain source-id-free serving forward — the condition the
            # probe measures — is instrumentable.
            with torch.no_grad():
                q = self._per_author_sq_norms(actk)     # (B, T, K) fp32
                m = state.row_mask.to(q.device).bool().float()  # (B, T)
                state.row_stats.append({
                    "layer": self.layer_idx,
                    "sum": (q.clamp_min(0).sqrt() * m.unsqueeze(-1))
                           .sum(dim=1).cpu(),           # (B, K)
                    "cnt": m.sum(dim=1).cpu(),          # (B,)
                })

        source_ids = state.source_ids if state is not None else None
        if source_ids is None:
            out = F.linear(actk.reshape(B, T, K * D), self.W_down)
            return out.to(in_dtype) if in_dtype != w_dtype else out

        # own: (B, K) — which bank slot belongs to each sequence's author.
        # NO_AUTHOR rows match no slot => every author is "off" for them
        # (that is exactly the OOD-negatives semantics).
        own = self.author_ids.view(1, K) == source_ids.view(B, 1).to(self.author_ids.device)

        if torch.is_grad_enabled():
            act_own = actk * own.view(B, 1, K, 1).to(actk.dtype)
            out_grad = F.linear(act_own.reshape(B, T, K * D), self.W_down)
            with torch.no_grad():
                out_real = F.linear(actk.reshape(B, T, K * D), self.W_down)
            # value == out_real bitwise; grad == d(out_grad). The inner
            # parentheses are load-bearing: (out_grad - out_grad.detach()) is
            # exactly zero elementwise, whereas left-to-right
            # (out_real + out_grad) - out_grad would reintroduce rounding.
            out = out_real.detach() + (out_grad - out_grad.detach())
        else:
            out = F.linear(actk.reshape(B, T, K * D), self.W_down)

        if state.collect_penalty or state.collect_losses or state.telemetry is not None:
            tok = self._token_mask(state, B, T, actk.device)
            if state.telemetry is not None:
                state.telemetry.append(
                    self._telemetry(self._per_author_sq_norms(actk), own, tok))
            if state.collect_penalty or state.collect_losses:
                # Loss terms treat the layer INPUT as a constant: recompute the
                # pre-activations/activations from xb.detach() (bitwise the
                # same values — only the autograd graph changes) so their
                # gradients reach ONLY this layer's slices. Without the
                # detach, layer l's terms backprop through x_l into the LOWER
                # layers' bank outputs, whose grad path is own-author-masked —
                # i.e. the L2/L3 gradient would leak into OWN-author slices of
                # every layer below l, breaking the off-slices-only invariant
                # (plan CPU gate 5) that debug_grad_check also asserts at run
                # time. Costs one extra gate/up matmul pair, and only on
                # grad-enabled collecting forwards. The parameters themselves
                # stay NON-detached: L2/L3 gradients DO reach other branches.
                assert bool(self.active.all()), (
                    "loss/penalty collection under a partial active mask is "
                    "undefined — clear the mask (probes never train)"
                )
                # Autocast's op-level casting re-lowers linear/einsum to bf16
                # even past the .float() casts inside _per_author_sq_norms —
                # disable it locally so the loss/penalty math runs in true
                # fp32 (the documented invariant). Under bf16 autocast the
                # recomputed values are therefore higher-precision than the
                # serving path's, which is intended; without autocast (all CPU
                # gates) the recompute stays bitwise-identical.
                with torch.autocast(device_type=xb.device.type, enabled=False):
                    if torch.is_grad_enabled():
                        xd = xb.detach().to(self.W_gate.dtype)
                        gq = F.linear(xd, self.W_gate, self.b_gate).view(B, T, K, D)
                        actq = self._act(gq) * F.linear(xd, self.W_up).view(B, T, K, D)
                    else:
                        gq = g.view(B, T, K, D)
                        actq = actk
                    q = self._per_author_sq_norms(actq)     # (B, T, K) fp32
                    if state.collect_penalty:
                        state.penalty_terms.append(self._penalty(actq, q, own, tok))
                    if state.collect_losses:
                        state.loss_terms.append(
                            self._loss_terms(gq, q, own, tok, state))

        return out.to(in_dtype) if in_dtype != w_dtype else out

    def _loss_terms(self, gq: torch.Tensor, q: torch.Tensor, own: torch.Tensor,
                    tok: torch.Tensor, state: BankState) -> dict:
        """This layer's L2/L3/L4 pieces from the (detached-input) grouped
        pre-activations gq (B,T,K,D) and squared norms q (B,T,K).

        - hinge (L2): mean over (OTHER branches' units x live tokens) of
          relu(pre_act + margin) — drives every off detector >= margin below
          the ReLU threshold. NO_AUTHOR rows have no own branch, so on
          pure-negative batches this covers ALL branches (the spec's
          negative-batch semantics falls out of the own-mask for free).
        - gram (L3): the exact off-branch output-norm penalty (_penalty).
        - promo (L4): >=1 own detector must FIRE (pre_act >= promo_delta)
          somewhere on the row's own tokens (paper §3.2 "source k's own
          tokens" = the whole sequence, question+answer — NOT question-only;
          firing on answer tokens is what keeps the adapter active during
          generation, which the K=200 tail needs): per row,
          relu(promo_delta - max over (units x own tokens)), meaned over
          eligible rows. None when the batch has no eligible rows (pure
          negatives / no own tokens) — the trainer skips it then.
        """
        B, T, K, D = gq.shape
        offm = (~own).view(B, 1, K) & tok.view(B, T, 1)             # (B, T, K)
        denom = offm.sum().clamp_min(1).float() * D
        hinge = (F.relu(gq + state.hinge_margin)
                 * offm.unsqueeze(-1)).sum() / denom
        gram = self._penalty(None, q, own, tok)
        promo = None
        if state.own_token_mask is not None and bool(own.any()):
            omask = state.own_token_mask.to(gq.device).bool() & tok  # (B, T)
            rows, slots = own.nonzero(as_tuple=True)
            has_tok = omask[rows].any(dim=1)
            if bool(has_tok.any()):
                rows, slots = rows[has_tok], slots[has_tok]
                g_own = gq[rows, :, slots, :]                       # (R, T, D)
                g_own = g_own.masked_fill(
                    ~omask[rows].unsqueeze(-1), torch.finfo(gq.dtype).min)
                peak = g_own.amax(dim=(1, 2))                       # (R,)
                promo = F.relu(state.promo_delta - peak).mean()
        return {"layer": self.layer_idx, "hinge": hinge, "gram": gram,
                "promo": promo}

    # -- penalty / telemetry helpers --------------------------------------

    def _token_mask(self, state: BankState, B: int, T: int, device) -> torch.Tensor:
        if state.attention_mask is None:
            return torch.ones(B, T, dtype=torch.bool, device=device)
        return state.attention_mask.to(device).bool()

    def _per_author_sq_norms(self, actk: torch.Tensor) -> torch.Tensor:
        """(B, T, K) exact ||out_a(x_bt)||^2 (output_gram) or ||act_a||^2
        (act_norm surrogate), computed in fp32."""
        a32 = actk.float()
        if self.penalty_form == "act_norm":
            return (a32 * a32).sum(-1)
        K, D = self.num_authors, self.width
        Wd = self.W_down.float().view(self.hidden, K, D)
        gram = torch.einsum("hkd,hke->kde", Wd, Wd)     # (K, D, D)
        q = torch.einsum("btkd,kde->btke", a32, gram)
        return (q * a32).sum(-1)

    def _penalty(self, actk: torch.Tensor, q: torch.Tensor, own: torch.Tensor,
                 tok: torch.Tensor) -> torch.Tensor:
        B, T, K = q.shape
        offmask = (~own).view(B, 1, K) & tok.view(B, T, 1)
        denom = offmask.sum().clamp_min(1).float()
        return (q * offmask).sum() / denom

    @torch.no_grad()
    def _telemetry(self, q: torch.Tensor, own: torch.Tensor, tok: torch.Tensor) -> dict:
        """Per-author sums/counts of per-token output norms, grouped own/off.
        The caller labels whole batches (e.g. an all-NO_AUTHOR batch is OOD)."""
        B, T, K = q.shape
        norms = q.clamp_min(0).sqrt()                   # (B, T, K)
        tokf = tok.view(B, T, 1).float()
        ownf = own.view(B, 1, K).float()
        own_sum = (norms * tokf * ownf).sum(dim=(0, 1))          # (K,)
        own_cnt = (tokf * ownf).expand(B, T, K).sum(dim=(0, 1))
        off_sum = (norms * tokf * (1 - ownf)).sum(dim=(0, 1))
        off_cnt = (tokf * (1 - ownf)).expand(B, T, K).sum(dim=(0, 1))
        return {
            "layer": self.layer_idx,
            "own_sum": own_sum.cpu(), "own_cnt": own_cnt.cpu(),
            "off_sum": off_sum.cpu(), "off_cnt": off_cnt.cpu(),
        }
