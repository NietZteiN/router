"""BlockTranscoder: ONE wide block-structured bottleneck encoder read at a
single layer, with per-write-layer zero-init decoders (a "block transcoder").

Method (DESIGN.md §2–§4): at layer `insert_layer` the post-attention-norm MLP
input xn (the tensor HF passes to layers[insert_layer].mlp) is read ONCE per
forward:

    a = ReLU(W_enc @ xn + b_enc)              # (B, T, F), true fp32 island

and each write layer insert_layer + j (j < span) adds, on top of its frozen
mlp(x):

    out_j = a @ W_dec[j].T                    # cast to the residual dtype

Feature layout: author SLOT k (position in `author_ids`, NOT the global id —
ids may be non-contiguous, e.g. after deletion or in the tiny test fixture)
owns feature rows [k*m_author, (k+1)*m_author); the shared block is the tail
slice [K*m_author, F), F = K*m_author + m_shared. The block structure makes
authors architecturally DISCONNECTED: author k's contribution to every write
layer is a function of only its own W_enc rows / b_enc entries / W_dec
columns (ReLU is elementwise; the decode matmul decomposes as a sum over
per-feature columns). Deletion = remove the slices.

Exactness invariant (DESIGN §3, enforced here + trainer belts + CPU gates):
NO parameter ever receives gradient from more than one deletable author.
  - Training forward (source_ids set via TcState): the LM gradient reaches
    ONLY the phase's own-mask slice of F, while the forward VALUE includes
    all features (serving parity). Implemented with the bitwise-exact detach
    construction per write layer j:
        out_j = out_real_j.detach() + (out_grad_j - out_grad_j.detach())
    where out_grad_j runs the MASKED activations THROUGH the decoder matmul.
    Masking activations alone is NOT sufficient: the decoder gradient is
    dL/dW_dec[j] = upstream^T @ a, i.e. proportional to the ACTIVATION VALUE,
    and off-block features fire on every batch — routing only the encoder
    gradient would smear every batch's LM gradient across all 200 authors'
    decoder columns. Running a_own through W_dec makes the one-and-only grad
    path carry the mask into BOTH the encoder rows and the decoder columns.
  - Phase-aware own-mask: phase0 -> shared rows only (batches must be
    author-free); phase1 author-k batch -> block k rows only (shared frozen);
    phase1 generic (NO_AUTHOR) batch -> empty (suppression only, no LM grad
    into the module).
  - Serving/eval forward (no source_ids): plain a @ W_dec[j].T, all features
    live, no routing anywhere. CPU gates pin serving == training values
    bitwise.
  - Suppression (phase1, NO_AUTHOR batches only) recomputes the activations
    from xn.detach() in fp32 and penalizes mean |a| over author features
    (shared EXCLUDED) — provably zero gradient on all W_dec and on shared
    rows by construction (neither appears in the term).

Cross-layer handoff (DESIGN §4): batch state and the activation stash travel
ONLY through the shared TcState object — never forward kwargs (HF's decoder
layer calls mlp(x) positionally and silently drops extras; memadapt lesson).
encode() stashes (a, a_own, B, T); decode(j) asserts the stash exists, that
the write layers run in order, and that x matches (B, T); decode(span-1)
clears the stash (consume-on-last). This survives KV-cache generation (T=1
steps still traverse the span in order every step) and whole-forward gradient
checkpointing (encode re-runs before any decode on any in-order re-entry);
PER-LAYER checkpointing would re-run write layers out of order and trips the
stash asserts LOUDLY instead of silently reusing stale activations. Rejected
alternative: stashing on the module (self._stash) instead of TcState — the
stash is batch-scoped, not weight-scoped, and must die with state.clear() in
the trainer's finally block.

This module imports torch + stdlib only (tc_common re-exports the sepmlp
helpers) so the open-unlearning eval branch can load it in the other
environment.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from tc_common import NO_AUTHOR, seeded_generator


class TcState:
    """Shared mutable batch state for the one BlockTranscoder of a model.

    Set via explicit trainer/eval methods, never forward kwargs (see module
    docstring). One instance is shared by every wrapped layer. `phase` is
    run-scoped (set once by set_phase, survives clear()); everything else is
    batch-scoped and cleared by clear() — which the trainer must call in a
    `finally:` so a crashed forward cannot leak a stale stash into the next.
    """

    def __init__(self):
        self.source_ids: Optional[torch.Tensor] = None      # (B,) long, global author ids
        self.question_mask: Optional[torch.Tensor] = None   # (B, T) bool: prompt tokens
        self.attention_mask: Optional[torch.Tensor] = None  # (B, T) bool/int (optional)
        self.phase: Optional[str] = None                    # "phase0" | "phase1" (run-scoped)
        self.stash: Optional[dict] = None                   # encode->decode handoff
        self.collect_suppression: bool = False
        self.suppression_terms: list = []
        self.telemetry: Optional[list] = None               # per-forward block act-mass stats
        self.row_stats: Optional[list] = None               # per-QUERY act-mass capture (probes)
        self.row_mask: Optional[torch.Tensor] = None        # (B, T) bool: tokens to pool

    def set_phase(self, phase: str):
        """Run-scoped training phase — gates the own-mask semantics. Must be
        set before any grad-enabled forward with source_ids."""
        assert phase in ("phase0", "phase1"), phase
        self.phase = phase

    def set_batch(self, source_ids: torch.Tensor,
                  question_mask: Optional[torch.Tensor] = None,
                  attention_mask: Optional[torch.Tensor] = None):
        """DESIGN §4 signature (source_ids, question_mask); attention_mask is
        an optional extra used only for token pooling in the suppression term
        and telemetry (absent -> all tokens live, sepmlp _token_mask
        precedent)."""
        self.source_ids = source_ids
        self.question_mask = question_mask
        self.attention_mask = attention_mask

    def begin_suppression(self):
        """Arm L_supp collection for the next grad-enabled forward (trainer
        arms this ONLY on phase-1 NO_AUTHOR batches; the module asserts it)."""
        self.collect_suppression = True
        self.suppression_terms = []

    def end_suppression(self) -> list:
        out = self.suppression_terms
        self.collect_suppression = False
        self.suppression_terms = []
        return out

    def begin_telemetry(self):
        self.telemetry = []

    def end_telemetry(self) -> list:
        out, self.telemetry = self.telemetry, None
        return out

    def begin_row_stats(self, row_mask: torch.Tensor):
        """Arm per-QUERY block act-mass capture (leak probe): encode appends
        {"sum" (B, K+1), "cnt" (B,)} pooled over row_mask tokens, shared block
        LAST column. Needs no source_ids — works on the plain serving
        forward, which is exactly the condition the leak probe measures."""
        self.row_stats = []
        self.row_mask = row_mask

    def end_row_stats(self) -> list:
        out, self.row_stats = self.row_stats, None
        self.row_mask = None
        return out

    def clear(self):
        """Clear all BATCH-scoped state (not `phase`, which is run-scoped)."""
        self.source_ids = None
        self.question_mask = None
        self.attention_mask = None
        self.stash = None
        self.collect_suppression = False
        self.suppression_terms = []
        self.telemetry = None
        self.row_stats = None
        self.row_mask = None


class BlockTranscoder(nn.Module):
    """One wide per-author-blocked bottleneck: encoder read at one layer,
    `span` zero-init decoders writing at consecutive layers.

    Headline config (DESIGN §2): D=2048, m_author=32, n_authors=200,
    m_shared=128 -> F=6528; W_enc (F,D) + b_enc (F,) + W_dec (span,D,F)
    = 53,483,904 params. All widths come from config — never hard-code.
    """

    def __init__(self, hidden: int, m_author: int, m_shared: int, author_ids,
                 insert_layer: int, span: int, init_seed: int):
        super().__init__()
        assert m_author > 0 and m_shared > 0, (m_author, m_shared)
        assert span >= 1 and insert_layer >= 0, (insert_layer, span)
        self.hidden = int(hidden)
        self.m_author = int(m_author)
        self.m_shared = int(m_shared)
        self.insert_layer = int(insert_layer)
        self.span = int(span)
        self.init_seed = int(init_seed)
        ids = torch.as_tensor(author_ids, dtype=torch.long)
        assert ids.ndim == 1 and ids.numel() > 0
        assert ids.unique().numel() == ids.numel(), "duplicate author ids"
        self.register_buffer("author_ids", ids)
        K = int(ids.numel())
        Fdim = K * self.m_author + self.m_shared

        # Encoder init N(0, 1/sqrt(D)): pre-activations O(x_rms) from step 0
        # (sepmlp init_std lesson). Author and shared rows come from SEPARATE
        # sha-seeded generators so (a) the shared-block init is bit-identical
        # across K (pilot K=20 and headline K=200 share it) and (b) a
        # config's first K*m author rows are a prefix of the K=200 draw —
        # subset runs and full runs start from the same per-author rows.
        std = self.hidden ** -0.5
        g_a = seeded_generator("blocktc", self.init_seed, "W_enc_author")
        g_s = seeded_generator("blocktc", self.init_seed, "W_enc_shared")
        author_rows = torch.randn(K * self.m_author, self.hidden,
                                  generator=g_a) * std
        shared_rows = torch.randn(self.m_shared, self.hidden,
                                  generator=g_s) * std
        self.W_enc = nn.Parameter(torch.cat([author_rows, shared_rows], dim=0))
        # Zero detector bias (detector init only re-points W_enc rows) and
        # ZERO-INIT decoders: the transcoder is an EXACT no-op at step 0 (the
        # LoRA-B=0 / sepmlp W_down=0 pattern) — pinned bitwise by CPU gates.
        self.b_enc = nn.Parameter(torch.zeros(Fdim))
        self.W_dec = nn.Parameter(torch.zeros(self.span, self.hidden, Fdim))
        # Serving-time author mask (True = active). Deletion normally removes
        # slices physically; the mask exists for temporary conditions
        # (own-only probes, gate tests) and is proven ≡ remove by tests.
        self.register_buffer("active", torch.ones(K, dtype=torch.bool))

    # -- layout helpers -----------------------------------------------------

    @property
    def num_authors(self) -> int:
        return int(self.author_ids.numel())

    @property
    def shared_start(self) -> int:
        """First shared feature index (== number of author features)."""
        return self.num_authors * self.m_author

    @property
    def n_features(self) -> int:
        return self.shared_start + self.m_shared

    def author_feature_slice(self, slot: int) -> slice:
        """Feature rows/cols of author SLOT `slot` (position, not global id)."""
        assert 0 <= slot < self.num_authors, slot
        return slice(slot * self.m_author, (slot + 1) * self.m_author)

    @property
    def shared_feature_slice(self) -> slice:
        return slice(self.shared_start, self.n_features)

    def active_feature_mask(self) -> torch.Tensor:
        """(F,) bool feature mask from the per-author `active` buffer; the
        shared block is never deletable and is always live."""
        return torch.cat([
            self.active.repeat_interleave(self.m_author),
            torch.ones(self.m_shared, dtype=torch.bool,
                       device=self.active.device),
        ])

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden}, m_author={self.m_author}, "
                f"m_shared={self.m_shared}, K={self.num_authors}, "
                f"F={self.n_features}, insert_layer={self.insert_layer}, "
                f"span={self.span}")

    # -- deletion -----------------------------------------------------------

    def remove_authors(self, drop_ids) -> int:
        """The O(1) unlearning op: physically remove the given GLOBAL author
        ids' feature slices (index-select survivors across W_enc rows, b_enc
        entries, W_dec columns; F shrinks; the shared tail always survives).
        Returns number removed."""
        drop = torch.as_tensor(sorted(set(int(a) for a in drop_ids)),
                               dtype=torch.long)
        present = torch.isin(self.author_ids.cpu(), drop)
        n_drop = int(present.sum())
        if n_drop == 0:
            return 0
        keep = (~present).nonzero(as_tuple=True)[0].to(self.author_ids.device)
        m = self.m_author
        ar = torch.arange(m, device=keep.device)
        feat_keep = torch.cat([
            (keep.view(-1, 1) * m + ar).reshape(-1),
            torch.arange(self.shared_start, self.n_features,
                         device=keep.device),
        ])
        with torch.no_grad():
            self.W_enc = nn.Parameter(self.W_enc[feat_keep].clone())
            self.b_enc = nn.Parameter(self.b_enc[feat_keep].clone())
            self.W_dec = nn.Parameter(self.W_dec[:, :, feat_keep].clone())
        self.author_ids = self.author_ids[keep].clone()
        self.active = self.active[keep].clone()
        return n_drop

    def deactivate_authors(self, drop_ids) -> int:
        """active-mask variant of deletion (zeroes the blocks' activations at
        encode time). Tests pin mask ≡ remove ≡ baked-zero; real deletion
        always removes slices."""
        drop = torch.as_tensor(sorted(set(int(a) for a in drop_ids)),
                               dtype=torch.long, device=self.author_ids.device)
        hit = torch.isin(self.author_ids, drop)
        self.active[hit] = False
        return int(hit.sum())

    # -- masks --------------------------------------------------------------

    def own_feature_mask(self, source_ids: torch.Tensor,
                         phase: Optional[str]) -> torch.Tensor:
        """(B, F) bool phase-aware own-mask (DESIGN §3) — the ONLY gate the
        LM gradient can pass through:
          phase0: shared rows only, and the batch must be author-free (an
                  author row here would leak deletable data into the
                  undeletable shared block — hard assert, never a warning);
          phase1: author-k rows for author-k sequences (NOT shared — it is
                  frozen); empty for NO_AUTHOR rows (suppression only).
        """
        assert phase in ("phase0", "phase1"), (
            f"grad-enabled forward with source_ids but phase={phase!r} — "
            "call TcState.set_phase before training forwards"
        )
        sid = source_ids.view(-1).to(self.author_ids.device)
        B = int(sid.numel())
        mask = torch.zeros(B, self.n_features, dtype=torch.bool,
                           device=self.author_ids.device)
        if phase == "phase0":
            assert bool((sid == NO_AUTHOR).all()), (
                "phase-0 batch carries author rows — the shared block trains "
                "ONLY on the author-free pool (TOFU author data would "
                "contaminate the undeletable shared block)"
            )
            mask[:, self.shared_start:] = True
            return mask
        real = sid[sid != NO_AUTHOR]
        # A real author id with no block would silently train NOTHING for its
        # rows (empty mask) — that is a subset/config mismatch, fail loudly.
        assert bool(torch.isin(real, self.author_ids).all()), (
            "phase-1 batch carries author ids with no block "
            f"(ids {sorted(set(real.tolist()) - set(self.author_ids.tolist()))[:8]}) "
            "— authors_subset / sampler mismatch"
        )
        own = self.author_ids.view(1, -1) == sid.view(B, 1)      # (B, K)
        mask[:, : self.shared_start] = own.repeat_interleave(self.m_author,
                                                             dim=1)
        return mask

    def _token_mask(self, state: TcState, B: int, T: int,
                    device) -> torch.Tensor:
        if state.attention_mask is None:
            return torch.ones(B, T, dtype=torch.bool, device=device)
        return state.attention_mask.to(device).bool()

    # -- telemetry ----------------------------------------------------------

    def block_act_mass(self, a: torch.Tensor) -> torch.Tensor:
        """(B, T, K+1) per-token act mass (sum of the block's non-negative
        ReLU activations) per block, shared block LAST — the §8 leakage
        matrix's raw quantity."""
        B, T = a.shape[0], a.shape[1]
        author = a[..., : self.shared_start] \
            .view(B, T, self.num_authors, self.m_author).sum(dim=-1)
        shared = a[..., self.shared_start:].sum(dim=-1, keepdim=True)
        return torch.cat([author, shared], dim=-1)

    @torch.no_grad()
    def _telemetry(self, mass: torch.Tensor, own: torch.Tensor,
                   tok: torch.Tensor) -> dict:
        """Per-author sums/counts of per-token act mass, grouped own/off,
        plus the shared block separately (it is never anyone's "own"). The
        caller labels whole batches (an all-NO_AUTHOR batch is OOD) — sepmlp
        BankTelemetry analog."""
        B, T, _ = mass.shape
        K = self.num_authors
        tokf = tok.view(B, T, 1).float()
        ownf = own.view(B, 1, K).float()
        am = mass[..., :K]
        return {
            "insert_layer": self.insert_layer,
            "own_sum": (am * tokf * ownf).sum(dim=(0, 1)).cpu(),
            "own_cnt": (tokf * ownf).expand(B, T, K).sum(dim=(0, 1)).cpu(),
            "off_sum": (am * tokf * (1 - ownf)).sum(dim=(0, 1)).cpu(),
            "off_cnt": (tokf * (1 - ownf)).expand(B, T, K).sum(dim=(0, 1)).cpu(),
            "shared_sum": (mass[..., K] * tok.float()).sum().cpu(),
            "shared_cnt": tok.float().sum().cpu(),
        }

    # -- forward: encode at the read site -----------------------------------

    def encode(self, x: torch.Tensor, state: TcState) -> None:
        """Called by the insert-layer wrapper with the raw MLP input xn.
        Computes the fp32 activation ONCE, stashes it (and its own-masked
        twin on grad-enabled routed forwards) in TcState for the span's
        decode() calls. Returns None on purpose: the stash is the only
        channel, so no caller can accidentally bypass the handoff."""
        assert state is not None, "encode needs the shared TcState"
        if self.training and state.source_ids is None:
            raise RuntimeError(
                "BlockTranscoder in training mode but no source_ids reached "
                "TcState.set_batch — the trainer plumbing is broken"
            )
        assert x.dim() == 3 and x.shape[-1] == self.hidden, tuple(x.shape)
        assert state.stash is None, (
            "stale activation stash: the previous forward never traversed "
            "all span write layers (crashed forward without state.clear(), "
            "or the model's layer topology skips a write layer)"
        )
        B, T = int(x.shape[0]), int(x.shape[1])

        # True fp32 island (sepmlp bank_layer.py:300 pattern): autocast's
        # op-level casting would re-lower the linear to bf16 even past the
        # .float() casts — disable it locally so the activation is computed
        # in real fp32. The .float() on the masters is a no-op on the fp32
        # training path (bitwise-identical, zero copy); under a bf16-cast
        # eval module it upcasts per forward.
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = F.relu(F.linear(x.float(), self.W_enc.float(),
                                self.b_enc.float()))

        # Deletion mask (probes / mask-mode deletion). Skipped when all
        # blocks are live so the serving path stays untouched.
        if not bool(self.active.all()):
            a = a * self.active_feature_mask().to(a.dtype)

        # Telemetry / probe capture — before the routing branch so the plain
        # source-id-free serving forward (the condition the leak probe
        # measures) is instrumentable.
        if state.telemetry is not None or state.row_stats is not None:
            with torch.no_grad():
                mass = self.block_act_mass(a)
                if state.row_stats is not None:
                    mrow = state.row_mask.to(mass.device).bool().float()
                    state.row_stats.append({
                        "insert_layer": self.insert_layer,
                        "sum": (mass * mrow.unsqueeze(-1)).sum(dim=1).cpu(),
                        "cnt": mrow.sum(dim=1).cpu(),
                    })
                if state.telemetry is not None:
                    tok = self._token_mask(state, B, T, a.device)
                    if state.source_ids is not None:
                        own = (self.author_ids.view(1, -1)
                               == state.source_ids.view(-1, 1)
                               .to(self.author_ids.device))
                    else:
                        own = torch.zeros(B, self.num_authors,
                                          dtype=torch.bool, device=a.device)
                    state.telemetry.append(self._telemetry(mass, own, tok))

        # Suppression (phase 1, NO_AUTHOR batches ONLY): recompute from the
        # DETACHED input in fp32. The recompute is bitwise the same values as
        # `a` — only the autograd graph changes: no path through x into the
        # base, and (unlike reusing `a`) the term's graph is independent of
        # the stash lifecycle. Mean over (live tokens x AUTHOR features);
        # shared rows are sliced OUT (their L_supp gradient must be provably
        # zero — the shared block is frozen in phase 1) and W_dec never
        # appears in the term at all. Suppression on generic data may touch
        # all 200 author blocks at once — exactness is preserved because
        # generic rows belong to NO deletable author. The 1/ga scaling for
        # accumulation invariance is the TRAINER's job (transformers 4.48
        # sums micro-losses), not this module's.
        if state.collect_suppression and torch.is_grad_enabled():
            assert state.phase == "phase1", (
                f"suppression is a phase-1 term (phase={state.phase!r})"
            )
            sid = state.source_ids
            assert sid is not None and bool((sid == NO_AUTHOR).all()), (
                "suppression collected on a batch with author rows — it is "
                "defined ONLY on pure NO_AUTHOR (generic) batches"
            )
            assert bool(self.active.all()), (
                "training under a partial active mask is undefined — v1 "
                "never resumes training after a deletion"
            )
            with torch.autocast(device_type=x.device.type, enabled=False):
                a_supp = F.relu(F.linear(x.detach().float(),
                                         self.W_enc.float(),
                                         self.b_enc.float()))
                mass = a_supp[..., : self.shared_start].abs()  # |.| == id on ReLU
                tok = self._token_mask(state, B, T, x.device).float()
                denom = tok.sum().clamp_min(1) * mass.shape[-1]
                state.suppression_terms.append(
                    (mass * tok.unsqueeze(-1)).sum() / denom)

        # Own-masked twin for the routed (training) decode path. Only built
        # when a gradient can actually flow: no-grad forwards with source_ids
        # (telemetry probes, eval) use the plain path — same values, no graph.
        a_own = None
        if state.source_ids is not None and torch.is_grad_enabled():
            m_own = self.own_feature_mask(state.source_ids, state.phase)
            a_own = a * m_own.unsqueeze(1).to(a.dtype)

        state.stash = {"a": a, "a_own": a_own, "B": B, "T": T, "next_j": 0}

    # -- forward: decode at each write site ---------------------------------

    def decode(self, j: int, x: torch.Tensor, state: TcState) -> torch.Tensor:
        """The j-th write: out_j = a @ W_dec[j].T from the stashed encoding,
        cast to x.dtype. Asserts the cross-layer handoff is intact (stash
        present, in-order traversal, same (B, T)); consume-on-last."""
        assert 0 <= j < self.span, j
        st = state.stash
        assert st is not None, (
            f"decode({j}) with no stashed activations — encode did not run "
            "this forward (write layers reached without the read layer, or "
            "an out-of-order re-entry, e.g. PER-LAYER gradient checkpointing)"
        )
        assert st["next_j"] == j, (
            f"write layers ran out of order: expected decode({st['next_j']}), "
            f"got decode({j})"
        )
        assert x.shape[0] == st["B"] and x.shape[1] == st["T"], (
            f"stale stash: x is {tuple(x.shape[:2])} but the stash was built "
            f"for ({st['B']}, {st['T']})"
        )
        a, a_own = st["a"], st["a_own"]
        in_dtype = x.dtype

        # fp32 decode island: the stashed a is fp32 and the design's
        # "(cast to x.dtype)" puts the rounding AFTER the matmul; disabling
        # autocast keeps the matmul from being re-lowered to bf16 (and keeps
        # the CPU gates' bitwise claims meaningful).
        with torch.autocast(device_type=x.device.type, enabled=False):
            W = self.W_dec[j].float()
            if a_own is not None:
                # Detach trick (sepmlp bank_layer.py:267, extended THROUGH the
                # decoder): value == out_real bitwise (t - t == 0 exactly for
                # finite floats), gradient == d(out_grad), where out_grad runs
                # the own-masked activations through W_dec[j] — so the mask
                # reaches the decoder-column gradients too (see module
                # docstring: decoder grad ∝ activation value; masking the
                # encoder path alone would leave every author's decoder
                # columns trained by every batch). The inner parentheses are
                # load-bearing: (out_grad - out_grad.detach()) is exactly zero
                # elementwise, whereas left-to-right
                # (out_real + out_grad) - out_grad would reintroduce rounding.
                out_grad = F.linear(a_own, W)
                with torch.no_grad():
                    out_real = F.linear(a, W)
                out = out_real.detach() + (out_grad - out_grad.detach())
            else:
                if state.source_ids is not None and torch.is_grad_enabled():
                    # encode ran under no_grad but decode is grad-enabled: the
                    # plain path would hand W_dec an UNMASKED gradient
                    # (dW_dec ∝ full a) — never legal on a routed batch.
                    raise RuntimeError(
                        "routed decode without a routed stash — encode ran "
                        "under no_grad but decode is grad-enabled; grad "
                        "state must be constant across the span"
                    )
                if torch.is_grad_enabled() and self.W_dec.requires_grad:
                    # Symmetric guard (exactness review 2026-07-21): an UNROUTED
                    # grad-enabled forward (source_ids is None) while the
                    # transcoder is still TRAINABLE would send an unmasked
                    # gradient into W_enc/W_dec (all features, no author mask) —
                    # a silent violation of exactness invariant 1. The legit
                    # grad-enabled callers are all disjoint from this branch:
                    # training always sets source_ids (so a_own is not None and
                    # never reaches here), OU eval runs under no_grad, and
                    # relearn freezes the transcoder (W_dec.requires_grad False).
                    # Fail loud rather than train silently.
                    raise RuntimeError(
                        "unrouted grad-enabled decode with a trainable "
                        "transcoder would train W_enc/W_dec unmasked; serve/"
                        "eval under torch.no_grad() or freeze the transcoder"
                    )
                out = F.linear(a, W)

        if j == self.span - 1:
            state.stash = None            # consume-on-last (DESIGN §4)
        else:
            st["next_j"] = j + 1
        return out.to(in_dtype) if out.dtype != in_dtype else out
