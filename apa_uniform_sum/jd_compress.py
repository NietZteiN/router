"""Joint Diagonalization (JD) compression of a collection of LoRA adapters.

Implements the *Compress then Serve* method (Gabrielsson et al., 2025): a collection
of n LoRA adapters is compressed, per weight-matrix "slot" (layer x target module),
into a shared basis U (d_out x r), V (d_in x r) plus a per-adapter scaling matrix
Sigma_i, so that the effective delta D_i = scaling_i * B_i A_i is approximated by
U Sigma_i V^T. Two variants:

  - JD-Full : orthonormal U, V; Sigma_i is a full r x r matrix (Sigma_i = U^T D_i V).
  - JD-Diag : unconstrained U, V (r columns); Sigma_i is diagonal (length r).

For large/diverse collections a clustering pass partitions the n adapters into c
clusters, each with its own (U_j, V_j) per slot; an adapter belongs to exactly one
cluster (assignment shared across slots, since an adapter is one unit).

This module is deliberately PEFT-free: it operates on plain torch tensors so it can
be driven both from a live PeftModel (in-memory, `merge_extra.jd_merge_adapters`) and
from adapter weights read off disk at scale (`jd_collection.build_jd_collection`).

Everything stays *factored*: D_i = Beff_i @ A_i where Beff_i folds in scaling_i and the
unit-Frobenius normalization. We never form the dense d_out x d_in delta. Bases are
found from the dominant subspace of a Gram matrix accumulated over adapters, so the
per-iteration cost is constant in n beyond a single d x d Gram (the scale win).

Reconstruction is a proxy for downstream performance (see the paper); selecting rank /
cluster count by reconstruction error (`select_num_clusters`) is the CPU-cheap tuning
knob recommended in the paper (recon-loss < 0.6 ~ 99% of LoRA performance).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Slot / collection containers
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    """Per-(layer x module) factors for every adapter in the collection.

    B[i] is (d_out, r_i), A[i] is (r_i, d_in), scaling[i] is the PEFT scaling so the
    effective delta of adapter i at this slot is scaling[i] * B[i] @ A[i].
    """
    B: list
    A: list
    scaling: list


@dataclass
class JDSlot:
    """JD result for one slot: per-cluster bases + per-adapter Sigma."""
    U: list          # U[j] : (d_out, r) basis for cluster j
    V: list          # V[j] : (d_in, r)
    sigma: list      # sigma[i] : (r, r) full, or (r,) diagonal, in adapter i's cluster basis


@dataclass
class JDCompressed:
    """A JD-compressed collection.

    assignment[i] = cluster index of adapter i (shared across slots).
    norm[i]       = original effective Frobenius norm of adapter i (over all slots),
                    folded out before JD and restored at reconstruction time.
    """
    adapter_ids: list
    variant: str                       # "full" | "diag"
    rank: int
    n_clusters: int
    assignment: list
    norm: torch.Tensor                 # (n,)
    slots: dict = field(default_factory=dict)   # slot_name -> JDSlot
    recon_err: torch.Tensor = None     # (n,) mean relative recon error over slots

    # -- reconstruction / serving -------------------------------------------------

    def reconstruct_delta(self, slot_name, i):
        """Dense effective delta D_hat_i = norm_i * U_j Sigma_i V_j^T for one slot.

        Materializes d_out x d_in — use only for tests / small slots.
        """
        js = self.slots[slot_name]
        j = self.assignment[i]
        U, V = js.U[j], js.V[j]
        sig = js.sigma[i]
        S = torch.diag(sig) if sig.ndim == 1 else sig
        return self.norm[i] * (U @ S @ V.t())

    def merge_keepset(self, keep, weights=None, out_rank=None):
        """Combine the kept adapters into ONE low-rank delta per slot.

        Merged delta D_S = sum_j U_j (sum_{i in keep, cluster_j} w_i norm_i Sigma_i) V_j^T,
        returned factored as (A_new, B_new) with B_new @ A_new ~= D_S at rank out_rank.

        Deleting an adapter from `keep` only changes the inner cluster sum by w_i norm_i
        Sigma_i — an O(1) update of a small r x r matrix, no refit (the paper's cheap
        deletion). `keep` is a list of adapter indices.
        """
        keep = list(keep)
        if weights is None:
            weights = [1.0 / len(keep)] * len(keep)
        wmap = {i: w for i, w in zip(keep, weights)}
        out = {}
        for slot_name, js in self.slots.items():
            d_out = js.U[0].shape[0]
            d_in = js.V[0].shape[0]
            dev, dt = js.U[0].device, js.U[0].dtype
            # Per-cluster accumulated Sigma-bar, then stack the cluster blocks.
            B_blocks, A_blocks = [], []
            for j in range(self.n_clusters):
                members = [i for i in keep if self.assignment[i] == j]
                if not members:
                    continue
                r = js.U[j].shape[1]
                sig_bar = torch.zeros(r, r, device=dev, dtype=dt)
                for i in members:
                    sig = js.sigma[i]
                    S = torch.diag(sig) if sig.ndim == 1 else sig
                    sig_bar += (wmap[i] * float(self.norm[i])) * S
                # U_j Sigma_bar V_j^T  -> factored block (U_j Sigma_bar) @ (V_j^T)
                B_blocks.append(js.U[j] @ sig_bar)     # (d_out, r)
                A_blocks.append(js.V[j].t())           # (r, d_in)
            if not B_blocks:
                out[slot_name] = (torch.zeros(0, d_in, device=dev, dtype=dt),
                                  torch.zeros(d_out, 0, device=dev, dtype=dt))
                continue
            B_cat = torch.cat(B_blocks, dim=1)         # (d_out, sum_j r)
            A_cat = torch.cat(A_blocks, dim=0)         # (sum_j r, d_in)
            r_target = out_rank or B_cat.shape[1]
            A_new, B_new = _compress_factored(B_cat, A_cat, r_target)
            out[slot_name] = (A_new, B_new)
        return out

    def reconstruction_error(self):
        """Mean over adapters of the relative Frobenius reconstruction error
        (cached at fit time; the paper's recon-error metric, lower is better)."""
        return self.recon_err.mean().item()


# ---------------------------------------------------------------------------
# Factored linear-algebra helpers (mirror merge_extra conventions)
# ---------------------------------------------------------------------------

def _compress_factored(B_cat, A_cat, rank):
    """SVD-compress delta = B_cat @ A_cat to `rank` without forming the dense delta.

    Same algorithm as merge_extra._compress_factored: QR both stacks, SVD the small
    inner core. Returns (A_new, B_new) of rank min(rank, inner-dim).
    """
    Qb, Rb = torch.linalg.qr(B_cat)
    Qa, Ra = torch.linalg.qr(A_cat.t())
    core = Rb @ Ra.t()
    U, S, Vh = torch.linalg.svd(core)
    r = min(rank, S.shape[0])
    B_new = Qb @ (U[:, :r] * S[:r])
    A_new = Vh[:r] @ Qa.t()
    return A_new, B_new


def _top_r_subspace_from_blocks(blocks, r, d):
    """Top-r left-singular subspace of the stacked [blocks...] (each block is (d, c_i)).

    Returns U (d, r) spanning the dominant column subspace of X = [X_1 | ... | X_n].
    Two regimes, picked by total column count m = sum_i c_i:
      - m <= d : thin SVD of X directly (O(d m^2)) -- far cheaper for small/moderate n.
      - m  > d : accumulate the d x d Gram sum_i X_i X_i^T and SVD it (O(d^2) memory,
                 constant in n) -- the scale path for thousands of adapters.
    SVD (not eigh) is used throughout: some MKL builds hit an SSYEVD workspace bug on
    large symmetric matrices.
    """
    m = sum(X.shape[1] for X in blocks)
    if m <= d:
        X = torch.cat([b.float() for b in blocks], dim=1)   # (d, m)
        U, _, _ = torch.linalg.svd(X, full_matrices=False)
        U = U[:, :r]
    else:
        dev = blocks[0].device
        M = torch.zeros(d, d, device=dev, dtype=torch.float32)
        for X in blocks:
            M += X.float() @ X.float().t()
        U, _, _ = torch.linalg.svd(M)                       # U == eigenvectors, S desc
        U = U[:, :r]
    # Always return exactly r columns. When the data subspace has rank < r (e.g. a
    # single low-rank adapter with a larger requested JD rank), pad with zero columns:
    # they are inert (zero Sigma rows, no effect on reconstruction or the merge) and keep
    # the basis shape uniform across clusters/slots for storage and the diag solver.
    if U.shape[1] < r:
        U = torch.cat([U, torch.zeros(d, r - U.shape[1], device=U.device, dtype=U.dtype)], dim=1)
    return U


def _residual_sq(U, V, sigma, B, A):
    """True ||D - U Sigma V^T||_F^2 / ||D||^2 for the factored delta D = B @ A.

    Valid for any (not necessarily orthonormal) U, V and full or diagonal Sigma:
      ||D||^2 - 2<D, U S V^T> + ||U S V^T||^2, all evaluated in the r x r space via
      M = U^T D V = (U^T B)(A V). Returns the *relative* squared error.
    """
    S = torch.diag(sigma) if sigma.ndim == 1 else sigma
    M = (U.t() @ B) @ (A @ V)                              # U^T D V  (r, r)
    d_sq = torch.trace((B.t() @ B) @ (A @ A.t())).item()   # ||D||^2
    cross = (S * M).sum().item()                           # <D, U S V^T> = tr(S^T M)
    recon = torch.trace((U.t() @ U) @ S @ (V.t() @ V) @ S.t()).item()  # ||U S V^T||^2
    return max(d_sq - 2.0 * cross + recon, 0.0) / max(d_sq, 1e-12)


# ---------------------------------------------------------------------------
# Single-slot JD optimizers (operate on unit-normalized effective deltas)
# ---------------------------------------------------------------------------

def _effective_blocks(Bs, As, scals, norms, idxs, device=None):
    """Yield (Beff_i, A_i) for the given adapter indices, with scaling and 1/norm folded
    into Beff so that Beff_i @ A_i is the unit-normalized effective delta.

    `device` (e.g. "cuda") moves just these blocks onto the compute device on demand, so
    the factors can live on CPU and only one module's worth is GPU-resident at a time —
    this keeps the build's peak GPU memory ~flat in the number of adapters (the scale win;
    otherwise all n adapters' factors would sit on the GPU and OOM for large collections)."""
    out = []
    for i in idxs:
        scale = scals[i] / max(float(norms[i]), 1e-12)
        B = scale * Bs[i].float()
        A = As[i].float()
        if device is not None:
            B = B.to(device)
            A = A.to(device)
        out.append((B, A))
    return out


def jd_full_slot(blocks, rank, iters=10):
    """JD-Full for one slot. blocks = [(Beff_i, A_i)]. Returns (U, V, [Sigma_i]).

    Alternating dominant-subspace iteration (paper Appendix A.1, Case 1):
      Sigma_i = U^T D_i V ;  U <- top-r of sum_i (D_i V)(D_i V)^T ;
      V <- top-r of sum_i (D_i^T U)(D_i^T U)^T.  All kept factored.
    """
    d_out = blocks[0][0].shape[0]
    d_in = blocks[0][1].shape[1]
    r = min(rank, d_out, d_in)
    # Init U from the dominant left subspace of the stacked Beff (deterministic).
    U = _top_r_subspace_from_blocks([B for B, _ in blocks], r, d_out)
    for _ in range(iters):
        # V-step: Y_i = D_i^T U = A_i^T (Beff_i^T U)   (d_in, r)
        Y = [A.t() @ (B.t() @ U) for B, A in blocks]
        V = _top_r_subspace_from_blocks(Y, r, d_in)
        # U-step: X_i = D_i V = Beff_i (A_i V)          (d_out, r)
        X = [B @ (A @ V) for B, A in blocks]
        U = _top_r_subspace_from_blocks(X, r, d_out)
    sigmas = [(U.t() @ B) @ (A @ V) for B, A in blocks]   # (r, r) each
    return U, V, sigmas


def jd_diag_slot(blocks, rank, iters=10, eps=1e-6):
    """JD-Diag for one slot. Returns (U, V, [diag_i]) with diag_i length r.

    Coordinate descent (paper Appendix A.1, Case 2): U, V via the small r x r linear
    systems, diagonal Sigma_i via the Hadamard closed form. Initialized from the
    JD-Full basis.
    """
    d_out = blocks[0][0].shape[0]
    d_in = blocks[0][1].shape[1]
    U, V, _ = jd_full_slot(blocks, min(rank, d_out, d_in), iters=iters)
    r = U.shape[1]                       # actual basis width (uniform, possibly padded)
    dev = U.device

    def diag_sigmas(U, V):
        UtU, VtV = U.t() @ U, V.t() @ V
        G = UtU * VtV                                   # Hadamard (r, r)
        Ginv = torch.linalg.pinv(G)
        out = []
        for B, A in blocks:
            # (U^T Beff_i  Hadamard  V^T A_i^T) summed over the inner rank -> (r,)
            P = U.t() @ B                               # (r, r_i)
            Q = V.t() @ A.t()                           # (r, r_i)
            rhs = (P * Q).sum(dim=1)                    # (r,)
            out.append(Ginv @ rhs)
        return out

    diags = diag_sigmas(U, V)
    for _ in range(iters):
        # U = (sum_i D_iV Sigma_i)(sum_i Sigma_i V^TV Sigma_i)^-1
        VtV = V.t() @ V
        U_num = torch.zeros(d_out, r, device=dev, dtype=torch.float32)
        U_den = torch.zeros(r, r, device=dev, dtype=torch.float32)
        for (B, A), s in zip(blocks, diags):
            DV = B @ (A @ V)                            # (d_out, r)
            U_num += DV * s.unsqueeze(0)                # scale columns by diag
            U_den += VtV * (s.unsqueeze(1) * s.unsqueeze(0))
        U = U_num @ torch.linalg.pinv(U_den + eps * torch.eye(r, device=dev))
        # V = (sum_i D_i^TU Sigma_i)(sum_i Sigma_i U^TU Sigma_i)^-1
        UtU = U.t() @ U
        V_num = torch.zeros(d_in, r, device=dev, dtype=torch.float32)
        V_den = torch.zeros(r, r, device=dev, dtype=torch.float32)
        for (B, A), s in zip(blocks, diags):
            DtU = A.t() @ (B.t() @ U)                   # (d_in, r)
            V_num += DtU * s.unsqueeze(0)
            V_den += UtU * (s.unsqueeze(1) * s.unsqueeze(0))
        V = V_num @ torch.linalg.pinv(V_den + eps * torch.eye(r, device=dev))
        diags = diag_sigmas(U, V)
    return U, V, diags


def _optimal_sigma(U, V, B, A, variant):
    """Optimal Sigma for a fixed basis (used at cluster reassignment)."""
    if variant == "full":
        return (U.t() @ B) @ (A @ V)
    UtU, VtV = U.t() @ U, V.t() @ V
    Ginv = torch.linalg.pinv(UtU * VtV)
    P, Q = U.t() @ B, V.t() @ A.t()
    return Ginv @ (P * Q).sum(dim=1)


# ---------------------------------------------------------------------------
# Norms, k-means, and the top-level collection compressor
# ---------------------------------------------------------------------------

def compute_norms(slots: dict):
    """Per-adapter effective Frobenius norm over all slots (one scalar per adapter).

    ||scaling_i B_i A_i||_F^2 = scaling_i^2 * tr((B_i^T B_i)(A_i A_i^T)) -- r x r traces,
    no dense delta. Returns a (n,) tensor.
    """
    any_slot = next(iter(slots.values()))
    n = len(any_slot.B)
    sq = torch.zeros(n, dtype=torch.float64)
    for slot in slots.values():
        for i in range(n):
            B = slot.B[i].float()
            A = slot.A[i].float()
            t = torch.trace((B.t() @ B) @ (A @ A.t())).item()
            sq[i] += (slot.scaling[i] ** 2) * max(t, 0.0)
    return sq.sqrt().to(torch.float32)


def _kmeans(X, k, seed, iters=50):
    """Tiny Lloyd's k-means with k-means++ init. X: (n, d). Returns assignment (n,)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    if k >= n:
        return torch.arange(n) % k
    # k-means++ seeding
    centers = [X[torch.randint(n, (1,), generator=g).item()]]
    for _ in range(1, k):
        d2 = torch.stack([((X - c) ** 2).sum(1) for c in centers], 0).min(0).values
        probs = d2 / d2.sum().clamp_min(1e-12)
        nxt = torch.multinomial(probs, 1, generator=g).item()
        centers.append(X[nxt])
    C = torch.stack(centers, 0)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        dist = torch.cdist(X, C)
        new = dist.argmin(1)
        if torch.equal(new, assign):
            assign = new
            break
        assign = new
        for j in range(k):
            mask = assign == j
            if mask.any():
                C[j] = X[mask].mean(0)
    return assign


def jd_compress_collection(slots: dict, adapter_ids, *, variant="full", clusters=1,
                           rank=16, iters=10, seed=0, max_rounds=10, compute_device=None):
    """Compress a collection (dict slot_name -> Slot) with JD (+ clustering).

    Returns a JDCompressed. `variant` in {"full","diag"}; `clusters` >= 1. With clustering,
    alternates per-cluster JD and cluster reassignment until the assignment stabilizes
    (Appendix A.3), capped at `max_rounds`. `compute_device` (e.g. "cuda") streams each
    module's factors onto the GPU on demand while the Slot tensors stay on CPU, so peak GPU
    memory is ~flat in the adapter count.
    """
    n = len(adapter_ids)
    norms = compute_norms(slots)
    opt = jd_full_slot if variant == "full" else jd_diag_slot

    def fit_basis(slot, idxs):
        blocks = _effective_blocks(slot.B, slot.A, slot.scaling, norms, idxs, device=compute_device)
        return opt(blocks, rank, iters=iters)

    # ---- assignment ----
    if clusters <= 1:
        assignment = [0] * n
    else:
        # A.3 initialization: single-basis JD, then k-means on the per-adapter Sigmas
        # (flattened and concatenated across slots).
        feats = torch.zeros(n, 0)
        for slot in slots.values():
            U, V, sig = fit_basis(slot, range(n))
            flat = torch.stack([
                (torch.diag(s) if s.ndim == 1 else s).reshape(-1) for s in sig
            ], 0).detach().cpu()                       # k-means runs on CPU
            feats = torch.cat([feats, flat], dim=1)
        # Cluster by Sigma *direction*: per-adapter feature magnitudes vary ~8x, so raw
        # Euclidean k-means makes large-norm adapters singleton centers and collapses the
        # rest into one cluster (observed [94,1,1,1,1,1,1] at k=100). L2-normalizing the
        # feature rows yields balanced, task-similarity clusters.
        feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-8)
        assignment = _kmeans(feats, clusters, seed).tolist()

        def reassign(assignment):
            """A.3 Step 1 (per-cluster JD) -> Step 2 (reassign by min recon error)."""
            bases = {j: {name: fit_basis(slot, [i for i in range(n) if assignment[i] == j])
                         for name, slot in slots.items()}
                     for j in set(assignment)}
            new = list(assignment)
            for i in range(n):
                best_j, best_cost = assignment[i], float("inf")
                for j, per_slot in bases.items():
                    cost = 0.0
                    for name, slot in slots.items():
                        U, V, _ = per_slot[name]
                        B, A = _effective_blocks(slot.B, slot.A, slot.scaling, norms, [i],
                                                 device=compute_device)[0]
                        cost += _residual_sq(U, V, _optimal_sigma(U, V, B, A, variant), B, A)
                    if cost < best_cost:
                        best_cost, best_j = cost, j
                new[i] = best_j
            return new

        # Alternate until no assignment changes (A.3 convergence), capped at max_rounds.
        for _ in range(max_rounds):
            new_assign = reassign(assignment)
            if new_assign == assignment:
                break
            assignment = new_assign

    n_clusters = max(assignment) + 1 if clusters > 1 else 1

    # ---- final per-cluster bases + per-adapter Sigma + recon error ----
    out_slots = {}
    recon_acc = torch.zeros(n)
    for name, slot in slots.items():
        U_list, V_list = [None] * n_clusters, [None] * n_clusters
        sigmas = [None] * n
        for j in range(n_clusters):
            idxs = [i for i in range(n) if assignment[i] == j]
            if not idxs:
                # empty cluster: dummy zero basis so indices stay aligned (match
                # compute_device so save_jd's torch.stack doesn't mix CPU/GPU tensors)
                d_out, d_in = slot.B[0].shape[0], slot.A[0].shape[1]
                r = min(rank, d_out, d_in)
                U_list[j] = torch.zeros(d_out, r, device=compute_device)
                V_list[j] = torch.zeros(d_in, r, device=compute_device)
                continue
            U, V, sig = fit_basis(slot, idxs)
            U_list[j], V_list[j] = U, V
            for i, s in zip(idxs, sig):
                sigmas[i] = s
        out_slots[name] = JDSlot(U=U_list, V=V_list, sigma=sigmas)
        blocks = _effective_blocks(slot.B, slot.A, slot.scaling, norms, range(n), device=compute_device)
        for i in range(n):
            j = assignment[i]
            B, A = blocks[i]
            recon_acc[i] += math.sqrt(_residual_sq(U_list[j], V_list[j], sigmas[i], B, A))
    recon_acc /= max(len(slots), 1)

    return JDCompressed(
        adapter_ids=list(adapter_ids), variant=variant, rank=rank,
        n_clusters=n_clusters, assignment=assignment, norm=norms, slots=out_slots,
        recon_err=recon_acc,
    )


def select_num_clusters(slots, adapter_ids, *, variant="full", rank=16,
                        candidates=(1, 2, 4, 7, 8, 10, 16, 25), threshold=0.6, seed=0,
                        probe_slot=None):
    """Paper Section 6.5 heuristic: smallest cluster count whose reconstruction error
    falls below `threshold`, measured on a single probe slot (CPU-cheap, no LLM eval).

    `probe_slot` defaults to the middle slot. Returns (chosen_c, {c: recon_error}).
    """
    names = list(slots.keys())
    probe = probe_slot or names[len(names) // 2]
    sub = {probe: slots[probe]}
    errs = {}
    for c in candidates:
        jd = jd_compress_collection(sub, adapter_ids, variant=variant,
                                    clusters=c, rank=rank, seed=seed)
        errs[c] = jd.reconstruction_error()
    chosen = next((c for c in candidates if errs[c] < threshold), candidates[-1])
    return chosen, errs


def recommend_jd_settings(n, slots=None, adapter_ids=None, *, threshold=0.6, seed=0):
    """Paper-recommended JD settings for a collection of `n` LoRAs (Sections 6.5 / F).

    - n <= 100 : JD-Full, no clustering, rank ~ (n/2)+7 (the paper's no-cluster rule).
    - n  > 100 : JD-Full + clustering at rank 16; if `slots`/`adapter_ids` are given,
                 choose the cluster count by the <threshold reconstruction-error sweep on
                 the middle module (`select_num_clusters`); otherwise default to 25
                 (the paper's setting for 512-1024 LoRAs).

    Returns dict {variant, clusters, rank}.
    """
    if n <= 100:
        return {"variant": "full", "clusters": 1, "rank": max(1, n // 2 + 7)}
    clusters = 25
    if slots is not None and adapter_ids is not None:
        clusters, _ = select_num_clusters(slots, adapter_ids, variant="full", rank=16,
                                          threshold=threshold, seed=seed)
    return {"variant": "full", "clusters": clusters, "rank": 16}
