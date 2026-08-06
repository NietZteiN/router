"""Post-hoc deterministic sparsification grid + DX diagnostics (composable_tv Wave 0, [w5]).

Zero-training preview of the ctv training arms on the existing Llama-2-7B e5 per-author r32
pool (config configs/sparsify_7b.json): sparsify each author's LoRA factors with a
deterministic, exactness-compatible op, compose author subsets (nested subset_authors(42,N))
via merge_subset._weighted_factor_cat, and materialize every merge with
write_effective_adapter so eval_tofu --preloaded_adapter can serve it (never pays the fp32
high-k eval law).

Exactness contract: every op uses per-task-LOCAL data or DATA-INDEPENDENT randomness only,
so deletion stays exact — drop author j = recompute the merge without j (dare/topk masks are
pure functions of the author's own files + SHA-256 seeds; the hash partition is a global
seeded permutation that never looks at any data).

Ops (config `ops` strings; labels sparse_<op>_N<n>_s<seed>, e.g. sparse_dare0p9_N8_s42):
  dare<p>   Bernoulli drop at rate p on B AND A entries with per-(author,tensor) SHA-256
            seeds; survivors rescaled 1/(1-p) (unbiased per factor). The seed key excludes
            p on purpose: dare0p5/dare0p9 share one uniform field per tensor, so the masks
            are NESTED (keep@0.9 ⊂ keep@0.5) — p is one dial, not two independent draws.
  topk<q>   keep the ceil(q*d_out) largest-norm ROWS of B, zero the rest (per-task-local
            magnitude information — allowed under exactness). A untouched.
  hash      seeded GLOBAL permutation of the d_out rows of B partitioned into N contiguous
            blocks; the subset's j-th author keeps ONLY block j's rows — the zero-training
            preview of the [wd] rowslice arm. Deliberately NOT the memsinks_tofu/masks.py
            hash functions (known int64-overflow artifact); seeded permutation blocks are
            overflow-free and disjoint+covering by construction.

  <op>sum   (dare<p>sum / topk<q>sum, 2026-07-20) SAME masks as the non-sum twin — the
            sparse adapter dirs are shared via the canonical op name — but composed at
            weight 1.0 each instead of 1/N: doc-1 method 7 (DARE + naive sum, separable ⇒
            exact drop-a-term deletion w.r.t. the DARE'd deltas). Pre-registered in
            log/merge_mechanism/2026-07-18_gapfill-preregistration.md.

Compose weights: 1/N for dare/topk (comparable to the additive_mean reference curve),
1.0 for hash (disjoint row support ⇒ the sum has no elementwise interference) and for
the <op>sum family (weight-1.0 sum on overlapping supports — the interference condition).

DX diagnostics (same script; slot-streaming — outer loop over slots, lazy safetensors reads
of just that slot's factors — so peak memory is a handful of dense slot deltas, ~180 MB fp32
each at 7B):
  --dx1  per-coordinate cancellation |Σᵢdᵢ| / Σᵢ|dᵢ| at N in dx1.n_values vs a sign-shuffled
         null (same magnitudes, seeded random signs, dx1.null_draws draws)
         → reports/ctv_dx1_cancellation.json. Interpretation contract: observed <= null
         means elementwise sign-fixing has no headroom (closes the W3 idea-space).
  --dx2  fraction of each author delta's Frobenius energy inside its own hash row block at
         each grid N (expected ≈1/N for unconstrained adapters), via r x r Grams (never
         dense) → reports/ctv_dx2_energy.json.

CLI (CPU only; the real 7B pass is a CPU SLURM job — never the login node):
  python sparsify_pool.py --config configs/sparsify_7b.json --dry_run
  python sparsify_pool.py --config configs/sparsify_7b.json [--ops dare0p9 hash]
                          [--n_ladder 2 4] [--limit_authors 4] [--force]
  python sparsify_pool.py --config configs/sparsify_7b.json --dx1 --dx2 [--report_dir D]

Eval manifest: {out_dir}/eval_manifest_sparse.txt in the 4-column nmerge format
(label \t adapter \t sid \t retain_ids='-') so submit_nmerge-style eval arrays consume it
unchanged. Per merge label: one own-recall row per probe author (probes ∩ subset(N)); the
FIRST probe row (perm[0] = author 82) doubles as the headline-mu row, exactly the nmerge /
analyze_nmerge convention — no separate headline line is emitted. iso_a<author> rows point
at the raw pool adapters.

CPU gate (run before any SLURM job): python test_sparsify_pool.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from jd_collection import _adapter_scaling, _read_adapter, _PREFIX
from merge_subset import (
    N_AUTHORS,
    author_permutation,
    probe_authors,
    subset_authors,
    write_effective_adapter,
    _weighted_factor_cat,
)

import sys

# ── site-path expansion (added on export) ────────────────────────────────────────────────────
# Configs used to carry absolute /storage2 paths. They now say "${TOFU_CKPT_ROOT}/..." etc, and
# this resolves them at load time, hard-erroring on an unset variable rather than writing a
# literal "${TOFU_CKPT_ROOT}" directory to disk (which is what happened before the guard).
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import expand_paths as _expand_site_paths, ensure_site_env as _ensure_site_env
except ImportError:                       # repo_env.py is at the repo root; absent => no-op
    def _expand_site_paths(o, _k=""): return o
    def _ensure_site_env(force=False): return {}

# Polite CPU cap on shared nodes; SLURM jobs get --cpus-per-task to match.
torch.set_num_threads(int(os.environ.get("SPARSIFY_THREADS", min(32, os.cpu_count() or 1))))

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Config + determinism helpers
# ---------------------------------------------------------------------------

def load_config(path):
    _ensure_site_env()
    with open(path) as f:
        cfg = _expand_site_paths(json.load(f))
    for key in ("pool_dir", "out_dir", "ops", "n_ladder", "pool_seed", "eval"):
        if key not in cfg:
            raise KeyError(f"config missing {key!r}")
    return cfg


def _resolve(path):
    """Config paths may be repo-relative (checkpoints/ is the /storage2 symlink)."""
    return path if os.path.isabs(path) else os.path.join(_REPO_DIR, path)


def _sha_seed(*parts, bits=63):
    """SHA-256-derived integer seed from string parts. Never python hash() (salted) and
    never the memsinks_tofu/masks.py scheme (int64-overflow artifact)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h, 16) % (2 ** bits)


def _script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def sparse_label(op, n, seed):
    return f"sparse_{op}_N{n}_s{seed}"


_OP_RE = re.compile(r"^(dare|topk)(\d+p\d+)(sum)?$")


def parse_op(op):
    """'dare0p9' -> ('dare', 0.9); 'topk0p25' -> ('topk', 0.25); 'hash' -> ('hash', None);
    'dare0p9sum' -> ('daresum', 0.9) — same masks as dare0p9 (seeds are sum-agnostic),
    composed at weight 1.0 each instead of 1/N (doc-1 method 7, DARE+naive-sum: separable,
    exact drop-a-term deletion w.r.t. the DARE'd deltas)."""
    if op == "hash":
        return "hash", None
    m = _OP_RE.match(op)
    if m is None:
        raise ValueError(f"unknown sparsify op {op!r} (want dare<p>[sum]/topk<q>[sum]/hash)")
    return m.group(1) + (m.group(3) or ""), float(m.group(2).replace("p", "."))


def _base_kind(kind):
    """'daresum' -> 'dare' (sum suffix only changes compose weights, never the masks)."""
    return kind[:-3] if kind.endswith("sum") else kind


def _canonical_op(op):
    """Adapter-dir name for an op: 'dare0p9sum' reuses 'dare0p9' sparse adapters
    (bit-identical masks — the sum variant differs only at compose time)."""
    kind, _ = parse_op(op)
    return op[:-3] if kind.endswith("sum") else op


# ---------------------------------------------------------------------------
# Sparsify ops (raw factor space; dare/topk masks are scale-invariant, so raw
# vs scaled factors is immaterial — the source adapter_config carries scaling)
# ---------------------------------------------------------------------------

def dare_mask(tensor, p, gen):
    """Bernoulli-drop entries at rate p; survivors rescaled 1/(1-p) (unbiased per tensor)."""
    keep = torch.rand(tensor.shape, generator=gen) >= p
    return tensor * keep.to(tensor.dtype) / (1.0 - p)


def topk_row_mask(B, q):
    """Keep the ceil(q * d_out) largest-norm rows of B, zero the rest (no rescale — the
    kept rows are served verbatim; rescaling top-k rows would bias the delta they carry)."""
    k = math.ceil(q * B.shape[0])
    idx = torch.topk(B.norm(dim=1), k).indices
    out = torch.zeros_like(B)
    out[idx] = B[idx]
    return out


def hash_row_blocks(d_out, n, pool_seed, slot):
    """N disjoint+covering row blocks from a seeded GLOBAL permutation of range(d_out).

    The permutation is per-(pool_seed, slot) — never per-author — so every author of a
    subset sees the same partition and block j is author j's slice (subset position).
    Contiguous integer boundaries (j*d_out)//n partition exactly (sizes differ by <=1).
    """
    if n > d_out:
        raise ValueError(f"hash: n={n} blocks > d_out={d_out} rows for slot {slot!r}")
    perm = np.random.RandomState(_sha_seed("ctv_hash", pool_seed, slot, bits=32)).permutation(d_out)
    return [np.sort(perm[(j * d_out) // n:((j + 1) * d_out) // n]) for j in range(n)]


def _keep_rows(B, rows):
    out = torch.zeros_like(B)
    idx = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    out[idx] = B[idx]
    return out


def sparsify_author(pool_dir, author, op, out_dir, pool_seed, block_rows=None):
    """Write ONE sparsified per-author adapter dir (source layout + config preserved, so
    _weighted_factor_cat applies the original rslora scaling to the sparsified factors)."""
    src = os.path.join(pool_dir, f"shard_{author}")
    slots, cfg = _read_adapter(src)
    kind, param = parse_op(op)
    kind = _base_kind(kind)  # sum variants sparsify identically to their mean twin
    tensors = {}
    for slot, (A, B) in slots.items():
        if kind == "dare":
            # per-(author,tensor) seeds; key excludes p => nested masks across rates
            gen_b = torch.Generator().manual_seed(_sha_seed("ctv_dare", pool_seed, author, slot, "B"))
            gen_a = torch.Generator().manual_seed(_sha_seed("ctv_dare", pool_seed, author, slot, "A"))
            B_sp, A_sp = dare_mask(B, param, gen_b), dare_mask(A, param, gen_a)
        elif kind == "topk":
            B_sp, A_sp = topk_row_mask(B, param), A
        elif kind == "hash":
            B_sp, A_sp = _keep_rows(B, block_rows[slot]), A
        else:  # pragma: no cover — parse_op already validated
            raise ValueError(kind)
        key = _PREFIX + slot
        tensors[key + ".lora_A.weight"] = A_sp.contiguous()
        tensors[key + ".lora_B.weight"] = B_sp.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return out_dir


# ---------------------------------------------------------------------------
# Grid: sparsify -> compose -> materialize -> eval manifest
# ---------------------------------------------------------------------------

def _plan_manifest(cfg, ops, ladder, pool_dir, out_dir, probes):
    """All eval rows as (label, adapter_dir, sid). iso rows first; per merge label one row
    per probe member of subset(N) — the FIRST (perm[0]) is the headline row (nmerge rule)."""
    seed = int(cfg["pool_seed"])
    n_probes = int(cfg.get("n_probes", 5))
    lines = [(f"iso_a{a}", os.path.join(pool_dir, f"shard_{a}"), a) for a in probes]
    for op in ops:
        for n in ladder:
            label = sparse_label(op, n, seed)
            adapter = os.path.join(out_dir, "merges", label)
            for a in probe_authors(seed, n, n_probes):
                lines.append((label, adapter, a))
    return lines


def run_grid(cfg, cfg_path, ops=None, n_ladder=None, limit_authors=None,
             dry_run=False, force=False):
    seed = int(cfg["pool_seed"])
    n_probes = int(cfg.get("n_probes", 5))
    ops = list(ops if ops is not None else cfg["ops"])
    for op in ops:
        parse_op(op)  # fail fast on typos before touching any files
    ladder = sorted({int(n) for n in (n_ladder if n_ladder is not None else cfg["n_ladder"])})
    if limit_authors is not None:
        dropped = [n for n in ladder if n > limit_authors]
        ladder = [n for n in ladder if n <= limit_authors]
        if dropped:
            print(f"[sparsify] --limit_authors {limit_authors}: dropped N {dropped}")
    if not ladder:
        raise SystemExit("empty n_ladder (after --limit_authors)")
    if "probe_authors" in cfg:  # config carries probes as documentation — runtime derives
        derived = probe_authors(seed, N_AUTHORS, n_probes)
        if list(cfg["probe_authors"]) != derived:
            raise ValueError(f"config probe_authors {cfg['probe_authors']} != derived {derived}")

    pool_dir, out_dir = _resolve(cfg["pool_dir"]), _resolve(cfg["out_dir"])
    max_n = max(ladder)
    authors_max = subset_authors(seed, max_n)
    probes = probe_authors(seed, max_n, n_probes)
    lines = _plan_manifest(cfg, ops, ladder, pool_dir, out_dir, probes)

    print(f"[sparsify] ops={ops} n_ladder={ladder} pool_seed={seed} "
          f"({len(ops) * len(ladder)} merges, {len(lines)} eval rows) script_sha={_script_sha()}")
    if dry_run:
        for op in ops:
            for n in ladder:
                print(f"[sparsify]   plan {sparse_label(op, n, seed)}: "
                      f"authors={subset_authors(seed, n)}")
        print(f"[sparsify] dry_run: no files read or written "
              f"(would write {os.path.join(out_dir, 'eval_manifest_sparse.txt')})")
        return lines

    os.makedirs(os.path.join(out_dir, "merges"), exist_ok=True)
    cap = cfg.get("eval", {}).get("cap", "smoke")
    os.makedirs(os.path.join(out_dir, "results", cap), exist_ok=True)

    # KS reference so forget_quality isn't NaN (same convention as merge_subset.do_plan).
    ref = cfg.get("retain_tr_source")
    if ref:
        ref = _resolve(ref)
        dst = os.path.join(out_dir, "results", cap, "retain_tr_scores.npy")
        if os.path.exists(ref) and not os.path.exists(dst):
            shutil.copy2(ref, dst)
            print(f"[sparsify] copied KS ref -> {dst}")
        elif not os.path.exists(ref):
            print(f"[sparsify] WARN missing retain_tr_source {ref} (forget_quality will be NaN)")

    for a in authors_max:
        if not os.path.isdir(os.path.join(pool_dir, f"shard_{a}")):
            raise FileNotFoundError(os.path.join(pool_dir, f"shard_{a}"))

    # dare/topk masks are N-independent (nested subsets reuse them across the ladder);
    # hash blocks depend on N, so those adapters live in per-N dirs.
    shared_dirs = {}
    for op in ops:
        kind, _ = parse_op(op)
        if kind == "hash":
            continue
        shared_dirs[op] = {}
        for a in authors_max:
            # sum variants reuse the mean twin's adapter dir (bit-identical masks)
            ad = os.path.join(out_dir, "sparse_adapters", _canonical_op(op), f"shard_{a}")
            if force or not os.path.exists(os.path.join(ad, "adapter_model.safetensors")):
                sparsify_author(pool_dir, a, op, ad, seed)
            shared_dirs[op][a] = ad

    for op in ops:
        kind, _ = parse_op(op)
        for n in ladder:
            label = sparse_label(op, n, seed)
            merge_dir = os.path.join(out_dir, "merges", label)
            if os.path.exists(os.path.join(merge_dir, "adapter_model.safetensors")) and not force:
                print(f"[sparsify] skip existing {merge_dir}")
                continue
            authors = subset_authors(seed, n)
            if kind == "hash":
                slots0, _ = _read_adapter(os.path.join(pool_dir, f"shard_{authors[0]}"))
                blocks = {slot: hash_row_blocks(B.shape[0], n, seed, slot)
                          for slot, (A, B) in slots0.items()}
                dirs = []
                for j, a in enumerate(authors):
                    ad = os.path.join(out_dir, "sparse_adapters", f"hash_N{n}", f"shard_{a}")
                    if force or not os.path.exists(os.path.join(ad, "adapter_model.safetensors")):
                        sparsify_author(pool_dir, a, op, ad, seed,
                                        block_rows={s: blocks[s][j] for s in blocks})
                    dirs.append(ad)
                weights, weights_desc = [1.0] * n, "1.0 each (disjoint row sum)"
            elif kind.endswith("sum"):
                dirs = [shared_dirs[op][a] for a in authors]
                weights, weights_desc = [1.0] * n, "1.0 each (sum-composed, doc-1 DARE+sum)"
            else:
                dirs = [shared_dirs[op][a] for a in authors]
                weights, weights_desc = [1.0 / n] * n, f"1/{n} uniform (additive_mean-matched)"
            merged, ref_cfg, out_rank, meta = _weighted_factor_cat(dirs, weights)
            write_effective_adapter(merge_dir, merged, ref_cfg, out_rank)
            with open(os.path.join(merge_dir, "merge_meta.json"), "w") as f:
                json.dump({"label": label, "op": op, "n": n, "pool_seed": seed,
                           "authors": authors, "weights": weights_desc,
                           "pool_dir": pool_dir, "sparse_dirs": dirs,
                           "sum_rank": meta["sum_rank"], "out_rank": out_rank,
                           "script_sha256": _script_sha(),
                           "config": os.path.abspath(cfg_path)}, f, indent=2)
            print(f"[sparsify] wrote {merge_dir} (rank {out_rank}, weights {weights_desc})")

    manifest = os.path.join(out_dir, "eval_manifest_sparse.txt")
    with open(manifest, "w") as f:
        for label, adapter, sid in lines:
            f.write(f"{label}\t{adapter}\t{sid}\t-\n")  # 4-col nmerge format, no retain_ids
    print(f"[sparsify] {len(lines)} eval rows -> {manifest}")
    return lines


# ---------------------------------------------------------------------------
# Lazy pool reader (slot-streaming: read ONLY the requested slot's two factors)
# ---------------------------------------------------------------------------

class PoolReader:
    """Per-slot lazy factor access over the raw pool via safetensors.safe_open — the
    slot-outer/author-inner DX1 loop never loads a whole adapter file per visit."""

    def __init__(self, pool_dir):
        self.pool_dir = pool_dir
        self._scaling = {}

    def _path(self, author):
        return os.path.join(self.pool_dir, f"shard_{author}")

    def scaling(self, author):
        if author not in self._scaling:
            with open(os.path.join(self._path(author), "adapter_config.json")) as f:
                self._scaling[author] = _adapter_scaling(json.load(f))
        return self._scaling[author]

    def slot_names(self, author):
        with safe_open(os.path.join(self._path(author), "adapter_model.safetensors"),
                       framework="pt") as f:
            keys = list(f.keys())
        slots = set()
        for key in keys:
            name = key[len(_PREFIX):] if key.startswith(_PREFIX) else key
            if name.endswith(".lora_A.weight"):
                slots.add(name[: -len(".lora_A.weight")])
        return sorted(slots)

    def factors(self, author, slot):
        with safe_open(os.path.join(self._path(author), "adapter_model.safetensors"),
                       framework="pt") as f:
            A = f.get_tensor(_PREFIX + slot + ".lora_A.weight").float()
            B = f.get_tensor(_PREFIX + slot + ".lora_B.weight").float()
        return A, B

    def delta(self, author, slot):
        A, B = self.factors(author, slot)
        return self.scaling(author) * (B @ A)


def _slot_type(slot):
    return slot.rsplit(".", 1)[-1]  # e.g. "q_proj", "up_proj"


# ---------------------------------------------------------------------------
# DX1: elementwise cancellation vs a sign-shuffled null
# ---------------------------------------------------------------------------

class Dx1Accumulator:
    """Streaming per-coordinate cancellation state for ONE slot.

    S = Σᵢ dᵢ, T = Σᵢ |dᵢ|, Z_d = Σᵢ εᵢ,d ⊙ |dᵢ| with εᵢ,d ∈ {±1} seeded per
    (pool_seed, slot, author, draw) — the null keeps every magnitude and randomizes only
    the signs, so observed-vs-null isolates sign structure from magnitude structure.
    """

    def __init__(self, shape, pool_seed, slot, null_draws=5):
        self.pool_seed, self.slot, self.null_draws = pool_seed, slot, null_draws
        self.S = torch.zeros(shape)
        self.T = torch.zeros(shape)
        self.Z = [torch.zeros(shape) for _ in range(null_draws)]

    def add(self, author, delta):
        mag = delta.abs()
        self.S += delta
        self.T += mag
        for d in range(self.null_draws):
            gen = torch.Generator().manual_seed(
                _sha_seed("ctv_dx1", self.pool_seed, self.slot, author, d))
            signs = 1.0 - 2.0 * (torch.rand(delta.shape, generator=gen) < 0.5).float()
            self.Z[d] += signs * mag

    def stats(self):
        """Raw sums (mergeable across slots): coordwise ratio sums + L1 numerators/denoms."""
        mask = self.T > 0
        n_coords = int(mask.sum().item())
        T_m = self.T[mask]
        out = {"n_coords": n_coords,
               "obs_ratio_sum": float((self.S.abs()[mask] / T_m).sum().item()),
               "obs_num_l1": float(self.S.abs().sum().item()),
               "den_l1": float(self.T.sum().item()),
               "null_ratio_sum": [], "null_num_l1": []}
        for Z in self.Z:
            out["null_ratio_sum"].append(float((Z.abs()[mask] / T_m).sum().item()))
            out["null_num_l1"].append(float(Z.abs().sum().item()))
        return out

    def summary(self):
        return _finalize_dx1(self.stats())


def _dx1_bucket_add(dst, st):
    if not dst:
        dst.update({k: (list(v) if isinstance(v, list) else v) for k, v in st.items()})
        return
    dst["n_coords"] += st["n_coords"]
    dst["obs_ratio_sum"] += st["obs_ratio_sum"]
    dst["obs_num_l1"] += st["obs_num_l1"]
    dst["den_l1"] += st["den_l1"]
    dst["null_ratio_sum"] = [a + b for a, b in zip(dst["null_ratio_sum"], st["null_ratio_sum"])]
    dst["null_num_l1"] = [a + b for a, b in zip(dst["null_num_l1"], st["null_num_l1"])]


def _finalize_dx1(bucket):
    """coord_mean = mean over coordinates of |S|/T; l1 = Σ|S| / ΣT (energy-style pooled)."""
    nc, den = max(bucket["n_coords"], 1), max(bucket["den_l1"], 1e-30)
    obs = {"coord_mean": bucket["obs_ratio_sum"] / nc, "l1": bucket["obs_num_l1"] / den}
    null_cm = [rs / nc for rs in bucket["null_ratio_sum"]]
    null_l1 = [num / den for num in bucket["null_num_l1"]]
    null = {"coord_mean_per_draw": null_cm, "l1_per_draw": null_l1,
            "coord_mean_mean": float(np.mean(null_cm)), "coord_mean_std": float(np.std(null_cm)),
            "l1_mean": float(np.mean(null_l1)), "l1_std": float(np.std(null_l1))}
    return {"observed": obs, "null": null,
            "observed_le_null_coord_mean": obs["coord_mean"] <= null["coord_mean_mean"],
            "observed_le_null_l1": obs["l1"] <= null["l1_mean"]}


def run_dx1(cfg, n_values, null_draws, out_path):
    """Streaming DX1 over the RAW pool deltas (true scale sᵢBᵢAᵢ), snapshotting the
    accumulators at each prefix count in n_values. Stream order = the seed's permutation
    (+ author 199 last iff 200 is requested — subset(200) is set-equal to perm ∪ {199})."""
    seed = int(cfg["pool_seed"])
    pool_dir = _resolve(cfg["pool_dir"])
    reader = PoolReader(pool_dir)
    n_values = sorted(set(int(n) for n in n_values))
    perm = author_permutation(seed).tolist()
    stream = perm + [N_AUTHORS - 1] if max(n_values) == N_AUTHORS else perm[: max(n_values)]
    for n in n_values:
        if n > len(stream):
            raise ValueError(f"dx1 n={n} > {len(stream)} streamable authors")

    slots = reader.slot_names(stream[0])
    buckets = {}  # (n, slot_type or "overall") -> raw sums
    for si, slot in enumerate(slots):
        A0, B0 = reader.factors(stream[0], slot)
        acc = Dx1Accumulator((B0.shape[0], A0.shape[1]), seed, slot, null_draws)
        for count, author in enumerate(stream, start=1):
            acc.add(author, reader.delta(author, slot))
            if count in n_values:
                st = acc.stats()
                _dx1_bucket_add(buckets.setdefault((count, _slot_type(slot)), {}), st)
                _dx1_bucket_add(buckets.setdefault((count, "overall"), {}), st)
        del acc  # keep peak memory = one slot's accumulators
        print(f"[dx1] slot {si + 1}/{len(slots)} {slot} done")

    per_n = {}
    for n in n_values:
        types = sorted(t for (m, t) in buckets if m == n and t != "overall")
        per_n[str(n)] = {"overall": _finalize_dx1(buckets[(n, "overall")]),
                         "per_slot_type": {t: _finalize_dx1(buckets[(n, t)]) for t in types}}
        ov = per_n[str(n)]["overall"]
        print(f"[dx1] N={n}: observed l1={ov['observed']['l1']:.4f} coord_mean="
              f"{ov['observed']['coord_mean']:.4f} | null l1={ov['null']['l1_mean']:.4f}"
              f"±{ov['null']['l1_std']:.4f} -> observed_le_null_l1={ov['observed_le_null_l1']}")

    out = {"pool_dir": pool_dir, "pool_seed": seed, "n_values": n_values,
           "null_draws": null_draws, "n_slots": len(slots),
           "stream": "author_permutation(seed) prefix (+199 last iff N=200)",
           "contract": "observed <= null => elementwise sign-fixing has no headroom "
                       "(closes the W3 idea-space)",
           "script_sha256": _script_sha(), "per_n": per_n}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[dx1] wrote {out_path}")
    return out


# ---------------------------------------------------------------------------
# DX2: energy in owned rows under the hash partition
# ---------------------------------------------------------------------------

def dx2_energy_fraction(B, A, rows):
    """||(B@A)[rows,:]||_F² / ||B@A||_F² via r x r Grams — never the dense delta.
    Adapter scaling cancels in the ratio, so raw factors are exact here."""
    G_A = A @ A.t()
    idx = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    B_r = B[idx]
    num = torch.trace((B_r.t() @ B_r) @ G_A).item()
    den = torch.trace((B.t() @ B) @ G_A).item()
    return num / max(den, 1e-30)


def run_dx2(cfg, n_values, out_path):
    seed = int(cfg["pool_seed"])
    pool_dir = _resolve(cfg["pool_dir"])
    per_n = []
    for n in sorted(set(int(v) for v in n_values)):
        authors = subset_authors(seed, n)
        blocks = {}  # slot -> per-position row blocks (computed once per slot per N)
        fracs, by_type, by_author = [], {}, {}
        for j, a in enumerate(authors):
            slots, _ = _read_adapter(os.path.join(pool_dir, f"shard_{a}"))
            a_fracs = []
            for slot, (A, B) in slots.items():
                if slot not in blocks:
                    blocks[slot] = hash_row_blocks(B.shape[0], n, seed, slot)
                frac = dx2_energy_fraction(B, A, blocks[slot][j])
                fracs.append(frac)
                a_fracs.append(frac)
                by_type.setdefault(_slot_type(slot), []).append(frac)
            by_author[str(a)] = float(np.mean(a_fracs))
        row = {"n": n, "expected_fraction": 1.0 / n,
               "frac_mean": float(np.mean(fracs)), "frac_std": float(np.std(fracs)),
               "frac_min": float(np.min(fracs)), "frac_max": float(np.max(fracs)),
               "per_slot_type": {t: float(np.mean(v)) for t, v in sorted(by_type.items())},
               "per_author": by_author}
        per_n.append(row)
        print(f"[dx2] N={n}: own-row energy {row['frac_mean']:.4f}±{row['frac_std']:.4f} "
              f"(expected ~{1.0 / n:.4f})")
    out = {"pool_dir": pool_dir, "pool_seed": seed,
           "note": "fraction of each author delta's Frobenius energy in its own hash row "
                   "block; ~1/N for unconstrained adapters, ->1 under the [wd] rowslice arm",
           "script_sha256": _script_sha(), "per_n": per_n}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[dx2] wrote {out_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ops", nargs="+", default=None,
                    help="override config ops, e.g. --ops dare0p9 hash")
    ap.add_argument("--n_ladder", nargs="+", type=int, default=None)
    ap.add_argument("--limit_authors", type=int, default=None,
                    help="cheap smoke: drop ladder/DX N values above this author count")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the plan; read and write nothing")
    ap.add_argument("--force", action="store_true", help="rebuild existing artifacts")
    ap.add_argument("--dx1", action="store_true", help="cancellation diagnostic (no grid)")
    ap.add_argument("--dx2", action="store_true", help="owned-row energy diagnostic (no grid)")
    ap.add_argument("--report_dir", default=None,
                    help="DX output dir (default <repo>/reports)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    report_dir = args.report_dir or os.path.join(_REPO_DIR, "reports")

    if args.dx1:
        dx1_cfg = cfg.get("dx1", {})
        n_values = [int(n) for n in dx1_cfg.get("n_values", [8, 32, 200])]
        if args.limit_authors is not None:
            n_values = [n for n in n_values if n <= args.limit_authors]
        run_dx1(cfg, n_values, int(dx1_cfg.get("null_draws", 5)),
                os.path.join(report_dir, "ctv_dx1_cancellation.json"))
    if args.dx2:
        n_values = [int(n) for n in (args.n_ladder or cfg["n_ladder"])]
        if args.limit_authors is not None:
            n_values = [n for n in n_values if n <= args.limit_authors]
        run_dx2(cfg, n_values, os.path.join(report_dir, "ctv_dx2_energy.json"))
    if not (args.dx1 or args.dx2):
        run_grid(cfg, args.config, ops=args.ops, n_ladder=args.n_ladder,
                 limit_authors=args.limit_authors, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
