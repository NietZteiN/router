"""N-merge interference sweep: merge arbitrary subsets of the per-author (k=200) LoRAs.

Exp 5 of the merge-mechanism study (log/merge_mechanism/2026-07-07_interference-vs-n-design.md):
fix one-LoRA-per-author granularity and vary the NUMBER merged, N in {1,2,4,...,200}. The prior
k-scaling sweep varied shard count with total data fixed (200/k authors per shard) — a different
axis. Subsets are NESTED: perm = RandomState(seed).permutation(199) (author 199 = the k=200
forget shard, held out except at N=200); subset(N) = perm[:N], so the probe authors perm[:5]
are members of every N and per-author recall is tracked longitudinally.

Every merge is materialized on CPU to a single normal PEFT adapter dir so eval_tofu.py
--preloaded_adapter never pays the fp32 high-k memory law (k200 x r32 in-model = ~65 GiB,
impossible on a 46 GiB A40).

Merge methods:
  additive_mean  true-scale mean (1/N) of effective deltas sum_i (1/N) scaling_i B_i A_i —
                 the honest merge for "effect of merging" per the 2026-07-01 rsLoRA-artifact
                 correction. Pure factor-space concat from safetensors, NO base model needed.
                 Output adapter uses use_rslora=false + lora_alpha=r (PEFT scaling == 1.0), so
                 the stored factors ARE the effective delta — no output-scaling subtlety.
                 N>svd threshold: exact cat rank = 32N doesn't fit eval (fp32 law), so the
                 delta is SVD-compressed to --svd_rank in FACTORED form (QR of the stacks +
                 SVD of the small core via merge_extra._compress_factored — never a dense
                 d_out x d_in product). Acceptance: validate svd vs exact at the largest N
                 where both fit (see the config's svd_n_values).
  additive_sum   composable-task-vector (ctv, Wave 0) sum mode: weight 1.0 per effective
                 delta, sum_i s_i B_i A_i — the same factor-space cat as additive_mean with
                 no 1/N. Deletion is literal subtraction of one delta (== dropping its
                 factor block), verified by verify_subtraction.py. Purely additive next to
                 the mean path: additive_mean's code and outputs are byte-untouched (Exp-6
                 comparability). Identical math to centered_lowrank rho=0, kept first-class
                 so configs/CLI reach it without the centered machinery.
  dare_ties      the established (sqrt-r-inflated) baseline convention, for comparability with
                 the prior k-scaling curve. Cannot be computed factor-free: loads the base
                 model ON CPU + the subset adapters and calls merge_lora.merge_shards (the
                 exact same code path as the in-model evals; density 0.7, majority_sign
                 "total", uniform 1/N, torch.manual_seed pinned), then saves the merged
                 adapter dir via PeftModel.save_pretrained.
  centered_pool  the "third regime" (shared ~1x, residuals ~1x) with S estimated by the FULL
  centered_lowrank
                 pool mean / a per-slot rank-rho SVD of the subset mean:
                 M = sum_i s_i B_i A_i - (N-1)*S = S + sum_i (Delta_i - S)
                 (log/merge_mechanism/2026-07-15_centered-merge-design.md). ⚠ The literal
                 PATHS_FORWARD §6.1 formula (S = the exact subset mean) is the ALGEBRAIC
                 IDENTITY k*mean - (k-1)*mean = mean == additive_mean and is deliberately not
                 a method here — rejected as degenerate; test_merge_subset proves the identity.
                 centered_pool: S = mean over pool_authors (default 0..198); non-degenerate
                 only while subset ⊊ pool, so cap via methods.centered_pool.max_n (≤64).
                 centered_lowrank: S = P_rho(subset mean) via _compress_factored per slot;
                 rho=0 -> the naive unit sum, rho>=cat rank -> the mean (both proven in the
                 CPU gate). Deletion stays exact for both: S is a deterministic function of
                 the adapter files — recompute without author j and re-merge.

CLI (config-driven; see configs/nmerge_interference_7b.json, configs/nmerge_centered_7b.json):
  python merge_subset.py plan    --config C          # manifests + permutation printout
  python merge_subset.py merge   --config C --method additive_mean --n 8 --seed 42 [--svd_rank 1024]
  python merge_subset.py merge   --config C --method centered_lowrank --n 8 --seed 42 --rho 16
  python merge_subset.py merge   --config C --cross_check          # the r8 dare_ties N=200 check
  python merge_subset.py overlap --config C          # per-N subset col(B)/cosine stats (CPU)

Run heavy merges on SLURM via submit_nmerge.sh (login node is for the `plan` stage only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import torch

from jd_collection import _adapter_scaling, _read_adapter, _PREFIX
from merge_extra import _compress_factored
import tofu_env as _tofu_env

# Polite CPU cap on shared nodes; SLURM jobs get --cpus-per-task to match.
torch.set_num_threads(int(os.environ.get("NMERGE_THREADS", min(32, os.cpu_count() or 1))))

N_AUTHORS = 200  # TOFU authors; author 199 = the k=200 forget shard


# ---------------------------------------------------------------------------
# Config + deterministic subset derivation
# ---------------------------------------------------------------------------

_UNEXPANDED = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _expand_paths(obj, _key="<root>"):
    """Expand ${VAR} / $VAR and ~ in every string of a loaded config.

    Lets a config say "${TOFU_CKPT_ROOT}/Llama-2-7B-chat-hf_k200_r32_e5_lr1e4" instead of an
    absolute /storage2 path, which is what makes the same config usable on another cluster
    (cluster_env.<site>.sh sets TOFU_CKPT_ROOT). Absolute paths pass through untouched, so every
    pre-existing config is unaffected.

    An UNSET variable is a hard error, not a silent literal: os.path.expandvars leaves "${FOO}"
    as-is when FOO is undefined, and a path-shaped key would then be created on disk verbatim —
    which is exactly what happened once (a literal `${TOFU_CKPT_ROOT}/` directory) before this
    guard existed.
    """
    if isinstance(obj, str):
        out = os.path.expanduser(os.path.expandvars(obj))
        if "$" in out and _UNEXPANDED.search(out) and not _key.startswith("_"):
            raise SystemExit(
                f"config key {_key!r}: unresolved variable in {out!r}. The site env is not "
                f"loaded — run through a submit_*.sh, or set the variable explicitly (e.g. "
                f"TOFU_CKPT_ROOT). See cluster_env.<site>.sh.")
        return out
    if isinstance(obj, list):
        return [_expand_paths(v, _key) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand_paths(v, k) for k, v in obj.items()}
    return obj


def load_config(path):
    # Pull TOFU_*/HF_HOME from cluster_env.<site>.sh when the caller did not come via a driver,
    # so `python merge_subset.py ...` by hand behaves like the same command inside a job.
    try:
        from tofu_env import ensure_site_env
        ensure_site_env()
    except ImportError:
        pass
    with open(path) as f:
        cfg = _expand_paths(json.load(f))
    for key in ("model_name", "shards_dir", "out_dir", "n_ladder", "subset_seeds", "eval"):
        if key not in cfg:
            raise KeyError(f"config missing {key!r}")
    return cfg


def author_permutation(seed):
    """Nested-subset source: permutation of authors 0..198 (author 199 held out)."""
    return np.random.RandomState(seed).permutation(N_AUTHORS - 1)


def subset_authors(seed, n):
    """subset(N) = first N of the seed's permutation; N=200 = ALL authors (incl. 199)."""
    if n == N_AUTHORS:
        return list(range(N_AUTHORS))
    perm = author_permutation(seed)
    if n > len(perm):
        raise ValueError(f"n={n} > {len(perm)} (only N=200 may include author 199)")
    return perm[:n].tolist()


def probe_authors(seed, n, n_probes):
    """Fixed probes = head of the permutation, present in every nested subset."""
    return author_permutation(seed)[: min(n, n_probes)].tolist()


def _parse_author_list(s):
    """'180-199' / '180,181,195' / '180-184,199' -> a sorted, de-duplicated author list."""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    authors = sorted(set(out))
    bad = [a for a in authors if not 0 <= a < N_AUTHORS]
    if bad:
        raise ValueError(f"author ids out of range 0..{N_AUTHORS - 1}: {bad}")
    if not authors:
        raise ValueError("empty author list")
    return authors


def _lam_tag(lam):
    """Label token for the additive_sum global coefficient. None/1.0 -> '' so the pre-existing
    grammar (`nmerge_sum_N{n}_s{seed}`) is byte-unchanged; 'isqrt' -> 'isqrt'; a float ->
    'L<value>' with '.'->'p' and '-'->'m' (PEFT disallows dots in adapter names)."""
    if lam is None or lam == 1.0:
        return ""
    if lam == "isqrt":
        return "isqrt"
    return "L" + str(lam).replace(".", "p").replace("-", "m")


def lam_weight(lam, n):
    """Per-adapter coefficient for additive_sum.

    None/1.0 = the literal APA rule (Delta = sum_i s_i B_i A_i).
    'isqrt'  = 1/sqrt(N), the MATCHED-NORM control: because this pool's per-author effective
               deltas are mutually near-orthogonal (measured mean |cos| 0.0009-0.0051 over 3
               slots, 2026-07-28), ||sum_i Delta_i|| grows as sqrt(N), so weighting by
               1/sqrt(N) holds the injected perturbation CONSTANT across N. That separates the
               aggregation RULE from delta MAGNITUDE -- without it, sum-vs-mean at N confounds
               the two (they differ in ||Delta|| by exactly N).
    """
    if lam is None:
        return 1.0
    if lam == "isqrt":
        return 1.0 / math.sqrt(n)
    return float(lam)


def parse_lam(s):
    """CLI/config lambda: None, the literal 'isqrt', or a float."""
    if s is None or s == "" or s == "-":
        return None
    if s == "isqrt":
        return "isqrt"
    return float(s)


def merge_label(method, n, seed, svd_rank=None, rho=None, lam=None):
    if method == "centered_lowrank":
        tag = f"cr{rho}"
    else:
        tag = {"additive_mean": "add", "additive_sum": "sum", "dare_ties": "dare",
               "centered_pool": "cpool"}[method]
    if method == "additive_sum":
        tag += _lam_tag(lam)
    svd = f"_svd{svd_rank}" if svd_rank else ""
    return f"nmerge_{tag}{svd}_N{n}_s{seed}"


def _script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Materialize: write a normal PEFT adapter dir with scaling == 1
# ---------------------------------------------------------------------------

def write_effective_adapter(out_dir, slots_AB, ref_cfg, out_rank):
    """Write {slot: (A, B)} whose B @ A IS the effective delta (PEFT scaling forced to 1.0
    via use_rslora=false + lora_alpha == r). Factors below out_rank are zero-padded so every
    slot shares the config rank (PEFT requires one r per adapter_config)."""
    tensors = {}
    for slot, (A, B) in slots_AB.items():
        r = A.shape[0]
        if r > out_rank:
            raise ValueError(f"slot {slot}: rank {r} > out_rank {out_rank}")
        A_pad = torch.zeros(out_rank, A.shape[1])
        B_pad = torch.zeros(B.shape[0], out_rank)
        A_pad[:r] = A
        B_pad[:, :r] = B
        key = _PREFIX + slot
        tensors[key + ".lora_A.weight"] = A_pad.contiguous()
        tensors[key + ".lora_B.weight"] = B_pad.contiguous()
    cfg = dict(ref_cfg)
    cfg["r"] = out_rank
    cfg["lora_alpha"] = out_rank      # alpha/r = 1.0 ...
    cfg["use_rslora"] = False         # ... under standard LoRA scaling
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return out_dir


def _weighted_factor_cat(shard_dirs, weights, *, svd_rank=None):
    """Factor-space weighted cat: per slot, B_cat = concat_i w_i s_i B_i, A_cat = concat_i A_i,
    so B_cat @ A_cat = sum_i w_i s_i B_i A_i exactly. svd_rank compresses that delta in
    factored form (exact truncated SVD of the full cat, never dense). The shared core of
    additive_mean (w = 1/N) and the centered merges (mixed-sign weights)."""
    assert len(weights) == len(shard_dirs)
    per_adapter, cfgs = [], []
    for d in shard_dirs:
        s, cfg = _read_adapter(d)
        per_adapter.append(s)
        cfgs.append(cfg)
    slot_names = list(per_adapter[0].keys())
    merged, sum_rank, svd_energy = {}, 0, []
    for name in slot_names:
        B_list, A_list = [], []
        for s, cfg, w in zip(per_adapter, cfgs, weights):
            if name not in s:
                raise ValueError(f"adapter missing slot {name!r}")
            A, B = s[name]
            B_list.append((w * _adapter_scaling(cfg)) * B)
            A_list.append(A)
        B_cat = torch.cat(B_list, dim=1)
        A_cat = torch.cat(A_list, dim=0)
        sum_rank = max(sum_rank, A_cat.shape[0])
        if svd_rank is not None and svd_rank < A_cat.shape[0]:
            A_new, B_new = _compress_factored(B_cat, A_cat, svd_rank)
            # retained-energy diagnostic without the dense delta: ||B_new A_new||_F / ||B_cat A_cat||_F
            num = _factored_fro(B_new, A_new)
            den = _factored_fro(B_cat, A_cat)
            svd_energy.append((num / max(den, 1e-30)) ** 2)
            merged[name] = (A_new, B_new)
        else:
            merged[name] = (A_cat, B_cat)
    out_rank = svd_rank if (svd_rank is not None and svd_rank < sum_rank) else sum_rank
    meta = {"sum_rank": sum_rank, "out_rank": out_rank,
            "svd_energy_mean": float(np.mean(svd_energy)) if svd_energy else None,
            "svd_energy_min": float(np.min(svd_energy)) if svd_energy else None}
    return merged, cfgs[0], out_rank, meta


def merge_additive_mean(shard_dirs, *, svd_rank=None):
    """Factor-space true mean: per slot, B_cat = concat_i (1/N) s_i B_i, A_cat = concat_i A_i,
    so B_cat @ A_cat = (1/N) sum_i s_i B_i A_i exactly. svd_rank compresses that delta in
    factored form (exact truncated SVD, never dense)."""
    n = len(shard_dirs)
    return _weighted_factor_cat(shard_dirs, [1.0 / n] * n, svd_rank=svd_rank)


def merge_additive_sum(shard_dirs, *, svd_rank=None, weight=1.0):
    """Factor-space unit sum (ctv sum mode): per slot, B_cat = concat_i w * s_i B_i,
    A_cat = concat_i A_i, so B_cat @ A_cat = w * sum_i s_i B_i A_i exactly. Deleting author j
    is literal subtraction of its effective delta (== dropping its factor block) — the
    property verify_subtraction.py checks. Math == centered_lowrank rho=0 (proven in the
    CPU gate); first-class so configs/CLI reach it without the centered machinery.

    `weight` is a single GLOBAL coefficient applied identically to every adapter (see
    lam_weight): it keeps the aggregation uniform and keeps drop-a-term exact, because a
    FIXED w makes Delta(S\\{j}) = Delta(S) - w*s_j B_j A_j. Default 1.0 = the literal rule,
    byte-identical to the pre-2026-07-28 behavior."""
    return _weighted_factor_cat(shard_dirs, [weight] * len(shard_dirs), svd_rank=svd_rank)


def merge_centered_pool(shard_dirs, pool_dirs, *, svd_rank=None):
    """Pool-mean centered sum: M = sum_i s_i B_i A_i - (N-1) * mean_pool(s_j B_j A_j).

    Subset factors enter at weight 1.0 (full strength), pool factors at -(N-1)/P — a single
    weighted cat, so one exact truncated SVD compresses M with no cascaded error. Subset
    members appear twice (once per role); that is the intended algebra (subset ⊂ pool).
    Degenerate boundary (proven in the CPU gate): pool == subset ⇒ M == the additive mean.
    """
    n, p = len(shard_dirs), len(pool_dirs)
    dirs = list(shard_dirs) + list(pool_dirs)
    weights = [1.0] * n + [-(n - 1) / p] * p
    return _weighted_factor_cat(dirs, weights, svd_rank=svd_rank)


def merge_centered_lowrank(shard_dirs, rho, *, svd_rank=None):
    """Low-rank centered sum: M = sum_i s_i B_i A_i - (N-1) * P_rho(subset mean)
                                = P_rho + sum_i (Delta_i - P_rho).

    P_rho = per-slot rank-rho truncated SVD of the subset mean (exact, factored). Endpoints
    (proven in the CPU gate): rho = 0 -> the naive unit sum; rho >= cat rank -> the mean.
    center_energy_* records ||P_rho||^2 / ||mean||^2 — how much of the mean is "shared" at
    rank rho; the remainder is amplified ~N x in M (the crosstalk term under measurement).
    """
    n = len(shard_dirs)
    if rho == 0:
        merged, ref_cfg, out_rank, meta = _weighted_factor_cat(
            shard_dirs, [1.0] * n, svd_rank=svd_rank)
        return merged, ref_cfg, out_rank, {**meta, "center_energy_mean": None,
                                           "center_energy_min": None}
    per_adapter, cfgs = [], []
    for d in shard_dirs:
        s, cfg = _read_adapter(d)
        per_adapter.append(s)
        cfgs.append(cfg)
    slot_names = list(per_adapter[0].keys())
    merged, sum_rank, svd_energy, center_energy = {}, 0, [], []
    for name in slot_names:
        B_full, A_full, B_mean, A_mean = [], [], [], []
        for s, cfg in zip(per_adapter, cfgs):
            if name not in s:
                raise ValueError(f"adapter missing slot {name!r}")
            A, B = s[name]
            scale = _adapter_scaling(cfg)
            B_full.append(scale * B)
            A_full.append(A)
            B_mean.append((scale / n) * B)
            A_mean.append(A)
        B_mean_cat = torch.cat(B_mean, dim=1)
        A_mean_cat = torch.cat(A_mean, dim=0)
        if rho >= A_mean_cat.shape[0]:
            A_p, B_p = A_mean_cat, B_mean_cat          # P_rho == mean (degenerate endpoint)
            center_energy.append(1.0)
        else:
            A_p, B_p = _compress_factored(B_mean_cat, A_mean_cat, rho)
            num = _factored_fro(B_p, A_p)
            den = _factored_fro(B_mean_cat, A_mean_cat)
            center_energy.append((num / max(den, 1e-30)) ** 2)
        B_cat = torch.cat(B_full + [-(n - 1) * B_p], dim=1)
        A_cat = torch.cat(A_full + [A_p], dim=0)
        sum_rank = max(sum_rank, A_cat.shape[0])
        if svd_rank is not None and svd_rank < A_cat.shape[0]:
            A_new, B_new = _compress_factored(B_cat, A_cat, svd_rank)
            num = _factored_fro(B_new, A_new)
            den = _factored_fro(B_cat, A_cat)
            svd_energy.append((num / max(den, 1e-30)) ** 2)
            merged[name] = (A_new, B_new)
        else:
            merged[name] = (A_cat, B_cat)
    out_rank = svd_rank if (svd_rank is not None and svd_rank < sum_rank) else sum_rank
    meta = {"sum_rank": sum_rank, "out_rank": out_rank,
            "svd_energy_mean": float(np.mean(svd_energy)) if svd_energy else None,
            "svd_energy_min": float(np.min(svd_energy)) if svd_energy else None,
            "center_energy_mean": float(np.mean(center_energy)) if center_energy else None,
            "center_energy_min": float(np.min(center_energy)) if center_energy else None}
    return merged, cfgs[0], out_rank, meta


def _factored_fro(B, A):
    """||B @ A||_F computed via the r x r Gram matrices (no dense product)."""
    G = (B.t() @ B) @ (A @ A.t())
    return math.sqrt(max(torch.trace(G).item(), 0.0))


def merge_dare_ties_cpu(model_name, shard_dirs, *, seed=0, density=0.7):
    """Replicate the in-model dare_ties merge on CPU and return the merged PeftModel + name.

    Loads the base bf16 on CPU + the subset adapters under names shard_0..shard_{N-1} (order =
    subset order) and calls merge_lora.merge_shards — the byte-same code path the k-scaling
    evals used (uniform 1/N, density, majority_sign 'total'). torch.manual_seed pins DARE's
    Bernoulli masks.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import merge_lora

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, shard_dirs[0], adapter_name="shard_0")
    for i, d in enumerate(shard_dirs[1:], start=1):
        model.load_adapter(d, adapter_name=f"shard_{i}")
    torch.manual_seed(seed)
    name = merge_lora.merge_shards(
        model, len(shard_dirs), "dare_ties", density=density, seed=seed,
        adapter_name="nmerge_dare")
    return model, name


def save_peft_adapter(model, adapter_name, out_dir):
    """Save ONE adapter of a multi-adapter PeftModel as a standalone adapter dir."""
    import tempfile
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out_dir.rstrip("/"))) as tmp:
        model.save_pretrained(tmp, selected_adapters=[adapter_name])
        src = os.path.join(tmp, adapter_name)
        os.makedirs(out_dir, exist_ok=True)
        for fn in os.listdir(src):
            os.replace(os.path.join(src, fn), os.path.join(out_dir, fn))
    return out_dir


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _merge_specs(cfg):
    """Yield dicts describing every merge the config asks for (drives plan + arrays)."""
    specs = []
    m = cfg["methods"]["additive_mean"]
    if m.get("enabled", True):
        svd_rank = m.get("svd_rank")
        svd_ns = set(m.get("svd_n_values", []))
        for seed in cfg["subset_seeds"]:
            for n in cfg["n_ladder"]:
                if n == 1:
                    continue  # N=1 = the raw shard dir, no merge artifact
                if n <= m.get("exact_max_n", 64):
                    specs.append({"method": "additive_mean", "n": n, "seed": seed, "svd_rank": None})
                if n in svd_ns:
                    specs.append({"method": "additive_mean", "n": n, "seed": seed, "svd_rank": svd_rank})
    asum = cfg["methods"].get("additive_sum", {})
    if asum.get("enabled", False):
        # ctv sum mode — default OFF so every pre-existing config's spec list is unchanged.
        sum_svd_rank = asum.get("svd_rank")
        sum_svd_ns = set(asum.get("svd_n_values", []))
        # lam_values: global coefficients to sweep. Default [None] = the literal unit sum,
        # so a config that predates the 2026-07-28 lambda arm emits an identical spec list.
        lam_values = [parse_lam(v) for v in asum.get("lam_values", [None])]
        # lam_n_values: optional per-lambda ladder restriction, keyed by the lambda as written
        # in lam_values. The matched-norm control needs only a few rungs to establish its slope,
        # and each materialized merge costs ~2.01M * rank * 4 B (rank = 32N, or svd_rank) —
        # 7.7 GiB at rank 1024 — on a partition that runs near-full.
        lam_ns = asum.get("lam_n_values", {})
        for seed in cfg["subset_seeds"]:
            for n in cfg["n_ladder"]:
                if n == 1:
                    continue
                for lam in lam_values:
                    allowed = lam_ns.get("null" if lam is None else str(lam))
                    if allowed is not None and n not in allowed:
                        continue
                    # the 'lam' key is OMITTED when None so a config without lam_values emits a
                    # spec list byte-identical to the pre-2026-07-28 one (test_merge_subset
                    # compares spec dicts exactly — that equality is the invariant).
                    extra_lam = {} if lam is None else {"lam": lam}
                    if n <= asum.get("exact_max_n", 64):
                        specs.append({"method": "additive_sum", "n": n, "seed": seed,
                                      "svd_rank": None, **extra_lam})
                    if n in sum_svd_ns:
                        specs.append({"method": "additive_sum", "n": n, "seed": seed,
                                      "svd_rank": sum_svd_rank, **extra_lam})
    d = cfg["methods"].get("dare_ties", {})
    if d.get("enabled", False):
        for seed in cfg["subset_seeds"]:
            for n in cfg["n_ladder"]:
                if n == 1:
                    continue
                specs.append({"method": "dare_ties", "n": n, "seed": seed, "svd_rank": None})
    cp = cfg["methods"].get("centered_pool", {})
    if cp.get("enabled", False):
        # always SVD-compressed (cat rank = 32*(N + pool)); capped at max_n — subset -> pool
        # degenerates the estimator back to the mean (see module docstring).
        for seed in cfg["subset_seeds"]:
            for n in cfg["n_ladder"]:
                if n == 1 or n > cp.get("max_n", 64):
                    continue
                specs.append({"method": "centered_pool", "n": n, "seed": seed,
                              "svd_rank": cp.get("svd_rank", 1024)})
    cl = cfg["methods"].get("centered_lowrank", {})
    if cl.get("enabled", False):
        svd_ns = set(cl.get("svd_n_values", []))
        for seed in cfg["subset_seeds"]:
            for rho in cl.get("rho_values", [16]):
                for n in cfg["n_ladder"]:
                    if n == 1:
                        continue
                    if n <= cl.get("exact_max_n", 64):
                        specs.append({"method": "centered_lowrank", "n": n, "seed": seed,
                                      "svd_rank": None, "rho": rho})
                    if n in svd_ns:
                        specs.append({"method": "centered_lowrank", "n": n, "seed": seed,
                                      "svd_rank": cl.get("svd_rank"), "rho": rho})
    return specs


def _subset_arms(cfg):
    """(method, rho) arms that get subset-conditioned utility rows (dare_ties never did)."""
    m = cfg["methods"]
    arms = []
    if m.get("additive_mean", {}).get("enabled", True):
        arms.append(("additive_mean", None))
    if m.get("additive_sum", {}).get("enabled", False):
        arms.append(("additive_sum", None))
    if m.get("centered_pool", {}).get("enabled", False):
        arms.append(("centered_pool", None))
    if m.get("centered_lowrank", {}).get("enabled", False):
        for rho in m["centered_lowrank"].get("rho_values", [16]):
            arms.append(("centered_lowrank", rho))
    return arms


def _canonical_label(cfg, method, n, seed, rho=None, lam=None):
    """The one servable label per (method, n): exact below exact_max_n, else svd-compressed;
    centered_pool is always svd-compressed."""
    mb = cfg["methods"][method]
    if method == "centered_pool":
        svd = mb.get("svd_rank", 1024)
    elif method in ("additive_mean", "additive_sum", "centered_lowrank"):
        svd = None if n <= mb.get("exact_max_n", 64) else mb.get("svd_rank")
    else:
        raise ValueError(f"no canonical label rule for {method!r}")
    return merge_label(method, n, seed, svd, rho, lam)


def _spec_argv(cfg_path, spec):
    argv = [sys.executable, os.path.abspath(__file__), "merge", "--config", cfg_path,
            "--method", spec["method"], "--n", str(spec["n"]), "--seed", str(spec["seed"])]
    if spec.get("svd_rank"):
        argv += ["--svd_rank", str(spec["svd_rank"])]
    if spec.get("rho") is not None:
        argv += ["--rho", str(spec["rho"])]
    if spec.get("lam") is not None:
        argv += ["--lam", str(spec["lam"])]
    return argv


def do_plan(cfg, cfg_path):
    out_dir = cfg["out_dir"]
    os.makedirs(os.path.join(out_dir, "merges"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results", "smoke"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    # KS reference so forget_quality isn't NaN (probe-run fq is ignored, but keep it defined).
    ref = cfg.get("retain_tr_source")
    dst = os.path.join(out_dir, "results", "smoke", "retain_tr_scores.npy")
    if ref and os.path.exists(ref) and not os.path.exists(dst):
        import shutil
        shutil.copy2(ref, dst)
        print(f"[plan] copied KS ref -> {dst}")

    for seed in cfg["subset_seeds"]:
        perm = author_permutation(seed)
        print(f"[plan] seed {seed}: perm[:10] = {perm[:10].tolist()}  "
              f"probes = {perm[:cfg.get('n_probes', 5)].tolist()}")

    # ---- merge manifest: one "method n seed svd_rank" line per merge (TSV) ----
    specs = _merge_specs(cfg)
    if cfg.get("cross_check", {}).get("enabled"):
        specs.append({"method": "cross_check", "n": cfg["cross_check"]["n"],
                      "seed": cfg["subset_seeds"][0], "svd_rank": None})
    merge_manifest = os.path.join(out_dir, "merge_manifest.txt")
    with open(merge_manifest, "w") as f:
        for s in specs:
            rho = s.get("rho")
            lam = s.get("lam")
            # 6th column (lam) added 2026-07-28; '-' for every non-additive_sum row and for the
            # literal unit sum, so submit_nmerge.sh's "empty or '-' => skip" rule keeps
            # pre-existing manifests working unchanged.
            f.write(f"{s['method']}\t{s['n']}\t{s['seed']}\t{s.get('svd_rank') or '-'}\t"
                    f"{'-' if rho is None else rho}\t{'-' if lam is None else lam}\n")
    print(f"[plan] {len(specs)} merge specs -> {merge_manifest}")

    # ---- eval manifest: "label \t adapter_dir|BASE \t eval_shard_id|- \t retain_ids|-" ----
    # retain_ids != '-' => subset-conditioned utility row (eval_tofu --retain_author_ids;
    # result file {label}__subset.json).
    lines = []
    n_probes = cfg.get("n_probes", 5)
    seed0 = cfg["subset_seeds"][0]

    # fixed_probe_authors (2026-07-28, default absent = unchanged): use ONE probe set for every
    # seed instead of each seed's own perm head. Two reasons, both load-bearing for Exp A:
    #  (1) per-seed probes are disjoint across seeds (perm(43)[:3]=[56,37,67]), so anchors and
    #      iso rows would have to be re-evaluated per seed and the retain draw would differ;
    #  (2) probe_authors clamps to min(n, n_probes), so N=2 would emit only 2 probe rows and
    #      break a ">=3 authors per condition" requirement.
    # With it pinned, `retain_indices` depends only on the measure shard and rng(0) is fixed, so
    # the retain/real/world question sets are byte-identical across every condition => paired
    # per-question tests are valid. NOTE the probes need not be subset members; callers must
    # read merge_meta.json["authors"] to tell "interfered" from "never trained".
    _fixed = cfg.get("fixed_probe_authors")

    def _probes(seed, n):
        return list(_fixed)[:n_probes] if _fixed else probe_authors(seed, n, n_probes)

    probes0 = _probes(seed0, N_AUTHORS)

    # iso references: the 5 probe adapters served alone (N=1 point = probes0[0]).
    for a in probes0:
        lines.append((f"iso_a{a}", os.path.join(cfg["shards_dir"], f"shard_{a}"), a, "-"))

    # ladder probe evals (headline mu = the probes0[0] job of each label).
    for spec in _merge_specs(cfg):
        label = merge_label(spec["method"], spec["n"], spec["seed"], spec.get("svd_rank"),
                            spec.get("rho"), spec.get("lam"))
        adapter = os.path.join(out_dir, "merges", label)
        pr = _probes(spec["seed"], spec["n"])
        exact_max = cfg["methods"].get(spec["method"], {}).get("exact_max_n")
        if spec.get("svd_rank") and exact_max is not None and spec["n"] <= exact_max:
            pr = pr[:1]  # svd-vs-exact validation point: one probe is enough
        for a in pr:
            lines.append((label, adapter, a, "-"))

    # subset-conditioned utility: retain_* scored ONLY on the merged authors' rows
    # ("did it learn what it was trained on"); forget stays the global forget author.
    sm = cfg.get("subset_mu", {})
    if sm.get("enabled"):
        cp_max = cfg["methods"].get("centered_pool", {}).get("max_n", 64)
        for n in sm.get("n_values", cfg["n_ladder"]):
            rids = ",".join(str(a) for a in subset_authors(seed0, n))
            if n == 1:
                a0 = probes0[0]
                lines.append((f"iso_a{a0}", os.path.join(cfg["shards_dir"], f"shard_{a0}"),
                              "-", str(a0)))
                continue
            for method, rho in _subset_arms(cfg):
                if method == "centered_pool" and n > cp_max:
                    continue
                label = _canonical_label(cfg, method, n, seed0, rho)
                lines.append((label, os.path.join(out_dir, "merges", label), "-", rids))
        # ceiling/floor on the SAME rows: joint-ft and base, retain-restricted to a
        # representative subset (anchor_n, default 8).
        an = sm.get("anchor_n", 8)
        rids_a = ",".join(str(a) for a in subset_authors(seed0, an))
        if cfg.get("anchors", {}).get("ft_adapter"):
            lines.append((f"ft_r32_sub{an}", cfg["anchors"]["ft_adapter"], "-", rids_a))
        if cfg.get("anchors", {}).get("base_model"):
            lines.append((f"base_model_sub{an}", "BASE", "-", rids_a))

    # anchors, probed at probes0[0] so the retain split matches the ladder.
    # anchors.at_all_probes (2026-07-28, default false = unchanged) repeats them at EVERY probe,
    # so each probe's ladder has its own floor/ceiling on the identical retain draw.
    anch = cfg.get("anchors", {})
    anchor_probes = probes0 if anch.get("at_all_probes") else probes0[:1]
    for a in anchor_probes:
        if anch.get("base_model"):
            lines.append(("base_model", "BASE", a, "-"))
        if anch.get("ft_adapter"):
            lines.append(("ft_r32", anch["ft_adapter"], a, "-"))
        if anch.get("retain90_adapter"):
            lines.append(("retain90_oracle", anch["retain90_adapter"], a, "-"))

    # r8 cross-check: same condition as the prior in-model row (forget 199, no probe).
    cc = cfg.get("cross_check", {})
    if cc.get("enabled"):
        label = cc.get("label", "nmerge_dare_N200_s42_r8")
        lines.append((label, os.path.join(out_dir, "merges", label), "-", "-"))

    eval_manifest = os.path.join(out_dir, "eval_manifest_nmerge.txt")
    with open(eval_manifest, "w") as f:
        for label, adapter, sid, rids in lines:
            f.write(f"{label}\t{adapter}\t{sid}\t{rids}\n")
    print(f"[plan] {len(lines)} eval tasks -> {eval_manifest}")
    return merge_manifest, eval_manifest


def do_merge(cfg, cfg_path, args):
    if args.cross_check:
        cc = cfg["cross_check"]
        method, n, seed = cc.get("method", "dare_ties"), cc["n"], cfg["subset_seeds"][0]
        label = cc.get("label", "nmerge_dare_N200_s42_r8")
        shards_dir = cc["shards_dir"]
        svd_rank = None
    else:
        method, n, seed, svd_rank = args.method, args.n, args.seed, args.svd_rank
        shards_dir = cfg["shards_dir"]
        lam = parse_lam(getattr(args, "lam", None))
        # --authors (2026-07-28): an EXPLICIT author set, bypassing the nested-permutation
        # derivation. Needed because author_permutation() covers 0..198 only and perm(42)[:20]
        # contains no author from 180..198 — yet 180..199 are the only authors that have
        # paraphrased questions AND lie outside the retain90 oracle, so every leave-one-out
        # selectivity condition needs a hand-specified set. Requires --label (the permutation
        # no longer determines the name).
        explicit = getattr(args, "authors", None)
        if explicit:
            authors = _parse_author_list(explicit)
            if n is not None and n != len(authors):
                raise SystemExit(f"--n {n} disagrees with --authors ({len(authors)} authors)")
            n = len(authors)
            if not args.label:
                raise SystemExit("--authors requires an explicit --label")
            label = args.label
        else:
            authors = subset_authors(seed, n)
            label = args.label or merge_label(method, n, seed, svd_rank, args.rho, lam)

    out_dir = os.path.join(cfg["out_dir"], "merges", label)
    done = os.path.join(out_dir, "adapter_model.safetensors")
    if os.path.exists(done) and not args.force:
        print(f"[merge] skip existing {out_dir}")
        return

    if args.cross_check:
        authors = subset_authors(seed, n)
        lam = None
    shard_dirs = [os.path.join(shards_dir, f"shard_{a}") for a in authors]
    for d in shard_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(d)

    print(f"[merge] {label}: {method} over {n} adapters from {shards_dir}"
          f"{f' svd_rank={svd_rank}' if svd_rank else ''}")
    extra = {}
    weights_desc = f"1/{n} uniform"
    if method == "additive_mean":
        merged, ref_cfg, out_rank, extra = merge_additive_mean(shard_dirs, svd_rank=svd_rank)
        write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
    elif method == "additive_sum":
        w = lam_weight(lam, n)
        merged, ref_cfg, out_rank, extra = merge_additive_sum(
            shard_dirs, svd_rank=svd_rank, weight=w)
        write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
        extra = {**extra, "lam": lam, "lam_weight": w}
        weights_desc = ("1.0 each (ctv unit sum)" if lam is None or lam == 1.0
                        else f"{w:.6g} each (global lambda={lam})")
    elif method == "centered_pool":
        cp = cfg["methods"]["centered_pool"]
        pool_authors = cp.get("pool_authors") or list(range(N_AUTHORS - 1))
        pool_dirs = [os.path.join(shards_dir, f"shard_{a}") for a in pool_authors]
        for d in pool_dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(d)
        merged, ref_cfg, out_rank, extra = merge_centered_pool(
            shard_dirs, pool_dirs, svd_rank=svd_rank)
        write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
        extra = {**extra, "center": "pool_mean", "pool_size": len(pool_dirs)}
        weights_desc = f"subset 1.0 each - ({n}-1)/{len(pool_dirs)} per pool adapter"
    elif method == "centered_lowrank":
        if args.rho is None:
            raise SystemExit("centered_lowrank needs --rho")
        merged, ref_cfg, out_rank, extra = merge_centered_lowrank(
            shard_dirs, args.rho, svd_rank=svd_rank)
        write_effective_adapter(out_dir, merged, ref_cfg, out_rank)
        extra = {**extra, "center": f"lowrank_rho{args.rho}", "rho": args.rho}
        weights_desc = f"subset 1.0 each - ({n}-1)*P_rho(mean), rho={args.rho}"
    elif method == "dare_ties":
        d_cfg = cfg["methods"].get("dare_ties", {}) if not args.cross_check else cfg["cross_check"]
        model, name = merge_dare_ties_cpu(
            cfg["model_name"], shard_dirs,
            seed=d_cfg.get("merge_seed", 0), density=d_cfg.get("density", 0.7))
        save_peft_adapter(model, name, out_dir)
        extra = {"density": d_cfg.get("density", 0.7), "merge_seed": d_cfg.get("merge_seed", 0)}
    else:
        raise ValueError(f"unknown method {method!r}")

    meta = {
        "label": label, "method": method, "n": n, "subset_seed": seed,
        "svd_rank": svd_rank, "authors": authors, "shards_dir": shards_dir,
        "weights": weights_desc, "script_sha256": _script_sha(),
        "git_hash": _git_hash(), "config": os.path.abspath(cfg_path), **extra,
    }
    with open(os.path.join(out_dir, "merge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[merge] wrote {out_dir} ({meta.get('out_rank', 'src-rank')} rank)"
          + (f" svd_energy mean={extra['svd_energy_mean']:.4f}" if extra.get("svd_energy_mean") else ""))


def do_norms(cfg, out_json):
    """CPU-only delta-norm ladder — no GPU, no materialization, no base model on the accelerator.

    For each (seed, N) over the config's `norms.n_values`, streaming the NESTED subset prefix
    slot by slot (so peak memory is one slot's factors, not the pool):

      fro_sum        ||sum_{i<N} s_i B_i A_i||_F         the injected perturbation
      fro_indiv_l2   sqrt(sum_i ||s_i B_i A_i||_F^2)     the mutually-orthogonal null
      kappa          fro_sum / fro_indiv_l2              1 => orthogonal (sqrt(N) growth),
                                                         sqrt(N) => aligned (N growth)
      rel_pert       fro_sum / ||W_0||_F                 the collapse-predictor axis
      fro_mean       fro_sum / N                         the additive_mean twin, free

    Why this exists: model_utility is a step function near collapse (scipy hmean returns
    exactly 0 if any one of its 9 components is 0), so a mu-vs-N curve cannot say HOW MUCH the
    model broke. rel_pert can, and because rel_pert_sum(N) = N * rel_pert_mean(N) the sum and
    mean ladders land on ONE axis — which is what makes "the 1/N is only doing norm control"
    a testable claim rather than an interpretation.

    ||W_0||_F comes from the base model's safetensors shards on CPU (never loaded into a
    module), keyed by the same slot names as the adapters.
    """
    import glob as _glob
    from safetensors import safe_open

    nc = cfg.get("norms", {})
    n_values = sorted(nc.get("n_values") or cfg["n_ladder"])
    seeds = cfg["subset_seeds"]
    shards_dir = cfg["shards_dir"]
    max_n = max(n_values)

    # --- base-weight Frobenius norms (optional; only needed for rel_pert) ---
    base_fro = {}
    if nc.get("base_weight_norms", True):
        pats = os.path.join(_tofu_env.hf_home(),
                            "hub", "models--" + cfg["model_name"].replace("/", "--"),
                            "snapshots", "*", "*.safetensors")
        files = sorted(_glob.glob(pats))
        if not files:
            print(f"[norms] WARN no base safetensors under {pats} — rel_pert will be null")
        for f in files:
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    if not k.endswith(".weight"):
                        continue
                    slot = k[:-len(".weight")]
                    if ".layers." not in slot:
                        continue
                    t = h.get_tensor(k)
                    if t.ndim != 2:
                        continue
                    base_fro[slot] = float(torch.linalg.matrix_norm(t.float(), ord="fro"))
        print(f"[norms] base ||W||_F for {len(base_fro)} slots")

    rows = []
    for seed in seeds:
        authors = subset_authors(seed, max_n)
        # slot-outer / author-inner streaming: read one author's factors at a time, accumulate
        # the running cat's Gram contributions, snapshot at each rung.
        per_adapter, cfgs = [], []
        for a in authors:
            s, c = _read_adapter(os.path.join(shards_dir, f"shard_{a}"))
            per_adapter.append(s)
            cfgs.append(c)
        scal = [_adapter_scaling(c) for c in cfgs]
        slots = list(per_adapter[0].keys())
        acc = {n: {"sum_sq": 0.0, "indiv_sq": 0.0, "base_sq": 0.0} for n in n_values}
        per_slot = {}
        for name in slots:
            Bs, As = [], []
            for s, sc in zip(per_adapter, scal):
                A, B = s[name]
                As.append(A)
                Bs.append(B * sc)
            # Gram-only: ||sum_{i<N}||^2 = sum_{i,j<N} tr(Bi^T Bj Aj Ai^T); never dense.
            G = [[float(torch.einsum('ij,ij->', (Bs[i].t() @ Bs[j]), (As[i] @ As[j].t())))
                  for j in range(max_n)] for i in range(max_n)]
            bf = base_fro.get(name)
            for n in n_values:
                tot = sum(G[i][j] for i in range(n) for j in range(n))
                ind = sum(G[i][i] for i in range(n))
                acc[n]["sum_sq"] += max(tot, 0.0)
                acc[n]["indiv_sq"] += ind
                if bf is not None:
                    acc[n]["base_sq"] += bf * bf
                per_slot.setdefault(name, {})[n] = {
                    "fro_sum": math.sqrt(max(tot, 0.0)),
                    "fro_indiv_l2": math.sqrt(max(ind, 0.0)),
                    "rel_pert": (math.sqrt(max(tot, 0.0)) / bf) if bf else None,
                }
        for n in n_values:
            fs = math.sqrt(acc[n]["sum_sq"])
            fi = math.sqrt(acc[n]["indiv_sq"])
            bs = math.sqrt(acc[n]["base_sq"]) if acc[n]["base_sq"] else None
            rel = [per_slot[s][n]["rel_pert"] for s in per_slot
                   if per_slot[s][n]["rel_pert"] is not None]
            rows.append({
                "seed": seed, "n": n,
                "fro_sum": fs, "fro_indiv_l2": fi,
                "kappa": (fs / fi) if fi else None,
                "fro_mean": fs / n,
                "rel_pert_global": (fs / bs) if bs else None,
                "rel_pert_slot_mean": (float(np.mean(rel)) if rel else None),
                "rel_pert_slot_max": (float(np.max(rel)) if rel else None),
                "growth_vs_n1": fs / math.sqrt(acc[1]["sum_sq"]) if 1 in acc and acc[1]["sum_sq"] else None,
                "sqrt_n": math.sqrt(n),
            })
            print(f"[norms] seed {seed} N={n:>4}: ||sum||={fs:.4f} kappa={rows[-1]['kappa']:.4f} "
                  f"rel_pert(mean slot)={rows[-1]['rel_pert_slot_mean']}")

    out = {
        "config": os.path.abspath(cfg.get("_path", "")), "model_name": cfg["model_name"],
        "shards_dir": shards_dir, "n_values": n_values, "seeds": seeds,
        "script_sha256": _script_sha(), "git_hash": _git_hash(),
        "rows": rows, "per_slot": {s: {str(k): v for k, v in d.items()} for s, d in per_slot.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[norms] wrote {out_json}")
    return out


def do_overlap(cfg, out_json):
    """Per-N subset geometry (seed0 only): mean pairwise cosine + col(B)/row(A) principal-angle
    overlap + shared-basis energy, and each PROBE adapter's row-mean col(B) overlap with its
    co-merged set (feeds fig4b / H3). Subsample above `subsample` for cost (means are unbiased)."""
    from jd_collection import build_collection_slots
    from subspace_overlap import (pairwise_cosine, principal_angle_cos,
                                  shared_subspace_energy, _offdiag_mean, _random_slots)

    ov = cfg.get("overlap", {})
    seed = cfg["subset_seeds"][0]
    n_probes = cfg.get("n_probes", 5)
    cap = ov.get("subsample", 48)
    rank = ov.get("rank", 32)
    results = []
    for n in ov.get("n_values", [v for v in cfg["n_ladder"] if v >= 2]):
        authors = subset_authors(seed, n)
        probes = probe_authors(seed, n, n_probes)
        if n > cap:
            rng = np.random.RandomState(seed + n)
            extra = [a for a in authors if a not in probes]
            keep = probes + rng.choice(extra, size=cap - len(probes), replace=False).tolist()
        else:
            keep = authors
        dirs = [os.path.join(cfg["shards_dir"], f"shard_{a}") for a in keep]
        slots, ids, _ = build_collection_slots(dirs, device="cpu")
        cos = pairwise_cosine(slots)
        angB, angA = principal_angle_cos(slots)
        energy = shared_subspace_energy(slots, rank)
        null = _random_slots(slots, "orthogonal", seed)
        nullB, _ = principal_angle_cos(null)
        probe_rowmean = {}
        for a in probes:
            i = keep.index(a)
            row = angB[i].clone()
            row[i] = float("nan")
            probe_rowmean[str(a)] = float(np.nanmean(row.numpy()))
        results.append({
            "n": n, "sampled": len(keep), "authors_sampled": keep,
            "cosine_offdiag_mean": _offdiag_mean(cos),
            "angB_offdiag_mean": _offdiag_mean(angB),
            "angA_offdiag_mean": _offdiag_mean(angA),
            "angB_null_orth_mean": _offdiag_mean(nullB),
            "shared_energy_mean": energy["mean_energy_retained"],
            "shared_energy_chance": energy["avg_rank_ratio"],
            "probe_angB_rowmean": probe_rowmean,
        })
        print(f"[overlap] N={n} (sampled {len(keep)}): cos={results[-1]['cosine_offdiag_mean']:.4f} "
              f"angB={results[-1]['angB_offdiag_mean']:.4f} (null {results[-1]['angB_null_orth_mean']:.4f}) "
              f"energy@r{rank}={results[-1]['shared_energy_mean']:.4f} "
              f"(chance {results[-1]['shared_energy_chance']:.4f})")
    out = {"subset_seed": seed, "rank": rank, "subsample_cap": cap,
           "shards_dir": cfg["shards_dir"], "per_n": results}
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[overlap] wrote {out_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write merge + eval manifests, print permutations")
    p.add_argument("--config", required=True)

    m = sub.add_parser("merge", help="materialize ONE subset merge as a PEFT adapter dir")
    m.add_argument("--config", required=True)
    m.add_argument("--method", choices=["additive_mean", "additive_sum", "dare_ties",
                                        "centered_pool", "centered_lowrank"])
    m.add_argument("--n", type=int)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--svd_rank", type=int, default=None)
    m.add_argument("--rho", type=int, default=None,
                   help="centered_lowrank: rank of the shared-component SVD of the subset mean")
    m.add_argument("--lam", default=None,
                   help="additive_sum: GLOBAL per-adapter coefficient. Omit/1.0 = the literal "
                        "unit sum; 'isqrt' = 1/sqrt(N) (matched-norm arm); or a float. A FIXED "
                        "lambda keeps drop-a-term exact.")
    m.add_argument("--authors", default=None,
                   help="explicit author set ('180-199' or '180,181,195'), bypassing the nested "
                        "permutation. Requires --label.")
    m.add_argument("--label", default=None,
                   help="override the derived merge label (required with --authors)")
    m.add_argument("--cross_check", action="store_true",
                   help="run the config's cross_check block instead of --method/--n")
    m.add_argument("--force", action="store_true", help="rebuild even if the dir exists")

    o = sub.add_parser("overlap", help="per-N subset geometry stats (CPU)")
    o.add_argument("--config", required=True)
    o.add_argument("--out", default=None)

    nm = sub.add_parser("norms", help="per-N delta-norm ladder: ||sum||_F, kappa, rel_pert (CPU)")
    nm.add_argument("--config", required=True)
    nm.add_argument("--out", default=None)

    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "plan":
        do_plan(cfg, args.config)
    elif args.cmd == "merge":
        if not args.cross_check and args.method is None:
            raise SystemExit("merge needs --method (or --cross_check)")
        if not args.cross_check and args.n is None and not args.authors:
            raise SystemExit("merge needs --n (or --authors, or --cross_check)")
        do_merge(cfg, args.config, args)
    elif args.cmd == "overlap":
        out = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reports",
            f"nmerge_overlap_s{cfg['subset_seeds'][0]}.json")
        do_overlap(cfg, out)
    elif args.cmd == "norms":
        cfg["_path"] = args.config
        out = args.out or cfg.get("norms", {}).get("out") or os.path.join(
            "reports", f"nmerge_norms_s{cfg['subset_seeds'][0]}.json")
        # A relative `norms.out` resolves against the REPO, not the caller's cwd — so the config
        # stays portable (no /home/<user> prefix) and a job that runs from anywhere still writes
        # to the checked-out reports/ dir.
        if not os.path.isabs(out):
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out)
        do_norms(cfg, out)


if __name__ == "__main__":
    main()
