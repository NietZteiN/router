"""Per-author SIGNED contribution decomposition of a materialized uniform-aggregation merge.

Experiments B and C (log/merge_mechanism/, 2026-07-28). Answers, per query type:
  * which adapters are numerically active - owned vs unowned queries, original vs paraphrase;
  * whether that activity is DIFFUSE (every adapter contributes ~equally: a superposition) or
    CONCENTRATED (something is implicitly routing);
  * whether the contributions REINFORCE or CANCEL once summed into the residual stream.

Why not measure_key_firing.py (which already ran on this pool):
  1. it hooks a PLAIN base model, so its hidden states are not the ones the aggregate actually
     sees - and what the aggregation does to them is the whole question;
  2. it reports UNSIGNED norms ||s_i B_i A_i h||, so cancellation between authors is invisible;
  3. discover_adapters globs shard_<int>/ dirs and cannot read a merged adapter at all.
Its result remains the right baseline and is reused as such (gate_median 1.1018, verdict LAZY,
frac_ratio_lt_2 = 1.0), which is why --hidden base exists: same query groups, same block
decomposition, hidden states from the unadapted base = both tiers from one script.

THE DECOMPOSITION. merge_subset._weighted_factor_cat builds, per slot,
    A_cat = concat_i A_i            (rank = r*N, d_in)
    B_cat = concat_i w * s_i * B_i  (d_out, rank = r*N)
and write_effective_adapter forces PEFT scaling to exactly 1.0, so the served delta is
B_cat @ A_cat and author merge_meta.json["authors"][i] owns rows/cols [r*i : r*(i+1)].
With z = A_cat h and z_i the i-th block,
    c_i = B_cat[:, r*i:r*(i+1)] @ z_i     the author's signed contribution to the residual
    sum_i c_i == B_cat @ z == the module's exact delta output   (asserted, not assumed)
Norms come from one Gram G = B_cat^T B_cat per slot (never a dense d_out product):
    ||c_i||^2 = z_i^T G[i,i] z_i          ||sum_i c_i||^2 = z^T G z
so the cancellation index ||sum c_i|| / sum ||c_i|| is exact and costs one rank x rank
quadratic form per token.

WARNING - VALID ONLY FOR AN UNCOMPRESSED MERGE. merge_extra._compress_factored QRs both stacks
and SVDs the core, destroying per-author block identity. This script refuses any adapter whose
merge_meta.json records an svd_rank. At N <= 32 the cat rank is <= 1024 and merges are exact.

Usage:
    python measure_expb_contrib.py --model_name meta-llama/Llama-2-7B-chat-hf \
        --adapter ${TOFU_CKPT_ROOT}/.../merges/nmerge_sum_N20_s42 --hidden served \
        --out reports/expb/contrib_sum_N20.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

import eval_tofu as E
from jd_collection import _PREFIX
from merge_subset import _parse_author_list

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

ALPACA_TRAIN_HEAD = 2000  # keep OOD alpaca rows disjoint from any training negatives


def _script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _prompt(q):
    return f"Question: {q}\nAnswer:"   # same convention as eval_tofu._build_qa_prompt


def build_groups(args, tofu, authors, rng):
    """group_name -> [prompt], covering owned / unowned x original / paraphrase.

      own_orig_a{A} / own_para_a{A}          the aggregate's own authors, both surfaces
      unowned_orig_a{A} / unowned_para_a{A}  pool authors NOT in the merge (Exp C tier C1)
      holdout10                              never-trained TOFU-style QA (tier C2; verified
                                             0/400 question overlap with `full`, 2026-07-28)
      ood_world_facts / ood_real_authors / ood_alpaca   progressively further OOD (C3/C4)
    """
    groups = {}
    full = tofu["full"]

    # author -> perturbed rows, joined on question text (author a owns full rows [20a, 20a+20))
    q2a = {}
    for i, r in enumerate(full):
        q2a.setdefault(r["question"], i // 20)
    para_by_author = defaultdict(list)
    for r in tofu["forget10_pert"]:
        a = q2a.get(r["question"])
        if a is not None:
            para_by_author[a].append(r)

    def _author_groups(prefix, ids):
        for a in ids:
            rows = [full[i] for i in range(a * 20, a * 20 + 20)][: args.questions_per_author]
            groups[f"{prefix}_orig_a{a}"] = [_prompt(r["question"]) for r in rows]
            pr = para_by_author.get(a, [])[: args.questions_per_author]
            if pr:
                groups[f"{prefix}_para_a{a}"] = [_prompt(r["paraphrased_question"]) for r in pr]

    _author_groups("own", authors)
    pool = [a for a in range(200) if a not in set(authors)]
    un = rng.choice(pool, size=min(args.unowned_authors, len(pool)), replace=False)
    _author_groups("unowned", sorted(int(x) for x in un))

    if args.holdout_n:
        from datasets import load_dataset
        ho = load_dataset("locuslab/TOFU", "holdout10")["train"]
        idx = rng.choice(len(ho), size=min(args.holdout_n, len(ho)), replace=False)
        groups["holdout10"] = [_prompt(ho[int(i)]["question"]) for i in idx]

    for name, ds in (("ood_world_facts", tofu["world_facts"]),
                     ("ood_real_authors", tofu["real_authors"])):
        idx = rng.choice(len(ds), size=min(args.ood_n, len(ds)), replace=False)
        groups[name] = [_prompt(ds[int(i)]["question"]) for i in idx]
    try:
        from skill_data import load_alpaca
        pairs = load_alpaca(ALPACA_TRAIN_HEAD + args.ood_n, os.environ["HF_HOME"])
        groups["ood_alpaca"] = [_prompt(p["question"]) for p in pairs[ALPACA_TRAIN_HEAD:]]
    except Exception as e:  # alpaca is a nice-to-have, not a premise
        print(f"[contrib] WARN alpaca unavailable ({type(e).__name__}: {e}) - skipping")
    return groups


def load_blocks(adapter_dir):
    """slot -> (A_cat, B_cat), the author order, and per-author rank. Refuses a compressed
    merge, whose factor blocks no longer correspond to authors."""
    from safetensors.torch import load_file
    meta_p = os.path.join(adapter_dir, "merge_meta.json")
    if not os.path.exists(meta_p):
        raise SystemExit(f"{adapter_dir} has no merge_meta.json - cannot map blocks to authors")
    meta = json.load(open(meta_p))
    if meta.get("svd_rank"):
        raise SystemExit(
            f"{adapter_dir} was SVD-compressed (svd_rank={meta['svd_rank']}). _compress_factored "
            "QRs the stacks and SVDs the core, destroying per-author block identity - the "
            "decomposition would be meaningless. Re-materialize this N exactly.")
    authors = meta["authors"]
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    blocks = {}
    for k, v in tensors.items():
        if k.endswith(".lora_A.weight"):
            key = k[: -len(".lora_A.weight")]
            # stored keys carry PEFT's "base_model.model." prefix; the canonical slot name (what
            # _read_adapter yields and what a plain nn.Module is named by) is the bare remainder.
            slot = key[len(_PREFIX):] if key.startswith(_PREFIX) else key
            blocks[slot] = (v, tensors[key + ".lora_B.weight"])
    if not blocks:
        raise SystemExit(f"no lora_A/lora_B pairs in {adapter_dir}")
    rank = next(iter(blocks.values()))[0].shape[0]
    if rank % len(authors):
        raise SystemExit(f"cat rank {rank} not divisible by {len(authors)} authors")
    return blocks, authors, rank // len(authors), meta


class ContribHooks:
    """Capture each target module's INPUT h, then decompose the delta it produces."""

    def __init__(self, model, blocks, n_authors, r, device, dtype=torch.float32):
        self.h, self.handles, self.grams, self.A = {}, [], {}, {}
        self.n, self.r = n_authors, r
        named = dict(model.named_modules())
        self.slots = []
        for slot, (A, B) in blocks.items():
            mod = self._resolve(named, slot)
            if mod is None:
                continue
            self.slots.append(slot)
            A = A.to(device=device, dtype=dtype)
            B = B.to(device=device, dtype=dtype)
            self.A[slot] = A
            self.grams[slot] = B.t() @ B          # (rank, rank) - every norm comes from this
            self.handles.append(mod.register_forward_pre_hook(self._mk(slot)))
        if not self.slots:
            raise SystemExit("no adapter slot resolved against the model - check the name prefix")

    @staticmethod
    def _resolve(named, slot):
        # merged-adapter keys are bare ("model.layers.0.self_attn.q_proj"); a PeftModel nests
        # everything under "base_model.model.". Prefer the peft path - its lora.Linear pre-hook
        # sees exactly the h the base Linear would see.
        for cand in (f"base_model.model.{slot}", slot):
            if cand in named:
                return named[cand]
        return None

    def _mk(self, slot):
        def hook(_mod, inputs):
            self.h[slot] = inputs[0].detach()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()

    def decompose(self, attn_mask):
        """Per-author ||c_i|| and ||sum_i c_i||, mean over real tokens, summed over slots."""
        n, r = self.n, self.r
        acc_auth = torch.zeros(n, dtype=torch.float64)
        acc_tot = acc_sumnorm = 0.0
        m = attn_mask.bool()
        ntok = max(int(m.sum().item()), 1)
        for slot in self.slots:
            A, G = self.A[slot], self.grams[slot]
            z = torch.nn.functional.linear(self.h[slot].to(A.dtype), A)   # (B, T, rank)
            z = z[m]                                                      # real tokens only
            tot = torch.einsum('ti,ij,tj->t', z, G, z).clamp_min(0)       # ||sum_i c_i||^2
            acc_tot += float(tot.sum().item()) / ntok
            zb = z.view(-1, n, r)
            per = torch.empty(z.shape[0], n, dtype=z.dtype, device=z.device)
            for i in range(n):
                Gi = G[i * r:(i + 1) * r, i * r:(i + 1) * r]
                per[:, i] = torch.einsum('ti,ij,tj->t', zb[:, i], Gi, zb[:, i]).clamp_min(0)
            acc_auth += (per.sum(0) / ntok).double().cpu()
            acc_sumnorm += float(per.sqrt().sum(1).sum().item()) / ntok
        return acc_auth.numpy(), acc_tot, acc_sumnorm

    def exactness_check(self, blocks, attn_mask, device):
        """sum_i c_i MUST equal the module's exact delta B_cat @ A_cat @ h. If it does not, the
        author block map is wrong and every number here is meaningless. Dense, one slot, once."""
        slot = self.slots[0]
        A_cat, B_cat = blocks[slot]
        A_cat = A_cat.to(device=device, dtype=torch.float32)
        B_cat = B_cat.to(device=device, dtype=torch.float32)
        h = self.h[slot][attn_mask.bool()].to(torch.float32)
        z = torch.nn.functional.linear(h, A_cat)
        ref = torch.nn.functional.linear(z, B_cat)
        acc = torch.zeros_like(ref)
        for i in range(self.n):
            sl = slice(i * self.r, (i + 1) * self.r)
            acc += torch.nn.functional.linear(z[:, sl], B_cat[:, sl])
        return float((acc - ref).norm() / ref.norm().clamp_min(1e-30))


def diffuseness(v):
    """Concentration profile of a non-negative attribution vector. Same vocabulary as
    analyze_orphan_destinations.py, so these sit beside the router-leak orphan tables."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0 or not np.isfinite(s):
        return {"max_share": None, "top3_share": None, "h_norm": None, "hhi": None,
                "n_eff": None, "k": int(v.size)}
    p = v / s
    nz = p[p > 0]
    k = len(p)
    return {"max_share": float(p.max()),
            "top3_share": float(np.sort(p)[::-1][:3].sum()),
            "h_norm": float(-(nz * np.log(nz)).sum() / math.log(k)) if k > 1 else 1.0,
            "hhi": float((p ** 2).sum()),
            "n_eff": float(1.0 / (p ** 2).sum()),
            "k": k}


def summarize(results, n):
    def _mean_over(prefix, key=None):
        vals = [(r["diffuseness"][key] if key else r["cancellation_index"])
                for g, r in results.items() if g.startswith(prefix)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def _own_ratio(prefix):
        """median over groups of (that group's own author's norm) / (mean of the others')."""
        out = []
        for g, r in results.items():
            if not g.startswith(prefix) or "_a" not in g:
                continue
            a = g.rsplit("_a", 1)[1]
            pa = r["per_author_norm"]
            if a not in pa:
                continue
            others = [v for k, v in pa.items() if k != a]
            if others and np.mean(others) > 0:
                out.append(pa[a] / float(np.mean(others)))
        return float(np.median(out)) if out else None

    ood = [v for v in (_mean_over("ood_world_facts", "n_eff"),
                       _mean_over("ood_real_authors", "n_eff"),
                       _mean_over("ood_alpaca", "n_eff")) if v is not None]
    return {
        "own_orig_selectivity_median": _own_ratio("own_orig"),
        "own_para_selectivity_median": _own_ratio("own_para"),
        "n_eff_own_orig": _mean_over("own_orig", "n_eff"),
        "n_eff_own_para": _mean_over("own_para", "n_eff"),
        "n_eff_unowned_orig": _mean_over("unowned_orig", "n_eff"),
        "n_eff_holdout10": _mean_over("holdout10", "n_eff"),
        "n_eff_ood": float(np.mean(ood)) if ood else None,
        "cancel_own_orig": _mean_over("own_orig"),
        "cancel_unowned_orig": _mean_over("unowned_orig"),
        "cancel_holdout10": _mean_over("holdout10"),
        "n_authors": n,
        "orthogonal_cancel_expectation": 1.0 / math.sqrt(n),
        "_verdict_rule": (
            "selectivity < 2 on BASE hidden states => training artifact (the keys were never "
            "selective); >= 5 on base but < 2 on served => aggregation artifact. n_eff ~ n on "
            "unowned queries => the response is a superposition over all adapters, i.e. no "
            "implicit routing. cancellation_index ~ 1/sqrt(n) => contributions are mutually "
            "orthogonal so the injected perturbation grows as sqrt(N); ~1 => they reinforce."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter", required=True, help="materialized (UNCOMPRESSED) merge dir")
    ap.add_argument("--hidden", choices=("served", "base"), default="served",
                    help="served = hidden states of base+merge (the real measurement); "
                         "base = unadapted hidden states (the measure_key_firing tier)")
    ap.add_argument("--authors", default=None,
                    help="override the merge's own author list (default: merge_meta.json)")
    ap.add_argument("--unowned_authors", type=int, default=5)
    ap.add_argument("--questions_per_author", type=int, default=10)
    ap.add_argument("--holdout_n", type=int, default=100)
    ap.add_argument("--ood_n", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[contrib] skip existing {args.out}")
        return
    t0 = time.time()

    blocks, meta_authors, r, meta = load_blocks(args.adapter)
    authors = _parse_author_list(args.authors) if args.authors else meta_authors
    n = len(meta_authors)
    print(f"[contrib] {args.adapter}: {n} authors x rank {r} = {n * r}, {len(blocks)} slots")

    hf_home = os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
    tofu = E.load_tofu_data(hf_home)

    if args.hidden == "served":
        model, tok = E.load_single_adapter(args.model_name, args.adapter, adapter_name="contrib")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.eval()
    device = next(model.parameters()).device

    rng = np.random.default_rng(args.seed)
    groups = build_groups(args, tofu, authors, rng)
    print(f"[contrib] {len(groups)} groups, {sum(len(v) for v in groups.values())} prompts")

    hooks = ContribHooks(model, blocks, n, r, device)
    results, exact_err = {}, None
    with torch.no_grad():
        for gi, (gname, prompts) in enumerate(sorted(groups.items())):
            auth_sq = np.zeros(n)
            tot_sq = sumnorm = 0.0
            nb = 0
            for s in range(0, len(prompts), args.batch_size):
                enc = tok(prompts[s:s + args.batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
                model(**enc)
                if exact_err is None:   # once, on the first real batch
                    exact_err = hooks.exactness_check(blocks, enc["attention_mask"], device)
                    print(f"[contrib] exactness ||sum_i c_i - delta||/||delta|| = {exact_err:.3e}")
                    if exact_err > 1e-4:
                        raise SystemExit(
                            f"block decomposition does not reconstruct the delta (rel err "
                            f"{exact_err:.3e}) - the author block map is wrong, aborting")
                a, t, sn = hooks.decompose(enc["attention_mask"])
                auth_sq += a
                tot_sq += t
                sumnorm += sn
                nb += 1
            nb = max(nb, 1)
            auth_norm = np.sqrt(auth_sq / nb)
            tot_norm = math.sqrt(tot_sq / nb)
            sumnorm /= nb
            results[gname] = {
                "n_prompts": len(prompts),
                "per_author_norm": {str(a): float(v) for a, v in zip(meta_authors, auth_norm)},
                "agg_norm": tot_norm,
                "sum_of_author_norms": sumnorm,
                # <1 => destructive interference; ~1/sqrt(n) = mutually orthogonal; ~1 => reinforce
                "cancellation_index": (tot_norm / sumnorm) if sumnorm else None,
                "diffuseness": diffuseness(auth_norm),
            }
            if gi % 5 == 0:
                d = results[gname]["diffuseness"]
                print(f"[contrib] {gname:26} n_eff={d['n_eff']:.1f}/{n} "
                      f"max_share={d['max_share']:.4f} "
                      f"cancel={results[gname]['cancellation_index']:.4f}", flush=True)
    hooks.close()

    summary = summarize(results, n)
    out = {
        "adapter": args.adapter, "hidden": args.hidden, "model_name": args.model_name,
        "authors": meta_authors, "rank_per_author": r, "n_slots": len(hooks.slots),
        "merge_meta": {k: meta.get(k) for k in
                       ("label", "method", "n", "subset_seed", "weights", "lam", "lam_weight")},
        "seed": args.seed, "script_sha256": _script_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
        "decomposition_rel_err": exact_err,
        "summary": summary, "groups": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("[contrib] summary:", json.dumps(
        {k: v for k, v in summary.items() if not k.startswith("_")}, indent=2))
    print(f"[contrib] wrote {args.out}")


if __name__ == "__main__":
    main()
