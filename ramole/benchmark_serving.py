"""E5 / Stage 3 — serving throughput: RAMoLE's fused heterogeneous batch vs LegoNet merges.

WHY: RAMoLE's serving pitch is that ONE batched forward can serve samples with DIFFERENT
expert sets (`set_routing`'s union + -inf mask), while the LegoNet production path must
`add_weighted_adapter`-merge once per distinct adapter-set and generate each sub-batch
separately. This benchmark quantifies that in tokens/sec on the SAME heterogeneous prompt
batch — per-sample expert sets taken from the frozen assignment, records chosen so the
top-k sets genuinely DIFFER — against the single-LoRA parity target:

  (a) ramole_batched   RamoleModel + set_routing(batch_idlists), one batched greedy generate
  (b) merge_per_group  LegoNet 1/k: group by identical adapter-set, activated(idxs)+generate
  (c) single_expert    LegoNet activated([j]) with one expert (single-LoRA throughput bound)

Protocol: greedy decode of --gen_tokens; 1 discarded warmup + --iters timed passes;
perf_counter around generate ONLY (cuda-synchronized; the per-group merge in (b) is set-up
cost outside the timed region — deliberately generous to the baseline); tokens/s counts
actually-generated tokens (output length minus input length — eos may stop early). Each
model is loaded ONCE outside timing, and the two 3B models are never held simultaneously:
(a) runs fully, is freed, then the LegoNet pool serves (b)+(c).

    python benchmark_serving.py --config configs/ramole_l32_3b.json \
        --batch_sizes 1 4 8 16 --iters 3 --gen_tokens 64 --device cuda --out PATH
    python benchmark_serving.py --smoke     # CPU fixture: batch_sizes 1 2, gen_tokens 4
"""
import argparse
import gc
import json
import os
import sys
import time

import torch

import ramole_common as rc  # also puts legonet_lora on sys.path (combine, eval imports)


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

THIS = os.path.dirname(os.path.abspath(__file__))

MODES = ("ramole_batched", "merge_per_group", "single_expert")


# ── batch selection (heterogeneous by construction) ────────────────────────────

def assignment_sets(cfg: dict, records: list[dict]) -> dict:
    """record_id -> tuple(top-k expert ids) from the FROZEN assignment (routing is not what
    is being timed, so we use the cached exact sets rather than re-embedding)."""
    with open(rc.source_paths(cfg).assignment_path) as f:
        r2k = json.load(f)["record_to_keys"]
    return {r["id"]: tuple(int(j) for j in r2k[r["id"]][: cfg["k"]]) for r in records}


def select_prompts(records: list[dict], sets: dict, max_b: int) -> list[dict]:
    """Deterministic greedy pick (corpus order): one record per DISTINCT expert set first,
    then fill with duplicates — so every prefix batch is maximally heterogeneous."""
    chosen, seen = [], set()
    for r in records:
        if sets[r["id"]] not in seen:
            seen.add(sets[r["id"]])
            chosen.append(r)
            if len(chosen) == max_b:
                return chosen
    if max_b > 1 and len(seen) < 2:
        raise RuntimeError("assignment yields one expert set — batch cannot be heterogeneous")
    have = {r["id"] for r in chosen}
    for r in records:
        if r["id"] not in have:
            chosen.append(r)
            if len(chosen) == max_b:
                return chosen
    raise RuntimeError(f"corpus has {len(records)} records < max batch size {max_b}")


# ── timing core ────────────────────────────────────────────────────────────────

def _gen_once(model, enc, gen_tokens: int, pad_id: int, cuda: bool):
    """One greedy generate — the ONLY timed region. Returns (seconds, generated_tokens, out).
    Token count = rows × (output len − padded input len); eos may stop the batch early."""
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=pad_id)
    if cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, int((out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0]), out


def _agg(times: list, toks: list) -> dict:
    """Index 0 is the discarded warmup; std is population std over the timed iters."""
    ts, ks = times[1:], toks[1:]
    mean = sum(ts) / len(ts)
    std = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5
    return {"tokens_per_s": (sum(ks) / len(ks)) / mean, "seconds_mean": mean, "seconds_std": std}


def _encode(tok, recs, device):
    # Decoder-only batched generate needs LEFT padding: right-pad would make generation
    # continue from pad tokens. Set locally on the served tokenizer (both loaders default
    # to the model's stock padding_side).
    tok.padding_side = "left"
    prompts = [rc.prompt_completion(r)[0] for r in recs]
    return tok(prompts, return_tensors="pt", padding=True).to(device)


def _grab(capture, tok, recs, enc, out):
    if capture is not None:
        in_len = enc["input_ids"].shape[1]
        for r, row in zip(recs, out):
            capture[r["id"]] = tok.decode(row[in_len:], skip_special_tokens=True)


# ── the three serving modes ────────────────────────────────────────────────────

def bench_ramole_batched(rm, recs, sets, gen_tokens, iters, cuda, capture=None) -> dict:
    idlists = [sets[r["id"]] for r in recs]
    rm.set_routing(idlists)                      # per-sample sets → union + -inf mask
    enc = _encode(rm.tokenizer, recs, rm.model.device)
    times, toks = [], []
    for i in range(iters + 1):
        dt, nt, out = _gen_once(rm.model, enc, gen_tokens, rm.tokenizer.pad_token_id, cuda)
        times.append(dt)
        toks.append(nt)
        if i == 0:
            _grab(capture, rm.tokenizer, recs, enc, out)
    return {**_agg(times, toks), "union_size": len({j for ids in idlists for j in ids})}


def bench_merge_per_group(lego, recs, sets, gen_tokens, iters, cuda, capture=None) -> dict:
    groups = {}                                  # insertion-ordered → deterministic
    for r in recs:
        groups.setdefault(sets[r["id"]], []).append(r)
    times, toks = [0.0] * (iters + 1), [0] * (iters + 1)
    for gset, grecs in groups.items():
        enc = _encode(lego.tokenizer, grecs, lego.model.device)
        with lego.activated(list(gset)) as m:    # 1/k merge = per-group set-up, NOT timed
            for i in range(iters + 1):
                dt, nt, out = _gen_once(m, enc, gen_tokens, lego.tokenizer.pad_token_id, cuda)
                times[i] += dt
                toks[i] += nt
                if i == 0:
                    _grab(capture, lego.tokenizer, grecs, enc, out)
    return {**_agg(times, toks), "union_size": len({j for s in groups for j in s}),
            "n_groups": len(groups)}


def bench_single_expert(lego, recs, expert_j, gen_tokens, iters, cuda, capture=None) -> dict:
    enc = _encode(lego.tokenizer, recs, lego.model.device)
    times, toks = [], []
    with lego.activated([expert_j]) as m:
        for i in range(iters + 1):
            dt, nt, out = _gen_once(m, enc, gen_tokens, lego.tokenizer.pad_token_id, cuda)
            times.append(dt)
            toks.append(nt)
            if i == 0:
                _grab(capture, lego.tokenizer, recs, enc, out)
    return {**_agg(times, toks), "union_size": 1}


# ── driver ─────────────────────────────────────────────────────────────────────

def run_benchmark(cfg, batch_sizes, gen_tokens, iters, device="cpu", out=None,
                  capture_outputs=False) -> dict:
    rc.set_determinism(cfg["base_seed"])
    cuda = device != "cpu" and torch.cuda.is_available()
    records = rc.load_records(rc.source_paths(cfg).records_path)
    sets = assignment_sets(cfg, records)
    chosen = select_prompts(records, sets, max(batch_sizes))
    batches = {b: chosen[:b] for b in sorted(batch_sizes)}

    comp = {}                                    # group-composition stats per batch size
    for b, recs in batches.items():
        ss = [sets[r["id"]] for r in recs]
        gsz = {}
        for s in ss:
            gsz[s] = gsz.get(s, 0) + 1
        comp[str(b)] = {"record_ids": [r["id"] for r in recs],
                        "expert_sets": [list(s) for s in ss],
                        "n_distinct_sets": len(gsz),
                        "group_sizes": sorted(gsz.values(), reverse=True),
                        "union_size": len({j for s in ss for j in s})}

    result = {"config": cfg["name"], "config_hash": rc.config_hash(cfg),
              "seed": cfg["base_seed"], "device": device, "gen_tokens": gen_tokens,
              "iters": iters, "batch_sizes": sorted(batch_sizes), "batches": comp,
              "modes": {m: {} for m in MODES}}
    caps = {m: {} for m in MODES} if capture_outputs else {m: None for m in MODES}

    # (a) RAMoLE fully first, then free — never hold both 3B models simultaneously.
    from ramole_model import RamoleModel
    rm = RamoleModel.from_config(cfg, device=device, load_router=True)
    for b, recs in batches.items():
        cap = caps["ramole_batched"].setdefault(str(b), {}) if capture_outputs else None
        result["modes"]["ramole_batched"][str(b)] = bench_ramole_batched(
            rm, recs, sets, gen_tokens, iters, cuda, capture=cap)
    del rm
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    # (b)+(c) share one LegoNet pool. Parity expert j = first record's top-1 (deterministic).
    from combine import LegoNetModel
    from eval_ramole import _source_lego_cfg
    lego = LegoNetModel.from_config(_source_lego_cfg(cfg),
                                    device_map=("auto" if device != "cpu" else "cpu"))
    result["single_expert_id"] = int(sets[chosen[0]["id"]][0])
    for b, recs in batches.items():
        capg = caps["merge_per_group"].setdefault(str(b), {}) if capture_outputs else None
        caps_ = caps["single_expert"].setdefault(str(b), {}) if capture_outputs else None
        result["modes"]["merge_per_group"][str(b)] = bench_merge_per_group(
            lego, recs, sets, gen_tokens, iters, cuda, capture=capg)
        result["modes"]["single_expert"][str(b)] = bench_single_expert(
            lego, recs, result["single_expert_id"], gen_tokens, iters, cuda, capture=caps_)
    if capture_outputs:
        result["outputs"] = caps

    print("\n| batch | union | groups | " + " | ".join(f"{m} tok/s" for m in MODES) + " |")
    print("|---:|---:|---:|" + "---:|" * len(MODES))
    for b in result["batch_sizes"]:
        cells = [str(b), str(comp[str(b)]["union_size"]),
                 str(result["modes"]["merge_per_group"][str(b)]["n_groups"])]
        cells += [f"{result['modes'][m][str(b)]['tokens_per_s']:.2f}" for m in MODES]
        print("| " + " | ".join(cells) + " |")
    if out:
        rc.write_json(out, result)
        print(f"[benchmark_serving] -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ramole_l32_3b.json")
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gen_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU fixture run: batch_sizes 1 2, gen_tokens 4, iters 1")
    args = ap.parse_args()

    if args.smoke:
        import tempfile
        sys.path.insert(0, os.path.join(THIS, "tests"))
        from _fixture import build_corpus_and_routing, build_source_run
        tmp = tempfile.mkdtemp(prefix="ramole_bench_")
        cfg = build_source_run(tmp, n=4, hidden=64, layers=2, heads=4, kv_heads=2,
                               rank=4, alpha=8, with_tokenizer=True)
        cfg["name"] = "ramole_bench_smoke"
        build_corpus_and_routing(cfg, per_cluster=8, device="cpu")
        args.batch_sizes, args.gen_tokens, args.iters, args.device = [1, 2], 4, 1, "cpu"
    else:
        cfg = rc.load_config(args.config)
        os.environ["HF_HOME"] = cfg["hf_home"]

    out = args.out or os.path.join(rc.Paths(cfg).results_dir, "serving_benchmark.json")
    run_benchmark(cfg, args.batch_sizes, args.gen_tokens, args.iters, args.device, out=out)


if __name__ == "__main__":
    if "--smoke" in sys.argv:  # smoke must never touch login-node GPUs (CUDA init is lazy)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
