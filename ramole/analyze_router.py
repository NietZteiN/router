"""E2 — alpha-weight diagnostics for the RouterLoRA cross-attention.

WHY: the whole point of RAMoLE over LegoNet is the *learned* per-layer alpha replacing the
uniform 1/k delta-average. E2 asks two falsifiable questions: (1) does the trained router
actually deviate from uniform 1/m (normalized entropy H < 1, max-share > 1/m), and (2) does
per-record routing sharpness (1-H) predict memorization (EM / -ln ppl)? If alpha stays
~uniform, RAMoLE's gains cannot come from routing and the composition story collapses to 1/k.

Mechanics: `RouterController.capture_alpha` (opt-in, zero-cost when off) makes every
RouterLoraLinear append `(active_idx (m,), alpha (m,b,l))` to `controller.captured[path]`.
We capture exactly ONE teacher-forced b=1 forward per record — NEVER `generate`: KV-cache
decode steps see l=1, so position pooling over the completion would be meaningless — pool the
stats online, and discard the raw tensors before the next record.

This module is both the shared library (capture_for_records / alpha_stats / pooling /
correlations / write_report — reused by tofu_sisa_lora/analyze_router_tofu.py) and the
DBpedia runner:

    python analyze_router.py --config configs/ramole_l32_3b.json --route keys \
        --n_eval 200 --device cuda --out .../results/alpha_diag_keys.json
    python analyze_router.py --config configs/ramole_l32_3b_d0.json --route keys \
        --router_ckpt .../runs/ramole_l32_3b_d0/router.safetensors ...   # ablation router
    python analyze_router.py --report ".../results/alpha_diag_*.json" --out ALPHA_REPORT.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

import ramole_common as rc
import router_lora as R

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

PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
_PATH_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.([qkvo]_proj)$")


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    return float(x[good].mean()) if good.any() else float("nan")


def _nanstd(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    return float(x[good].std()) if good.any() else float("nan")


# ── Per-record alpha statistics ────────────────────────────────────────────────

def alpha_stats(captured: dict, active, ideal_expert, prompt_len: int = 0) -> dict:
    """Pool one record's captured alphas. Per path (layer i, proj in q/k/v/o) and token
    position t: normalized entropy H = -sum_i a_i ln a_i / ln m (uniform -> 1; m==1 -> 1.0),
    max-share max_i a_i, and ideal-mass = alpha on `ideal_expert`, indexed by its POSITION
    within `active` (NaN + counted absent when not routed). Positions >= prompt_len are the
    completion; their H values feed a 10-decile position profile."""
    active = [int(j) for j in active]
    m = len(active)
    ideal_present = None
    ideal_pos = None
    if ideal_expert is not None:
        ideal_present = int(ideal_expert) in active
        ideal_pos = active.index(int(ideal_expert)) if ideal_present else None

    rows = []           # (layer, proj, H (l,), max_share (l,), ideal (l,))
    n_layers = 0
    for path in sorted(captured):
        entries = captured[path]
        mo = _PATH_RE.search(path)
        if mo is None:
            raise ValueError(f"cannot parse layer/proj from captured path {path!r}")
        if len(entries) != 1:
            raise ValueError(
                f"{path}: {len(entries)} captured forwards for one record — clear "
                "controller.captured per record and never capture generate() (KV-cache l=1)")
        act, alpha = entries[0]
        if [int(j) for j in act.tolist()] != active:
            raise ValueError(f"{path}: captured active {act.tolist()} != routed set {active}")
        if alpha.dim() != 3 or alpha.shape[0] != m or alpha.shape[1] != 1:
            raise ValueError(f"{path}: alpha shape {tuple(alpha.shape)} != (m={m}, b=1, l)")
        a = alpha[:, 0, :].to(torch.float64).numpy()               # (m, l)
        if m == 1:
            H = np.ones(a.shape[1])                                # softmax over 1 expert
        else:
            H = -np.where(a > 0, a * np.log(np.clip(a, 1e-300, None)), 0.0).sum(0) / np.log(m)
        maxs = a.max(0)
        ideal = a[ideal_pos] if ideal_pos is not None else np.full(a.shape[1], np.nan)
        layer, proj = int(mo.group(1)), mo.group(2)
        n_layers = max(n_layers, layer + 1)
        rows.append((layer, proj, H, maxs, ideal))

    per_layer = {p: [float("nan")] * n_layers for p in PROJS}
    dec_vals: list[list[float]] = [[] for _ in range(10)]
    H_all, max_all, ideal_all = [], [], []
    for layer, proj, H, maxs, ideal in rows:
        per_layer[proj][layer] = float(H.mean())
        H_all.append(H); max_all.append(maxs); ideal_all.append(ideal)
        n_comp = len(H) - prompt_len
        if n_comp > 0:
            for t in range(prompt_len, len(H)):
                d = min((t - prompt_len) * 10 // n_comp, 9)
                dec_vals[d].append(float(H[t]))

    H_all = np.concatenate(H_all)
    return {
        "m": m,
        "H_norm_mean": float(H_all.mean()),
        "H_norm_std": float(H_all.std()),
        "max_share_mean": float(np.concatenate(max_all).mean()),
        "ideal_mass_mean": _nanmean(np.concatenate(ideal_all)),
        "ideal_present": ideal_present,
        "per_layer": per_layer,
        "per_position_decile": [_nanmean(v) if v else float("nan") for v in dec_vals],
    }


# ── Capture loop (teacher-forced, b=1, online pooling) ─────────────────────────

def capture_for_records(rm, records, sets, max_length, text_fn, ideals=None):
    """Generator: per record, activate its routed set, capture alphas over ONE teacher-forced
    b=1 forward (labels=input_ids -> per-record NLL from out.loss), pool via alpha_stats, then
    discard the raw tensors. `rm` needs .model/.tokenizer/.controller (RamoleModel or the TOFU
    RamoleTofuModel — the latter has no .set_active, so we drive the controller directly, and
    we forward through .model to bypass the TOFU wrapper's internal re-routing).
    text_fn(rec) -> (full_text, prompt_text); the prompt is tokenized separately so
    completion positions (>= prompt token length) can be pooled by decile."""
    model, tok, ctrl = rm.model, rm.tokenizer, rm.controller
    device = next(model.parameters()).device
    for rec in records:
        rid = rec["id"]
        active = tuple(int(j) for j in sets[rid])
        if hasattr(rm, "set_active"):
            rm.set_active(active)
        else:
            ctrl.set_active(active)
        text, prompt = text_fn(rec)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to(device)
        p_len = int(tok(prompt, return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids.shape[1])
        p_len = max(0, min(p_len, input_ids.shape[1] - 1))   # keep >=1 completion position
        ctrl.captured.clear()
        ctrl.capture_alpha = True
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids,
                            attention_mask=enc.attention_mask.to(device), labels=input_ids)
        finally:
            ctrl.capture_alpha = False
        stats = alpha_stats(ctrl.captured, active,
                            None if ideals is None else ideals.get(rid), prompt_len=p_len)
        ctrl.captured.clear()   # free the raw (m,1,l) tensors before the next record
        stats.update({"id": rid, "nll": float(out.loss), "active": [int(j) for j in active],
                      "prompt_len": p_len, "seq_len": int(input_ids.shape[1])})
        yield stats


# ── Pooling across records + correlations + result assembly ────────────────────

def _spearman(x, y) -> dict:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(good.sum())}
    rho, p = spearmanr(x[good], y[good])
    return {"rho": float(rho), "p": float(p), "n": int(good.sum())}


def assemble_result(stats: list[dict], *, config, router_ckpt, route, m, extras=None) -> dict:
    """stats = capture_for_records output with 'em'/'ppl' already joined per record."""
    present = [s["ideal_present"] for s in stats if s["ideal_present"] is not None]
    pooled = {
        "H_norm_mean": _nanmean([s["H_norm_mean"] for s in stats]),
        "H_norm_std": _nanstd([s["H_norm_mean"] for s in stats]),
        "max_share_mean": _nanmean([s["max_share_mean"] for s in stats]),
        "ideal_mass_mean": _nanmean([s["ideal_mass_mean"] for s in stats]),
        "ideal_present_rate": (float(np.mean(present)) if present else float("nan")),
    }
    n_layers = max(len(s["per_layer"][PROJS[0]]) for s in stats)
    per_layer = {}
    for proj in PROJS:
        arr = np.full((len(stats), n_layers), np.nan)
        for i, s in enumerate(stats):
            v = s["per_layer"][proj]
            arr[i, :len(v)] = v
        per_layer[proj] = [_nanmean(arr[:, l]) for l in range(n_layers)]
    dec = np.array([s["per_position_decile"] for s in stats], dtype=np.float64)
    sharp = [1.0 - s["H_norm_mean"] for s in stats]
    neglogppl = [-np.log(s["ppl"]) if np.isfinite(s.get("ppl", np.nan)) and s["ppl"] > 0
                 else float("nan") for s in stats]
    out = {
        "config": config, "router_ckpt": router_ckpt, "route": route,
        "n_records": len(stats), "m": int(m),
        "pooled": pooled,
        "per_layer": per_layer,
        "per_position_decile": [_nanmean(dec[:, d]) for d in range(10)],
        "per_record": [{"id": s["id"], "H_norm": s["H_norm_mean"],
                        "max_share": s["max_share_mean"], "ideal_mass": s["ideal_mass_mean"],
                        "nll": s["nll"], "em": s.get("em", float("nan")),
                        "ppl": s.get("ppl", float("nan")), "active": s["active"]}
                       for s in stats],
        "correlations": {
            "spearman_sharpness_em": _spearman(sharp, [s.get("em", np.nan) for s in stats]),
            "spearman_sharpness_neglogppl": _spearman(sharp, neglogppl),
        },
    }
    out.update(extras or {})
    return out


# ── DBpedia runner ─────────────────────────────────────────────────────────────

def run_dbpedia(config_path, route, n_eval, device, router_ckpt=None, out=None) -> dict:
    cfg = rc.load_config(config_path)
    os.environ["HF_HOME"] = cfg["hf_home"]
    rc.set_determinism(cfg["base_seed"])
    sp = rc.source_paths(cfg)
    records = rc.load_records(sp.records_path)[:n_eval]

    import eval_ramole
    sets = eval_ramole.routed_sets(cfg, records, route, cfg["k"], False, device)
    with open(sp.assignment_path) as f:
        r2k = json.load(f)["record_to_keys"]
    ideals = {r["id"]: int(r2k[r["id"]][0]) for r in records}   # own-cluster = ideal expert

    from ramole_model import RamoleModel
    if router_ckpt:   # ablation router (e.g. d0) served through this config's expert pool
        rm = RamoleModel.from_config(cfg, device=device, load_router=False)
        R.load_router(rm.model, router_ckpt)
    else:
        rm = RamoleModel.from_config(cfg, device=device, load_router=True)
        # build_ramole_model silently skips loading when router.safetensors is absent; for
        # this diagnostic that is a SILENT FAILURE (a random-init router reads as H_norm~1.0,
        # i.e. exactly the "router stayed uniform" conclusion E2 exists to test) — raise.
        if "router_loaded_from" not in rm.meta:
            raise RuntimeError(
                f"no trained router at {rc.Paths(cfg).router_path} — alpha diagnostics on a "
                "random-init router would read as uniform; train it or pass --router_ckpt")

    def text_fn(rec):
        prompt, completion = rc.prompt_completion(rec)
        return prompt + completion, prompt

    stats = list(capture_for_records(rm, records, sets, cfg["train"]["max_length"],
                                     text_fn, ideals=ideals))

    # join em/perplexity from the matching eval result (rows keyed by id; NaN when absent)
    res_path = os.path.join(rc.Paths(cfg).results_dir, f"router_{route}_iid.json")
    by_id = {}
    if os.path.isfile(res_path):
        with open(res_path) as f:
            by_id = {row["id"]: row for row in json.load(f)["rows"]}
    else:
        print(f"[analyze_router] WARNING: no eval result at {res_path}; em/ppl -> NaN")
    for s in stats:
        row = by_id.get(s["id"], {})
        s["em"] = float(row.get("em", float("nan")))
        s["ppl"] = float(row.get("perplexity", float("nan")))

    result = assemble_result(
        stats, config=config_path, router_ckpt=router_ckpt, route=route, m=cfg["k"],
        extras={"name": cfg["name"], "dropout_p": cfg["dropout_p"], "seed": cfg["base_seed"],
                "joined_results": res_path if by_id else None})
    if out:
        rc.write_json(out, result)
        print(f"[analyze_router] -> {out}")
    print(f"[analyze_router] pooled={json.dumps(result['pooled'])} "
          f"corr={json.dumps(result['correlations'])}")
    return result


# ── Markdown report over one or more result JSONs ──────────────────────────────

def _fmt(v, nd=4):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def write_report(json_paths, out_md: str):
    runs = []
    for p in sorted(json_paths):
        with open(p) as f:
            runs.append(json.load(f))
    lines = ["# E2 — RouterLoRA alpha-weight diagnostics", "",
             f"{len(runs)} run(s). Uniform-routing anchors: H_norm = 1.0, max-share = 1/m.", "",
             "## Pooled sharpness vs the uniform anchor",
             "| run | route | dropout_p | m | n | H_norm (mean±std) | max_share (unif) "
             "| ideal_mass | ideal_present |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        po, m = r["pooled"], r["m"]
        lines.append(
            f"| {r.get('name', os.path.basename(r['config']))} | {r['route']} "
            f"| {r.get('dropout_p', 'n/a')} | {m} | {r['n_records']} "
            f"| {_fmt(po['H_norm_mean'])} ± {_fmt(po['H_norm_std'])} "
            f"| {_fmt(po['max_share_mean'])} (1/m={1.0 / m:.3f}) "
            f"| {_fmt(po['ideal_mass_mean'])} | {_fmt(po['ideal_present_rate'], 3)} |")
    lines += ["", "## Sharpness ↔ memorization (Spearman; sharpness = 1 − H_norm)",
              "| run | rho(EM) | p | n | rho(−ln ppl) | p | n |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        ce = r["correlations"]["spearman_sharpness_em"]
        cp = r["correlations"]["spearman_sharpness_neglogppl"]
        lines.append(f"| {r.get('name', r['config'])} | {_fmt(ce['rho'])} | {_fmt(ce['p'])} "
                     f"| {ce['n']} | {_fmt(cp['rho'])} | {_fmt(cp['p'])} | {cp['n']} |")
    lines += ["", "## Layer-depth profile (mean H_norm at first / middle / last layer)",
              "| run | proj | first | mid | last |", "|---|---|---|---|---|"]
    for r in runs:
        for proj in PROJS:
            v = r["per_layer"][proj]
            lines.append(f"| {r.get('name', r['config'])} | {proj} | {_fmt(v[0])} "
                         f"| {_fmt(v[len(v) // 2])} | {_fmt(v[-1])} |")
    lines += ["", "## Completion-position deciles (mean H_norm, prompt excluded)",
              "| run | " + " | ".join(f"d{d}" for d in range(10)) + " |",
              "|---|" + "---|" * 10]
    for r in runs:
        lines.append(f"| {r.get('name', r['config'])} | "
                     + " | ".join(_fmt(v) for v in r["per_position_decile"]) + " |")
    by_p = {}
    for r in runs:
        by_p.setdefault(r.get("dropout_p"), []).append(r)
    if 0.5 in by_p and 0.0 in by_p:   # the d0 ablation contrast (dropout is the OOD lever)
        lines += ["", "## Dropout contrast (train-time Random LoRA Dropout 0.5 vs 0)",
                  "| metric | p=0.5 | p=0.0 | Δ (0.5 − 0) |", "|---|---|---|---|"]
        for key in ("H_norm_mean", "max_share_mean", "ideal_mass_mean"):
            a = _nanmean([r["pooled"][key] for r in by_p[0.5]])
            b = _nanmean([r["pooled"][key] for r in by_p[0.0]])
            lines.append(f"| {key} | {_fmt(a)} | {_fmt(b)} | {_fmt(a - b)} |")
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[analyze_router] report ({len(runs)} runs) -> {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--route", choices=["keys", "retriever"], default="keys")
    ap.add_argument("--router_ckpt", default=None,
                    help="explicit router.safetensors (e.g. the d0 ablation); default = cfg's own")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None,
                    help="glob of alpha-diag result JSONs -> write the markdown report to --out")
    args = ap.parse_args()
    if args.report:
        paths = sorted(glob.glob(args.report))
        if not paths:
            raise SystemExit(f"--report matched no files: {args.report}")
        write_report(paths, args.out)
        return
    if not args.config:
        raise SystemExit("--config is required unless --report is given")
    run_dbpedia(args.config, args.route, args.n_eval, args.device,
                router_ckpt=args.router_ckpt, out=args.out)


if __name__ == "__main__":
    main()
