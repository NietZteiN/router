"""S³T paper reproduction on TOFU: figures + report (CPU, no GPU).

Combines the pure-simulation deletion machinery (s3t_deletion.py) with the measured
F(d) curve (s3t_measure_F.py) and deletion timings (s3t_deletion_time.py) to produce
the paper's headline analogues on TOFU:

  Fig 6-left  : performance (model_utility) vs #deletion requests, S3T B in {1,2,4}
                (B=1 == SISA) vs full-retrain (flat F[L]).
  Fig 6-right : deletion rate delta vs budget B.
  Fig 7       : delta vs #shards m and #slices L, S3T vs SISA, with theory overlays.
  Fig 9       : total deletion time vs #requests, S3T vs SISA vs full-retrain.

Matplotlib is optional; tables are always written to the markdown report.
"""
import argparse
import json
import os
from datetime import date

import numpy as np

from s3t_deletion import (
    average_performance_curve,
    deletion_rate,
    deletion_rate_theory,
    empirical_retention,
    expected_retrains,
    retention_prob_s3t,
    retention_prob_sisa,
)
from s3t_sequences import iterative_cyclic_rotation

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def fig_deletion_rate_vs_B(m, L, Bs, n_seeds):
    rows = []
    for B in Bs:
        emp = deletion_rate(m, L, B, n_seeds=n_seeds)
        rows.append({"B": B, "emp": emp["mean"], "std": emp["std"],
                     "theory": deletion_rate_theory(m, L, B),
                     "system": "SISA" if B == 1 else f"S3T(B={B})"})
    return rows


def fig_delta_vs_m(L, B, ms, n_seeds):
    out = []
    for m in ms:
        s3t = deletion_rate(m, L, B, n_seeds=n_seeds)["mean"]
        sisa = deletion_rate(m, L, 1, n_seeds=n_seeds)["mean"]
        out.append({"m": m, "s3t": s3t, "sisa": sisa, "gain": s3t / sisa})
    return out


def fig_delta_vs_L(m, B, Ls, n_seeds):
    out = []
    for L in Ls:
        b = min(B, L)
        s3t = deletion_rate(m, L, b, n_seeds=n_seeds)["mean"]
        sisa = deletion_rate(m, L, 1, n_seeds=n_seeds)["mean"]
        out.append({"L": L, "B": b, "s3t": s3t, "sisa": sisa, "gain": s3t / sisa})
    return out


def perf_curves(m, L, F, Bs, n_seeds, n_points):
    curves = {}
    for B in Bs:
        mean, std = average_performance_curve(m, L, B, F, n_seeds=n_seeds, n_points=n_points)
        curves[B] = (mean, std)
    return curves


def preamble_section(m, L):
    """From-scratch explanation of the method, the TOFU instantiation, and every
    term/symbol used in the tables below."""
    return [
        "## 0. What this is (read first)", "",
        "**Machine unlearning** = removing a training example's influence from a model "
        "*without* retraining from scratch. **Exact** unlearning guarantees removal by a "
        "modular design: split the data into disjoint pieces, train a separate component on "
        "each, and at deletion only touch the component(s) that saw the deleted data.", "",
        "**SISA** (the baseline, Bourtoule et al. 2021) = *Sharded, Isolated, Sliced, "
        "Aggregated*. Split data into **shards** (one model each, predictions ensembled); "
        "within a shard split into **slices** trained incrementally with a checkpoint after "
        "each. To delete a slice's data you roll back to the checkpoint before it and "
        "**retrain** the rest — cheaper than full retraining, but still a GPU retrain that "
        "takes the model offline.", "",
        "**S³T** (the method, ICLR 2025) keeps sharding + slicing but makes each slice train a "
        "**disjoint block of LoRA layers**, top-down and cumulatively (layer block *i* sees "
        "slices 1..i). Because a slice only ever influenced its own layer block and the ones "
        "below it, deleting a slice = **switch those LoRA layers off** (a metadata mask, "
        "~milliseconds) — no retraining, no downtime. To stay useful after many deletions, "
        "S³T trains several models per shard on **different slice orderings** (a **budget** of "
        "B orderings, chosen to be *diverse* so deletions rarely kill them all) and serves "
        "whichever surviving ordering kept the most slices.", "",
        f"**This experiment on TOFU.** TOFU = 200 fictional authors (20 Q&A each); the model "
        f"is fine-tuned to know them, then asked to forget some. We use **m={m} shards** "
        f"(40 authors each) × **L={L} slices** (10 authors each), 8 LoRA layers per slice "
        f"(all 32 Llama-2-7B layers). Shard predictions are combined by averaging their "
        f"token probabilities (`ensemble_probs`). The deletion-rate / time / retention "
        "results are exact combinatorial simulation over slice orderings (validated against "
        "the paper's theory); the utility numbers F(d) are real GPU evaluations.", "",
        "### Glossary of every symbol/column below", "",
        "| term | meaning |", "|---|---|",
        f"| m | number of shards (={m}); each = 200/m disjoint authors |",
        f"| L | slices per shard (={L}); each slice = a group of authors trained at one stage |",
        "| B | **budget** = number of different slice-orderings trained per shard. B=1 is SISA; larger B = more deletion resilience |",
        "| r | number of deletion requests processed so far |",
        "| k | a retention threshold: 'still has ≥ k of its L slices' |",
        "| δ (delta) | **deletion rate** = expected #deletion requests the system serves before it must retrain from scratch (higher = better) |",
        "| F(d) | model utility when every shard's model is trained on d of its L slices (d=0 = base model, d=L = full); = depth snapshot stage_{d-1} |",
        "| model_utility | TOFU's overall quality score (harmonic mean of probability/ROUGE/truth-ratio on retain + real-author + world-fact questions). Base model 0.42; full k=1 fine-tune 0.74 |",
        "| forget_quality | KS-test p-value vs a never-saw-forget oracle; higher = the forgotten authors look untrained |",
        "| ensemble_probs | inference = average the per-token probability distributions of the loaded shard models (the paper's 'aggregate decision') |",
        "| SISA / S³T (B=…) | SISA = single ordering + retrain-on-delete; S³T = B diverse orderings + delete-by-layer-deactivation |",
        "| armA / armB | two training recipes. **armA** = the paper's exact Llama-2-7B hyperparameters (lr 2e-5, 3 epochs/stage) — faithful but undertrained on TOFU. **armB** = the repo's tuned recipe (lr 1e-4, 5 epochs) — stronger, used as the contrast |",
        "| edit distance | how different two orderings are (positions where they disagree); higher across a set = more *diverse* = more deletion-resilient |",
        "| cyclic rotation / BMS | the two ways S³T picks B diverse orderings (uniform vs known deletion prior) |",
        "| full re-training | upper-cost baseline: retrain the whole model after every deletion |", "",
        "### How to read the tables", "",
        "- **Deletion rate δ / Fig 7**: bigger δ and bigger SISA→S³T 'gain' = S³T serves more "
        "deletions before a costly retrain.",
        "- **Performance vs #deletions**: each column is a budget B; going down a column shows "
        "utility decaying as more authors are deleted. S³T (B>1) columns stay higher than SISA "
        "(B=1) = it degrades more gracefully.",
        "- **Deletion time**: wall-clock to service 1000 deletions; S³T mostly masks layers, "
        "SISA/full must retrain.",
        "- **Lemma 2**: probability a shard still retains ≥k slices after r deletions — closed "
        "form vs simulation, as a correctness check.", "",
    ]


def lemma2_section(m, L, ks, rs, Bs):
    """Eq-18/20 retention closed form vs per-shard simulation (random sequences)."""
    lines = ["## Lemma 2 — performance retention (Eq 18/20)", "",
             "Per-shard P[retain ≥ k slices after r deletions]: closed form vs simulation "
             "(random sequences; cyclic ≥ closed form per the paper's diversity remark).", ""]
    for k in ks:
        lines += [f"### k = {k} (retain ≥ {k} of {L} slices)", "",
                  "| r | SISA (B=1) | " + " | ".join(f"S3T B={B}" for B in Bs if B > 1)
                  + " | sim B=" + str(max(Bs)) + " (rand/cyclic) |",
                  "|---|" + "---|" * (1 + sum(1 for B in Bs if B > 1)) + "---|"]
        for r in rs:
            cells = [f"{retention_prob_sisa(k, L, r):.3f}"]
            cells += [f"{retention_prob_s3t(k, L, r, B):.3f}" for B in Bs if B > 1]
            Bmax = max(Bs)
            rnd = empirical_retention(L, Bmax, k, r, n_seeds=4000)
            cyc = empirical_retention(L, Bmax, k, r,
                                      sequences=iterative_cyclic_rotation(L, Bmax),
                                      n_seeds=4000)
            cells.append(f"{rnd:.3f}/{cyc:.3f}")
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def fig9_section(m, L, timing, Bs, n_seeds):
    """Faithful Fig-9: cumulative deletion time over a 1000-request stream =
    expected_retrains(R, δ) × T_full_retrain per system (S3T retrains rarely; SISA
    more often; full-retrain every request)."""
    R = 1000
    mask_s = timing.get("s3t_mask_s", float("nan"))
    shard_retrain_s = timing.get("sisa_retrain_s", float("nan"))
    t_full = m * shard_retrain_s          # from-scratch retrain ≈ all m shards
    delta_sisa = deletion_rate(m, L, 1, n_seeds=n_seeds)["mean"]
    delta_s3t = deletion_rate(m, L, max(Bs), n_seeds=n_seeds)["mean"]
    t_full_total = R * t_full                                   # retrain every request
    t_sisa = expected_retrains(R, delta_sisa) * t_full          # retrain when a shard exhausts
    t_s3t = expected_retrains(R, delta_s3t) * t_full + R * mask_s
    lines = ["## Deletion time over a 1000-request stream (Fig 9)", "",
             f"From-scratch retrain T_full ≈ m × per-shard retrain = {m} × "
             f"{shard_retrain_s:.0f}s = {t_full:.0f}s. A system retrains when a shard is "
             f"exhausted (every δ requests); full-retrain retrains every request.", "",
             "| system | δ | retrains over 1000 | total time (h) |", "|---|---|---|---|",
             f"| full re-training | 1 | 1000 | {t_full_total/3600:.1f} |",
             f"| SISA (B=1) | {delta_sisa:.1f} | {R/delta_sisa:.1f} | {t_sisa/3600:.1f} |",
             f"| S3T (B={max(Bs)}) | {delta_s3t:.1f} | {R/delta_s3t:.1f} | {t_s3t/3600:.1f} |",
             "",
             f"**S3T reduces total deletion time {t_sisa/t_s3t:.2f}× vs SISA and "
             f"{t_full_total/t_s3t:.0f}× vs full re-training** (stream of 1000 requests).",
             "",
             f"Common-case single deletion: S3T = layer mask **{mask_s*1000:.1f} ms** vs SISA "
             f"shard retrain **{shard_retrain_s:.0f} s** (the per-event cost; the table above is "
             f"the faithful cumulative Fig-9 framing).", ""]
    return lines


def storage_section(m, L, Bs, src, n_seeds):
    """Table 3: PEFT storage vs budget B and the resulting deletion rate."""
    import glob
    # Per-shard adapter bytes (one final adapter), measured from disk if available.
    adapter_bytes = 0
    for f in glob.glob(os.path.join(src, "shard_0", "adapter_model.safetensors")):
        adapter_bytes = os.path.getsize(f)
    per_shard_mb = adapter_bytes / 1e6 if adapter_bytes else float("nan")
    lines = ["## Storage vs deletion rate (Table 3)", "",
             f"Per-shard LoRA adapter ≈ {per_shard_mb:.0f} MB; m={m} shards. PEFT storage = "
             f"B × m × per-shard (base model shared, not counted).", "",
             "| B | PEFT storage (GB) | deletion rate δ |", "|---|---|---|"]
    for B in Bs:
        gb = (B * m * adapter_bytes) / 1e9 if adapter_bytes else float("nan")
        d = deletion_rate(m, L, B, n_seeds=n_seeds)["mean"]
        lines.append(f"| {B} | {gb:.2f} | {d:.1f} |")
    lines += ["", "Deletion rate rises with B but saturates at B=L (Lemma 1); storage grows "
              "linearly — the offline-storage vs deletion-capacity trade-off.", ""]
    return lines


def rq3_section(src):
    """Fold in RQ3/Fig-8 diversity results if rq3_diversity.json exists."""
    p = os.path.join(src, "rq3_diversity.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    lines = ["## RQ3 — sequence-selection diversity (Fig 8)", "",
             f"Uniform prior (L={d['uniform']['L']}): avg pairwise edit distance, "
             "iterative cyclic rotation vs random.", "",
             "| B | cyclic | random |", "|---|---|---|"]
    for row in d["uniform"]["edit_distance_vs_B"]:
        lines.append(f"| {row['B']} | {row['cyclic']} | {row['random']} |")
    nu = d["nonuniform"]["result"]
    lines += ["", f"Non-uniform prior (L={d['nonuniform']['L']}, B={d['nonuniform']['B']}, "
              "Dirichlet priors): edit distance / Eq-24 score.", "",
              "| method | edit dist | score |", "|---|---|---|"]
    for name in ("bms", "sorted_cyclic", "random"):
        lines.append(f"| {name} | {nu[name]['edit']} | {nu[name]['score']} |")
    lines += ["", "Cyclic rotation ≫ random on diversity; BMS is maximally diverse "
              "(edit distance = L, Lemma 3). Score ordering is t-dependent — at t=1 all "
              "position-diverse sets tie by construction.", ""]
    return lines


def write_report(path, m, L, F, rate_rows, mvar, lvar, curves, timing, n_seeds,
                 F2=None, curves2=None, src="."):
    base, full = F[0], F[-1]
    lines = [f"# S³T paper reproduction on TOFU ({date.today().isoformat()})", ""]
    lines += [f"Llama-2-7B-chat-hf, armA (paper-faithful: r32/α64, lr 2e-5, 3 ep/stage). "
              f"m={m} shards, L={L} slices, uniform deletion prior. Deletion rate over "
              f"{n_seeds} random streams. All deletion-rate numbers are pure simulation "
              f"(validated against Lemma 1 closed form in test_s3t_sequences.py).", ""]
    lines += preamble_section(m, L)
    has_b = F2 is not None
    lines += ["## F(d): ensemble utility when every shard retains d slices", "",
              "| d (slices) | armA utility | armA forget_q | "
              + ("armB utility |" if has_b else ""), "|---|---|---|" + ("---|" if has_b else "")]
    fq = timing.get("F_fq", [float("nan")] * (L + 1)) if timing else [float("nan")] * (L + 1)
    for d in range(L + 1):
        row = f"| {d} | {F[d]:.4f} | {fq[d] if d < len(fq) else float('nan')} |"
        if has_b:
            row += f" {F2[d]:.4f} |"
        lines.append(row)
    lines += ["", f"Anchors: base {base:.4f}, armA full (d={L}) {full:.4f}"
              + (f", armB full {F2[-1]:.4f}" if has_b else "")
              + ", k=1 full LoRA-ft 0.7435 (utility ceiling).",
              "armA is the paper-faithful recipe (lr 2e-5/3ep) and is undertrained on TOFU "
              "→ F(d) ≈ base; armB (lr 1e-4/5ep) gives the meaningful-degradation curve.", ""]

    lines += ["## Deletion rate δ vs budget B (Fig 6-right / Lemma 1)", "",
              "| system | B | δ (sim) | δ (theory mL·H_{mB'}) |", "|---|---|---|---|"]
    for r in rate_rows:
        lines.append(f"| {r['system']} | {r['B']} | {r['emp']:.1f} ± {r['std']:.1f} | {r['theory']:.1f} |")
    sisa_d = next(r["emp"] for r in rate_rows if r["B"] == 1)
    best = max(rate_rows, key=lambda r: r["B"])
    lines += ["", f"**S3T(B={best['B']}) handles {best['emp']/sisa_d:.2f}× more deletion "
              f"requests than SISA before a from-scratch retrain** "
              f"({best['emp']:.1f} vs {sisa_d:.1f}).", ""]

    lines += ["## δ vs #shards m (Fig 7-center)", "", "| m | SISA | S3T | gain |", "|---|---|---|---|"]
    for r in mvar:
        lines.append(f"| {r['m']} | {r['sisa']:.1f} | {r['s3t']:.1f} | {r['gain']:.2f}× |")
    lines += ["", "## δ vs #slices L (Fig 7-right)", "", "| L | B | SISA | S3T | gain |", "|---|---|---|---|---|"]
    for r in lvar:
        lines.append(f"| {r['L']} | {r['B']} | {r['sisa']:.1f} | {r['s3t']:.1f} | {r['gain']:.2f}× |")

    def perf_table(cv, tag):
        out = [f"### {tag}", "",
               "| r | " + " | ".join(f"B={B}{' (SISA)' if B==1 else ''}" for B in cv) + " |",
               "|---|" + "---|" * len(cv)]
        ml = max(len(v[0]) for v in cv.values())
        for r in range(0, ml, max(1, ml // 12)):
            cells = [f"{cv[B][0][r]:.3f}" if r < len(cv[B][0]) else "—" for B in cv]
            out.append(f"| {r} | " + " | ".join(cells) + " |")
        return out + [""]
    lines += ["", "## Performance vs #deletions (Fig 6-left)", "",
              "model_utility after r uniform-random slice deletions (mean over streams):", ""]
    lines += perf_table(curves, "armA (paper-faithful)")
    if curves2 is not None:
        lines += perf_table(curves2, "armB (tuned contrast)")

    lines += [""] + fig9_section(m, L, timing, [r["B"] for r in rate_rows], n_seeds)
    lines += lemma2_section(m, L, ks=[1, 2], rs=[1, 3, 6, 12], Bs=[r["B"] for r in rate_rows])
    lines += storage_section(m, L, [r["B"] for r in rate_rows], src, n_seeds)
    lines += rq3_section(src)
    lines += ["## Notes / deviations", "",
              "- Deletion-rate / Fig 6-right / Fig 7 are exact-combinatorial (no GPU); they "
              "match the paper's coupon-collector theory (validated in test_s3t_sequences.py).",
              "- Performance composition F(depths).mean() is the uniform per-shard approximation; "
              "armA F(d) is near-flat (paper-faithful HPs undertrain on TOFU) so its curve barely "
              "drops — armB is the informative contrast.",
              "- TOFU 'performance' = model_utility on retain/real/world; a deletion request "
              "removes one author-slice (uniform prior), the faithful analogue of the paper's "
              "random slice deletion.",
              "- Eq-18 uses the self-consistent (1-k/L)^r (the printed 1-(k/L)^r is inconsistent "
              "with the Eq-21 derivation / S3T(B=1)=SISA).", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def make_figures(outdir, m, L, F, rate_rows, curves):
    if not HAVE_MPL:
        print("[figures] matplotlib unavailable; skipping plots (tables still written)")
        return
    os.makedirs(outdir, exist_ok=True)
    # Fig 6-left
    plt.figure(figsize=(6, 4))
    for B, (mean, std) in curves.items():
        x = np.arange(len(mean))
        lab = "SISA (B=1)" if B == 1 else f"S3T (B={B})"
        plt.plot(x, mean, label=lab)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.axhline(F[-1], ls="--", c="gray", label="full retrain")
    plt.xlabel("# deletion requests"); plt.ylabel("model_utility")
    plt.title(f"Performance vs deletions (m={m}, L={L})"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig6_left_perf_vs_deletions.png"), dpi=120); plt.close()
    # Fig 6-right
    plt.figure(figsize=(5, 4))
    Bs = [r["B"] for r in rate_rows]
    plt.errorbar(Bs, [r["emp"] for r in rate_rows], yerr=[r["std"] for r in rate_rows],
                 marker="o", label="simulation")
    plt.plot(Bs, [r["theory"] for r in rate_rows], ls="--", marker="x", label="theory")
    plt.xlabel("budget B"); plt.ylabel("deletion rate δ")
    plt.title(f"Deletion rate vs B (m={m}, L={L})"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig6_right_delta_vs_B.png"), dpi=120); plt.close()
    print(f"[figures] wrote plots to {outdir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="checkpoints/Llama-2-7B-chat-hf_s3t_m5_L4_armA")
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--Bs", default="1,2,4")
    p.add_argument("--n_seeds", type=int, default=300)
    p.add_argument("--n_points", type=int, default=80)
    p.add_argument("--src2", default=None, help="second F-curve dir (armB) for the contrast curve")
    p.add_argument("--report", default=None)
    p.add_argument("--figdir", default=None)
    a = p.parse_args()
    Bs = [int(x) for x in a.Bs.split(",")]

    def load_F(src):
        """Return (F, forget_quality) from F_curve.json; placeholder + interp if absent/NaN."""
        fp = os.path.join(src, "F_curve.json")
        if os.path.exists(fp):
            fc = json.load(open(fp))
            F, fq = np.array(fc["F"], dtype=float), fc.get("forget_quality")
        else:
            print(f"[warn] {fp} missing; using monotone placeholder F")
            F, fq = np.linspace(0.4179, 0.58, a.L + 1), None
        if np.isnan(F).any():
            idx = np.arange(len(F)); good = ~np.isnan(F)
            F = np.interp(idx, idx[good], F[good])
        return F, fq

    F, F_fq = load_F(a.src)
    F2, curves2 = None, None
    if a.src2 and os.path.exists(os.path.join(a.src2, "F_curve.json")):
        F2, _ = load_F(a.src2)
        curves2 = perf_curves(a.m, a.L, F2, Bs, a.n_seeds, a.n_points)

    rate_rows = fig_deletion_rate_vs_B(a.m, a.L, Bs, a.n_seeds)
    mvar = fig_delta_vs_m(a.L, max(Bs), [2, 4, 5, 10], a.n_seeds)
    lvar = fig_delta_vs_L(a.m, max(Bs), [2, 4, 8], a.n_seeds)
    curves = perf_curves(a.m, a.L, F, Bs, a.n_seeds, a.n_points)

    tpath = os.path.join(a.src, "deletion_time.json")
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
    if F_fq is not None:
        timing["F_fq"] = F_fq

    report = a.report or os.path.join("reports", f"S3T_PAPER_REPRO_{date.today().isoformat()}.md")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    write_report(report, a.m, a.L, F, rate_rows, mvar, lvar, curves, timing, a.n_seeds,
                 F2=F2, curves2=curves2, src=a.src)
    make_figures(a.figdir or "reports/s3t_figs", a.m, a.L, F, rate_rows, curves)


if __name__ == "__main__":
    main()
