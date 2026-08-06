"""CPU gate for the APA uniform-summation study (Experiments A / B / C, 2026-07-28).

Run before any SLURM submission:
    ${TOFU_PYTHON:-python3} test_expa.py

Covers the five things that would silently corrupt the result rather than crash:
  1. label grammar - nmerge_sum_* must parse to additive_sum, and specifically NOT to
     "centered_lowrank_rm" (what a regex-only patch produces via tag[2:] on "sum");
  2. default-identity - configs without the new keys emit byte-identical manifests, and
     eval_tofu's metric functions are bit-identical with per_example=None;
  3. the paraphrase path is real - question_key actually changes the measurement;
  4. holdout10 is disjoint from `full` and the perturbed splits cover only 0-19 / 180-199
     (both are load-bearing premises of Exp B and Exp C, and one contradicts CLAUDE.md);
  5. the block decomposition is exhaustive - sum_i c_i reconstructs the merged delta - and
     drop-a-term is exact under a FIXED per-adapter weight.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np
import torch


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import merge_subset as MS                                    # noqa: E402
from analyze_nmerge import parse_label                       # noqa: E402


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

# Module-level os.environ[...] reads: the site env must be loaded HERE, not inside
# load_config, or a plain `import` dies with a bare KeyError.
_ensure_site_env()

OK = "ok  "


# ---------------------------------------------------------------------------
# 1. label grammar
# ---------------------------------------------------------------------------

def test_label_grammar():
    cases = {
        "nmerge_add_N8_s42": ("additive_mean", 8, 42, None),
        "nmerge_sum_N10_s42": ("additive_sum", 10, 42, None),
        "nmerge_sumisqrt_N20_s43": ("additive_sum_isqrt", 20, 43, None),
        "nmerge_sumL0p5_N4_s42": ("additive_sum_l0p5", 4, 42, None),
        "nmerge_sum_svd1024_N128_s42": ("additive_sum", 128, 42, 1024),
        "nmerge_cpool_svd1024_N20_s42": ("centered_pool", 20, 42, 1024),
        "nmerge_cr16_N8_s42": ("centered_lowrank_r16", 8, 42, None),
        "nmerge_dare_N200_s42_r8": ("dare_ties", 200, 42, None),
    }
    for lab, (meth, n, seed, svd) in cases.items():
        r = parse_label(lab)
        assert r["kind"] == "merge", (lab, r)
        assert (r["method"], r["n"], r["seed"], r["svd_rank"]) == (meth, n, seed, svd), (lab, r)
    # THE trap: a regex-only patch matches "sum", misses the tag map, and falls through to the
    # cr{rho} branch producing "centered_lowrank_" + "sum"[2:] == "centered_lowrank_rm".
    assert parse_label("nmerge_sum_N10_s42")["method"] != "centered_lowrank_rm"
    # and it must not silently become an unanalyzed row
    assert parse_label("nmerge_sumisqrt_N2_s42")["kind"] == "merge"
    assert parse_label("iso_a82")["kind"] == "iso"
    assert parse_label("base_model")["kind"] == "anchor"
    assert parse_label("total_garbage")["kind"] == "other"
    print(OK + "label grammar: sum / sumisqrt / sumL* parse correctly (centered_lowrank_rm trap held)")


# ---------------------------------------------------------------------------
# 2. default-identity
# ---------------------------------------------------------------------------

def test_label_defaults_unchanged():
    assert MS.merge_label("additive_sum", 8, 42) == "nmerge_sum_N8_s42"
    assert MS.merge_label("additive_sum", 8, 42, lam=None) == "nmerge_sum_N8_s42"
    assert MS.merge_label("additive_sum", 8, 42, lam=1.0) == "nmerge_sum_N8_s42"
    assert MS.merge_label("additive_mean", 64, 42, 1024) == "nmerge_add_svd1024_N64_s42"
    assert MS.merge_label("additive_sum", 8, 42, lam="isqrt") == "nmerge_sumisqrt_N8_s42"
    assert MS.lam_weight(None, 20) == 1.0 and MS.lam_weight(1.0, 20) == 1.0
    assert abs(MS.lam_weight("isqrt", 16) - 0.25) < 1e-12
    print(OK + "merge_label / lam_weight defaults byte-unchanged")


def test_specs_without_lam_are_unchanged():
    """A config predating lam_values must emit spec dicts with NO 'lam' key at all — the
    equality test_merge_subset asserts is the invariant that keeps old manifests valid."""
    cfg = {"methods": {"additive_mean": {"enabled": False},
                       "additive_sum": {"enabled": True, "exact_max_n": 64}},
           "subset_seeds": [42], "n_ladder": [1, 2, 4]}
    specs = MS._merge_specs(cfg)
    assert specs == [{"method": "additive_sum", "n": 2, "seed": 42, "svd_rank": None},
                     {"method": "additive_sum", "n": 4, "seed": 42, "svd_rank": None}], specs
    cfg["methods"]["additive_sum"]["lam_values"] = [None, "isqrt"]
    cfg["methods"]["additive_sum"]["lam_n_values"] = {"isqrt": [4]}
    specs = MS._merge_specs(cfg)
    assert len(specs) == 3, specs
    assert sum(1 for s in specs if s.get("lam") == "isqrt") == 1, specs
    print(OK + "spec generation: no 'lam' key without lam_values; lam_n_values restricts the arm")


def test_plan_is_identity_without_new_keys():
    """do_plan on a config lacking fixed_probe_authors / at_all_probes must reproduce the
    manifests byte-for-byte (the pin that keeps the pre-existing campaigns reproducible)."""
    # script-relative, not CWD-relative: submit_expa.sh's gate stage invokes this by absolute
    # path without cd-ing, so a relative path would resolve against the caller's cwd.
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "configs", "nmerge_interference_7b.json")
    cfg = MS.load_config(cfg_path)
    assert "fixed_probe_authors" not in cfg and not cfg["anchors"].get("at_all_probes")
    with tempfile.TemporaryDirectory() as td:
        cfg = dict(cfg, out_dir=td, retain_tr_source=None)
        mm, em = MS.do_plan(cfg, cfg_path)
        got_merge = open(mm).read()
        got_eval = open(em).read()
    ref = os.path.join(os.environ["TOFU_CKPT_ROOT"], "Llama-2-7B-chat-hf_nmerge_r32")
    if os.path.exists(f"{ref}/merge_manifest.txt"):
        old = open(f"{ref}/merge_manifest.txt").read().strip().split("\n")
        new = got_merge.strip().split("\n")
        # Manifest columns have only ever been APPENDED (rho 2026-07-15, lam 2026-07-28), and
        # submit_nmerge.sh treats a missing trailing column as '-'. So the invariant is that the
        # columns the on-disk manifest actually has are unchanged — comparing full rows would
        # just re-detect that this file predates the rho column.
        ncol = max(len(l.split("\t")) for l in old)
        norm = ["\t".join(l.split("\t")[:ncol]) for l in new]
        assert norm == old, (f"merge manifest changed in its first {ncol} columns for a "
                             f"pre-existing config\n  new: {norm[:2]}\n  old: {old[:2]}")
        # every appended column must be the inert '-' for a config that does not use it
        assert all(c == "-" for l in new for c in l.split("\t")[ncol:]), \
            "a config without lam/rho emitted a non-'-' value in an appended column"
        oldev = open(f"{ref}/eval_manifest_nmerge.txt").read()
        assert got_eval.replace(td, ref).strip() == oldev.strip(), \
            "eval manifest changed for a pre-existing config"
        print(OK + f"do_plan identical vs the on-disk e5 manifests in all {ncol} original "
                   f"columns + eval manifest byte-identical (new keys are opt-in)")
    else:
        print(OK + "do_plan ran; no on-disk reference manifest to diff against")


def test_eval_tofu_sink_identity():
    """per_example=None must be bit-identical, and the paraphrase path must actually bite."""
    import eval_tofu as E
    from transformers import LlamaConfig, LlamaForCausalLM
    from test_fixtures import resolve_tokenizer
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)).eval()
    # Only the TOKENIZER is real (the model is a random micro Llama sized to its vocab).
    tok = resolve_tokenizer("meta-llama/Llama-2-7B-chat-hf")
    tok.pad_token = tok.eos_token

    qs = ["Who wrote the book?", "Where was the author born?", "What genre is it?"]
    gs = ["Written by A. N. Other.", "Born in Taipei, Taiwan.", "Leadership."]

    a = E.get_rouge(model, tok, qs, gs, max_new_tokens=8)
    sink = []
    b = E.get_rouge(model, tok, qs, gs, max_new_tokens=8, per_example=sink)
    assert (a == b) or (math.isnan(a) and math.isnan(b)), (a, b)
    assert len(sink) == len(qs)
    assert abs(float(np.mean([r["score"] for r in sink])) - a) < 1e-12

    a = E.get_answer_probability(model, tok, qs, gs)
    sink = []
    b = E.get_answer_probability(model, tok, qs, gs, per_example=sink)
    assert a == b, (a, b)
    kept = [r for r in sink if r["kept"]]
    assert abs(float(np.mean([r["score"] for r in kept])) - a) < 1e-12

    ds = [{"question": q, "answer": g, "paraphrased_question": "Restated: " + q,
           "paraphrased_answer": g + " (rephrased)",
           "perturbed_answer": ["Wrong one.", "Wrong two."]} for q, g in zip(qs, gs)]
    a = E.get_truth_ratio_scores(model, tok, ds, "paraphrased_answer")
    sink = []
    b = E.get_truth_ratio_scores(model, tok, ds, "paraphrased_answer", per_example=sink)
    assert np.array_equal(a, b), (a, b)
    assert np.allclose([r["tr"] for r in sink if r["kept"]], a)
    c = E.get_truth_ratio_scores(model, tok, ds, "paraphrased_answer",
                                 question_key="paraphrased_question")
    assert not np.allclose(a, c), "question_key had NO effect — the paraphrase path is dead"
    print(OK + "eval_tofu: per_example is identity; question_key changes the measurement")


# ---------------------------------------------------------------------------
# 3. data premises
# ---------------------------------------------------------------------------

def test_data_premises():
    """The two premises Exp B and Exp C rest on. One of them contradicts a documented
    CLAUDE.md invariant ('perturbed splits cover ~2 rows/author'), which is why it is pinned."""
    from datasets import load_dataset
    full = load_dataset("locuslab/TOFU", "full")["train"]
    q2a = {}
    for i, r in enumerate(full):
        q2a.setdefault(r["question"], i // 20)

    cov = {}
    for cfg_name, expect in (("forget10_perturbed", set(range(180, 200))),
                             ("retain_perturbed", set(range(0, 20)))):
        ds = load_dataset("locuslab/TOFU", cfg_name)["train"]
        assert "paraphrased_question" in ds.column_names, cfg_name
        auth = [q2a[r["question"]] for r in ds if r["question"] in q2a]
        got = set(auth)
        assert got == expect, f"{cfg_name} covers {min(got)}..{max(got)}, expected {sorted(expect)[0]}..{sorted(expect)[-1]}"
        per = {a: auth.count(a) for a in got}
        assert min(per.values()) >= 19, f"{cfg_name} thin author: {min(per.values())} rows"
        cov[cfg_name] = (len(got), min(per.values()), max(per.values()))
        # paraphrased questions must be genuinely different text
        assert all(r["paraphrased_question"] != r["question"] for r in ds)
    # authors 20..179 have NO paraphrase coverage -> they are illegal Exp-B targets
    covered = set(range(0, 20)) | set(range(180, 200))
    assert not (set(range(20, 180)) & covered)

    ho = load_dataset("locuslab/TOFU", "holdout10")["train"]
    overlap = sum(1 for q in ho["question"] if q in q2a)
    assert overlap == 0, f"holdout10 overlaps `full` in {overlap}/{len(ho)} rows — not never-trained"
    print(OK + f"data premises: forget10_perturbed{cov['forget10_perturbed']} / "
               f"retain_perturbed{cov['retain_perturbed']} (n_authors, min, max rows); "
               f"holdout10 0/{len(ho)} overlap with full")


# ---------------------------------------------------------------------------
# 4. block decomposition + exact drop-a-term
# ---------------------------------------------------------------------------

def _fake_pool(td, n_authors, d_in=16, d_out=24, r=4, seed=0):
    """n tiny single-slot LoRA adapter dirs in the on-disk format merge_subset reads."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    slot = "model.layers.0.self_attn.q_proj"
    dirs = []
    for a in range(n_authors):
        d = os.path.join(td, f"shard_{a}")
        os.makedirs(d, exist_ok=True)
        A = torch.randn(r, d_in, generator=g)
        B = torch.randn(d_out, r, generator=g)
        save_file({f"base_model.model.{slot}.lora_A.weight": A,
                   f"base_model.model.{slot}.lora_B.weight": B},
                  os.path.join(d, "adapter_model.safetensors"))
        json.dump({"r": r, "lora_alpha": 2 * r, "use_rslora": False, "peft_type": "LORA",
                   "target_modules": ["q_proj"]}, open(os.path.join(d, "adapter_config.json"), "w"))
        dirs.append(d)
    return dirs, slot


def test_block_decomposition_and_exact_drop():
    """(a) the concatenated merge decomposes exhaustively into per-author blocks;
       (b) with a FIXED weight, dropping author j == subtracting w * its delta, exactly."""
    import measure_expb_contrib as C
    with tempfile.TemporaryDirectory() as td:
        n = 5
        dirs, slot = _fake_pool(td, n)
        w = 1.0 / n   # FIXED (not renormalized) — the drop-a-term precondition

        full_dir = os.path.join(td, "merge_full")
        merged, ref_cfg, out_rank, _ = MS.merge_additive_sum(dirs, weight=w)
        MS.write_effective_adapter(full_dir, merged, ref_cfg, out_rank)
        json.dump({"authors": list(range(n)), "svd_rank": None, "label": "merge_full"},
                  open(os.path.join(full_dir, "merge_meta.json"), "w"))

        blocks, authors, r, meta = C.load_blocks(full_dir)
        assert authors == list(range(n)) and r == 4
        A_cat, B_cat = blocks[slot]

        # (a) exhaustive decomposition: sum_i c_i == the exact delta, for random inputs
        h = torch.randn(7, A_cat.shape[1])
        z = torch.nn.functional.linear(h, A_cat.float())
        ref = torch.nn.functional.linear(z, B_cat.float())
        acc = torch.zeros_like(ref)
        for i in range(n):
            sl = slice(i * r, (i + 1) * r)
            acc += torch.nn.functional.linear(z[:, sl], B_cat.float()[:, sl])
        rel = float((acc - ref).norm() / ref.norm())
        assert rel < 1e-5, f"decomposition not exhaustive (rel err {rel:.2e})"

        # the Gram shortcut must agree with the dense per-author norms
        G = B_cat.float().t() @ B_cat.float()
        for i in range(n):
            sl = slice(i * r, (i + 1) * r)
            dense = torch.nn.functional.linear(z[:, sl], B_cat.float()[:, sl]).norm(dim=1)
            gram = torch.einsum('ti,ij,tj->t', z[:, sl], G[sl, sl], z[:, sl]).clamp_min(0).sqrt()
            assert torch.allclose(dense, gram, atol=1e-4), f"Gram norm != dense norm (author {i})"

        # (b) exact drop-a-term
        drop = 2
        keep = [d for a, d in enumerate(dirs) if a != drop]
        loo_dir = os.path.join(td, "merge_loo")
        m2, c2, rank2, _ = MS.merge_additive_sum(keep, weight=w)   # SAME fixed w
        MS.write_effective_adapter(loo_dir, m2, c2, rank2)
        bl2, _, _, _ = C.load_blocks(
            (json.dump({"authors": [a for a in range(n) if a != drop], "svd_rank": None},
                       open(os.path.join(loo_dir, "merge_meta.json"), "w")), loo_dir)[1])
        A2, B2 = bl2[slot]
        full_delta = B_cat.float() @ A_cat.float()
        loo_delta = B2.float() @ A2.float()
        sl = slice(drop * r, (drop + 1) * r)
        dropped = B_cat.float()[:, sl] @ A_cat.float()[sl, :]
        rel = float((full_delta - dropped - loo_delta).norm() / loo_delta.norm())
        assert rel < 1e-5, f"drop-a-term not exact (rel err {rel:.2e})"
    print(OK + f"block decomposition exhaustive (rel {1e-5:.0e}); Gram == dense; "
               f"fixed-weight drop-a-term exact")


def test_diffuseness_metric():
    import measure_expb_contrib as C
    n = 20
    flat = C.diffuseness(np.ones(n))
    assert abs(flat["n_eff"] - n) < 1e-9 and abs(flat["h_norm"] - 1.0) < 1e-9
    assert abs(flat["max_share"] - 1.0 / n) < 1e-12
    one = np.zeros(n)
    one[3] = 1.0
    peak = C.diffuseness(one)
    assert abs(peak["n_eff"] - 1.0) < 1e-9 and peak["max_share"] == 1.0
    assert peak["h_norm"] == 0.0
    assert C.diffuseness(np.zeros(n))["n_eff"] is None
    print(OK + "diffuseness: uniform -> n_eff=n, H=1; one-hot -> n_eff=1, H=0; all-zero -> None")


def main():
    from test_fixtures import FixtureMissing
    # Split by prerequisite: the first six are hermetic (pure logic + a synthetic pool), the last
    # two need a real tokenizer / the TOFU cache. A missing fixture SKIPS loudly and sets a
    # non-zero-worthy flag rather than passing silently — a gate that quietly stops testing is
    # the failure mode this whole file exists to prevent.
    hermetic = [test_label_grammar, test_label_defaults_unchanged,
                test_specs_without_lam_are_unchanged, test_plan_is_identity_without_new_keys,
                test_diffuseness_metric, test_block_decomposition_and_exact_drop]
    needs_fixtures = [test_eval_tofu_sink_identity, test_data_premises]
    for t in hermetic:
        t()
    skipped = []
    for t in needs_fixtures:
        try:
            t()
        except FixtureMissing as e:
            skipped.append(t.__name__)
            print(f"SKIP {t.__name__}: {e}")
    if skipped:
        print(f"\ntest_expa.py: {len(hermetic)} passed, {len(skipped)} SKIPPED for missing "
              f"fixtures ({', '.join(skipped)}) — NOT a clean run; pre-warm $HF_HOME.")
        return 1
    print("\nALL test_expa.py GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
