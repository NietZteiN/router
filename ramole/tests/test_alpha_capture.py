"""CPU correctness tests for the E2 alpha-capture path (router_lora.RouterController
.capture_alpha/.captured) and the analyze_router pooling math. Run before any SLURM job
(CLAUDE.md §4):

    ${TOFU_PYTHON:-python3} tests/test_alpha_capture.py

Anchors:
  - capture is OFF by default (a forward leaves controller.captured empty — zero serving cost)
  - one teacher-forced b=1 forward captures every installed path exactly once,
    alpha shape (m,1,l), sums to 1 over experts
  - THE CROSS-CHECK: an external forward hook reproducing the q/k/s/alpha math
    (test_router_lora.py) matches the captured alpha to 1e-6
  - near-uniform at init (H_norm > 0.95): B_r init std 0.01 => scores ~0 => alpha ~ 1/m
  - alpha_stats unit math on hand-built tensors: known entropy, max-share, ideal-mass
    (present / absent / m==1 guard), per-layer slots, completion-position deciles,
    multi-entry raise
  - capture_for_records generator contract: pools online, clears captured, restores
    capture_alpha=False, finite NLL
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only test; never touch the (login-node) GPUs
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import math
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))  # ramole/
sys.path.insert(0, THIS)

from _fixture import build_source_run  # noqa: E402
import analyze_router as ar            # noqa: E402
import router_lora as R                # noqa: E402


def _checks():
    tmp = tempfile.mkdtemp(prefix="ramole_alpha_")
    # with_tokenizer=True: build_ramole_model loads the tokenizer, and capture_for_records
    # needs text->ids (the real Llama-3.2 tokenizer is copied from local storage, offline-safe).
    cfg = build_source_run(tmp, n=3, hidden=64, layers=2, heads=4, kv_heads=2, rank=4, alpha=8,
                           with_tokenizer=True)
    model, tok, ctrl, meta, installed = R.build_ramole_model(
        cfg, device="cpu", load_router_weights=False)
    model.eval()
    ids = tok("the quick brown fox jumps over", return_tensors="pt").input_ids
    assert ids.shape[0] == 1 and ids.shape[1] >= 4

    # ── capture is off by default ───────────────────────────────────────────────
    ctrl.set_active([0, 1, 2])
    with torch.no_grad():
        model(input_ids=ids)
    assert ctrl.capture_alpha is False and ctrl.captured == {}, \
        f"capture must be opt-in; got {len(ctrl.captured)} captured paths with it off"
    print("[ok] capture off by default: forward leaves controller.captured empty")

    # ── one b=1 forward captures every path once; alpha (m,1,l) sums to 1 ───────
    # + THE CROSS-CHECK: external hook reproducing the module's q/k/s/alpha math
    hook_out = {}

    def hook(mod, inp, out):
        x = inp[0]
        active = mod.controller.active_idx
        A = mod.expert_A.index_select(0, active).float()
        B = mod.expert_B.index_select(0, active).float()
        xf = x.float()
        Ax = torch.einsum("mrd,bld->mblr", A, xf)
        v = mod.scaling * torch.einsum("mor,mblr->mblo", B, Ax)
        q = torch.einsum("rd,bld->blr", mod.A_r, xf)
        kk = torch.einsum("or,mblo->mblr", mod.B_r, v)
        s = torch.einsum("blr,mblr->mbl", q, kk) / mod._sqrt_r
        if mod.controller.logit_mask is not None:
            s = s + mod.controller.logit_mask.transpose(0, 1).unsqueeze(-1)
        hook_out["alpha"] = torch.softmax(s, dim=0)

    probe_path = installed[0]
    h = model.get_submodule(probe_path).register_forward_hook(hook)
    ctrl.captured.clear()
    ctrl.capture_alpha = True
    with torch.no_grad():
        model(input_ids=ids)
    ctrl.capture_alpha = False
    h.remove()

    m, l = 3, ids.shape[1]
    assert set(ctrl.captured) == set(installed), \
        f"captured {len(ctrl.captured)}/{len(installed)} installed paths"
    for path in installed:
        entries = ctrl.captured[path]
        assert len(entries) == 1, f"{path}: {len(entries)} entries for one forward"
        act, alpha = entries[0]
        assert act.tolist() == [0, 1, 2]
        assert alpha.shape == (m, 1, l), f"{path}: alpha shape {tuple(alpha.shape)}"
        assert alpha.dtype == torch.float32 and alpha.device.type == "cpu"
        asum = alpha.sum(0)
        assert torch.allclose(asum, torch.ones_like(asum), atol=1e-5), \
            f"{path}: alpha sum != 1 ({asum.min():.6f}..{asum.max():.6f})"
    print(f"[ok] one forward: all {len(installed)} paths captured once, alpha (m,1,l) sums to 1")

    d = (ctrl.captured[probe_path][0][1] - hook_out["alpha"]).abs().max().item()
    assert d < 1e-6, f"captured alpha != external-hook alpha on {probe_path}: {d}"
    print(f"[ok] cross-check vs external hook on {probe_path}: max|Δ|={d:.2e} < 1e-6")

    # ── near-uniform at init (B_r std 0.01 => scores ~0 => H_norm ~ 1) ──────────
    stats = ar.alpha_stats(ctrl.captured, [0, 1, 2], ideal_expert=1, prompt_len=2)
    assert stats["H_norm_mean"] > 0.95, f"init not near-uniform: H_norm={stats['H_norm_mean']}"
    assert stats["ideal_present"] is True
    assert abs(stats["ideal_mass_mean"] - 1 / 3) < 0.05 and abs(stats["max_share_mean"] - 1 / 3) < 0.05
    print(f"[ok] near-uniform at init: H_norm={stats['H_norm_mean']:.4f} > 0.95, "
          f"max_share={stats['max_share_mean']:.4f}~1/3")
    ctrl.captured.clear()

    # ── alpha_stats unit math on a hand-built tensor ────────────────────────────
    L = 10
    alpha = torch.zeros(2, 1, L)
    alpha[0], alpha[1] = 0.75, 0.25
    cap = {"model.layers.0.self_attn.q_proj": [(torch.tensor([5, 9]), alpha)]}
    H_exp = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2)
    s = ar.alpha_stats(cap, [5, 9], ideal_expert=9, prompt_len=4)
    assert abs(s["H_norm_mean"] - H_exp) < 1e-6, (s["H_norm_mean"], H_exp)
    assert abs(s["max_share_mean"] - 0.75) < 1e-6
    assert abs(s["ideal_mass_mean"] - 0.25) < 1e-6 and s["ideal_present"] is True
    assert abs(s["per_layer"]["q_proj"][0] - H_exp) < 1e-6
    assert all(math.isnan(s["per_layer"][p][0]) for p in ("k_proj", "v_proj", "o_proj"))
    # 6 completion positions (t=4..9) -> deciles {0,1,3,5,6,8} filled with H_exp, rest NaN
    dec = s["per_position_decile"]
    filled = {i for i, v in enumerate(dec) if not math.isnan(v)}
    assert filled == {0, 1, 3, 5, 6, 8}, filled
    assert all(abs(dec[i] - H_exp) < 1e-6 for i in filled)
    # ideal absent -> NaN mass, counted absent
    s2 = ar.alpha_stats(cap, [5, 9], ideal_expert=3, prompt_len=0)
    assert math.isnan(s2["ideal_mass_mean"]) and s2["ideal_present"] is False
    # one-hot -> H = 0 exactly (0*ln0 guarded); m==1 -> H = 1.0 by definition
    hot = torch.zeros(2, 1, 4)
    hot[0] = 1.0
    s3 = ar.alpha_stats({"model.layers.1.self_attn.v_proj": [(torch.tensor([0, 2]), hot)]},
                        [0, 2], None, prompt_len=0)
    assert s3["H_norm_mean"] == 0.0 and s3["ideal_present"] is None
    s4 = ar.alpha_stats({"model.layers.0.self_attn.o_proj": [(torch.tensor([2]), torch.ones(1, 1, 4))]},
                        [2], ideal_expert=2, prompt_len=0)
    assert s4["H_norm_mean"] == 1.0 and abs(s4["ideal_mass_mean"] - 1.0) < 1e-9
    # two captured entries for one record must raise (the never-capture-generate guard)
    try:
        ar.alpha_stats({"model.layers.0.self_attn.q_proj":
                        [(torch.tensor([5, 9]), alpha), (torch.tensor([5, 9]), alpha)]},
                       [5, 9], None)
        raise AssertionError("alpha_stats must reject multi-entry captures")
    except ValueError:
        pass
    print(f"[ok] alpha_stats unit math: H={H_exp:.4f} exact, ideal present/absent/m==1, "
          "deciles, multi-entry raise")

    # ── capture_for_records: online pooling, cleanup, finite NLL ────────────────
    rm = SimpleNamespace(model=model, tokenizer=tok, controller=ctrl)  # no .set_active =>
    recs = [{"id": "r0"}, {"id": "r1"}]                                # controller fallback
    sets = {"r0": (0, 1), "r1": (1, 2)}
    ideals = {"r0": 0, "r1": 0}   # r1: ideal 0 not in (1,2) -> absent
    out = list(ar.capture_for_records(
        rm, recs, sets, 32, lambda r: ("the quick brown fox jumps", "the quick"),
        ideals=ideals))
    assert [s["id"] for s in out] == ["r0", "r1"] and all(s["m"] == 2 for s in out)
    assert all(np.isfinite(s["nll"]) for s in out)
    assert out[0]["ideal_present"] is True and out[1]["ideal_present"] is False
    assert math.isnan(out[1]["ideal_mass_mean"]) and out[0]["active"] == [0, 1]
    assert 0 < out[0]["prompt_len"] < out[0]["seq_len"]
    assert ctrl.captured == {} and ctrl.capture_alpha is False, \
        "capture_for_records must discard raw tensors and restore capture_alpha=False"
    print(f"[ok] capture_for_records: 2 records pooled online (nll={out[0]['nll']:.3f}, "
          f"{out[1]['nll']:.3f}), captured cleared, capture off")


def _dropped_checks():
    """analyze_router_tofu --dropped gate (H-TRAINED): masked-active-set capture on the stub
    router — the dropped experts never appear in any captured active set, the softmax
    renormalizes over the survivors (alpha sums to 1), and the assembled JSON carries the
    per-query array contract the family analyzer consumes."""
    TOFU_DIR = os.path.join(os.path.dirname(os.path.dirname(THIS)), "tofu_sisa_lora")
    if TOFU_DIR not in sys.path:
        sys.path.insert(0, TOFU_DIR)
    import analyze_router_tofu as art

    tmp = tempfile.mkdtemp(prefix="ramole_alpha_drop_")
    cfg = build_source_run(tmp, n=3, hidden=64, layers=2, heads=4, kv_heads=2, rank=4, alpha=8,
                           with_tokenizer=True)
    model, tok, ctrl, meta, installed = R.build_ramole_model(
        cfg, device="cpu", load_router_weights=False)
    model.eval()
    rm = SimpleNamespace(model=model, tokenizer=tok, controller=ctrl)

    # dropped_active: mask filters the routed set; an emptied set falls back to the top-k
    # SURVIVORS by the same index ranking (the serving fallback)
    index = np.eye(3, dtype="float32")
    qv = np.array([0.1, 0.2, 0.9], dtype="float32")   # nearest = expert 2 (the affected one)
    d1, fb1 = art.dropped_active((0, 1, 2), {2}, index, qv, 2)
    assert d1 == (0, 1) and fb1 is False
    d2, fb2 = art.dropped_active((2,), {2}, index, qv, 2)
    assert fb2 is True and d2 == (1, 0) and 2 not in d2   # survivor ranking: e1 (0.2) > e0 (0.1)
    print("[ok] dropped_active: mask filter + emptied-set fallback ranks survivors (1,0)")

    # masked-active-set capture: dropped expert never in any captured active; softmax over
    # the survivors renormalizes (sums to 1 with m=2, not a column-mask of the m=3 softmax)
    rec = {"id": 0, "text": "the quick brown fox jumps over", "prompt": "the quick"}
    ids = tok(rec["text"], return_tensors="pt").input_ids
    ctrl.set_active([0, 1])
    ctrl.captured.clear()
    ctrl.capture_alpha = True
    with torch.no_grad():
        model(input_ids=ids)
    ctrl.capture_alpha = False
    for path in installed:
        act, alpha = ctrl.captured[path][0]
        assert act.tolist() == [0, 1] and 2 not in act.tolist(), \
            f"{path}: dropped expert leaked into the captured active set"
        assert alpha.shape[0] == 2
        asum = alpha.sum(0)
        assert torch.allclose(asum, torch.ones_like(asum), atol=1e-5), \
            f"{path}: renormalized alpha sum != 1"
    ctrl.captured.clear()
    print(f"[ok] masked active set: dropped expert absent from all {len(installed)} captures; "
          "renormalized alpha sums to 1")

    # _one_captured_forward: strict contract (alpha_stats inside), finite stats, cleanup
    s_full = art._one_captured_forward(rm, rec, (0, 1, 2), 32)
    s_drop = art._one_captured_forward(rm, rec, d1, 32)
    assert ctrl.captured == {} and ctrl.capture_alpha is False, \
        "_one_captured_forward must clear captured and restore capture_alpha=False"
    for s, m in ((s_full, 3), (s_drop, 2)):
        assert np.isfinite([s["h_norm"], s["max_share"], s["top1_share"], s["nll"]]).all()
        assert 1.0 / m - 1e-6 <= s["top1_share"] <= 1.0 + 1e-6 and len(s["active"]) == m
    assert 2 not in s_drop["active"]
    print(f"[ok] _one_captured_forward: full m=3 / dropped m=2 finite "
          f"(top1_share {s_full['top1_share']:.3f}/{s_drop['top1_share']:.3f}), captured cleared")

    # assemble_dropped_result: the analyzer JSON array contract
    rows = [{"id": i, "is_forget": grp == "forget", "author": i, "fallback": bool(i == 1),
             "full": dict(s_full), "drop": dict(s_drop)}
            for i, grp in enumerate(("forget", "forget", "retain", "retain"))]
    res = art.assemble_dropped_result(rows)
    for key in ("h_norm", "max_share", "top1_share", "top1_share_full", "is_forget",
                "author_of_q", "h_norm_full", "max_share_full", "nll", "nll_full",
                "n_active", "n_active_full", "fallback_used", "query_id"):
        assert key in res and len(res[key]) == len(rows), f"contract array missing/short: {key}"
    assert set(res["groups"]) == {"forget", "retain"}
    for g in res["groups"].values():
        assert g["n"] == 2 and np.isfinite(g["top1_share_ratio_mean"])
    assert 0.0 <= res["auc_h_norm"] <= 1.0 and 0.0 <= res["auc_max_share"] <= 1.0
    assert res["groups"]["forget"]["fallback_rate"] == 0.5
    print("[ok] assemble_dropped_result: contract arrays + group means + AUCs present")


if __name__ == "__main__":
    _checks()
    _dropped_checks()
    print("\nALL ALPHA-CAPTURE TESTS PASSED")
