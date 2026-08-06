"""CPU correctness tests for the RouterLoRA cross-attention (router_lora.py).

Run before any SLURM job (CLAUDE.md §4):
    ${TOFU_PYTHON:-python3} tests/test_router_lora.py

Anchors:
  - extraction: 4*layers paths, GQA d_out split, rank/scaling correct
  - single active expert ≡ that LoRA applied directly via PEFT (the load-bearing identity)
  - m identical experts ≡ one (attention sums to 1 over identical v_i)
  - per-sample mask routes each row to its own expert
  - alpha sums to 1 over experts; no NaN across many dropout steps
  - save/load round-trips bitwise; only A_r/B_r carry gradients
"""
import os
import sys
import tempfile

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from _fixture import build_source_run, load_base  # noqa: E402
import router_lora as R  # noqa: E402

TOL = 2e-5


def _checks():
    tmp = tempfile.mkdtemp(prefix="ramole_rl_")
    cfg = build_source_run(tmp, n=3, hidden=64, layers=2, heads=4, kv_heads=2, rank=4, alpha=8)
    torch.manual_seed(0)
    ids = torch.randint(0, 64, (2, 5))  # vocab=64 in the fixture

    # ── extraction ────────────────────────────────────────────────────────────
    experts, meta = R.extract_expert_weights(cfg, dtype=torch.float32)
    assert meta["num_paths"] == 4 * 2, f"expected 8 paths, got {meta['num_paths']}"
    assert meta["rank"] == 4 and abs(meta["scaling"] - 8 / 4) < 1e-9, meta
    douts = {mp: experts[mp]["B"].shape[1] for mp in experts}
    qo = [v for k, v in douts.items() if k.endswith(("q_proj", "o_proj"))]
    kv = [v for k, v in douts.items() if k.endswith(("k_proj", "v_proj"))]
    assert set(qo) == {64} and set(kv) == {32}, f"GQA dims wrong: qo={qo} kv={kv}"
    print(f"[ok] extraction: {meta['num_paths']} paths, GQA d_out q/o=64 k/v=32, scaling={meta['scaling']}")

    # ── build a RAMoLE model (no router ckpt) ───────────────────────────────────
    def fresh():
        base = load_base(cfg)
        ex, me = R.extract_expert_weights(cfg, dtype=torch.float32)
        ctrl = R.RouterController(me["n"])
        R.install_router(base, ex, me["scaling"], cfg["router"]["rank"], ctrl)
        names = R.freeze_to_router(base)
        return base.eval(), ctrl, names

    model, ctrl, rnames = fresh()
    assert len(rnames) == 2 * meta["num_paths"], f"expected {2 * meta['num_paths']} router params, got {len(rnames)}"

    # ── identity: single active expert ≡ direct PEFT adapter ────────────────────
    from copy import deepcopy
    from peft import PeftModel
    for j in range(cfg["n"]):
        ctrl.set_active([j])
        with torch.no_grad():
            out_r = model(input_ids=ids).logits
        pm = PeftModel.from_pretrained(
            deepcopy(load_base(cfg)), os.path.join(tmp, "runs", "fix", "adapters", f"a{j}"))
        pm.eval()
        with torch.no_grad():
            out_p = pm(input_ids=ids).logits
        d = (out_r - out_p).abs().max().item()
        assert d < TOL, f"single-expert a{j} mismatch vs PEFT: {d}"
    print(f"[ok] single active expert ≡ direct PEFT adapter (max|Δ|<{TOL}) for all {cfg['n']}")

    # ── m identical experts ≡ one (active=[0,0,0]) ──────────────────────────────
    ctrl.set_active([1])
    with torch.no_grad():
        out_one = model(input_ids=ids).logits
    ctrl.set_active([1, 1, 1])
    with torch.no_grad():
        out_three = model(input_ids=ids).logits
    d = (out_one - out_three).abs().max().item()
    assert d < TOL, f"3 identical experts != 1: {d}"
    print(f"[ok] m identical experts ≡ one (max|Δ|<{TOL})")

    # ── per-sample mask routes each row to its own expert ───────────────────────
    # batch of 2; row0 → expert 0, row1 → expert 2. union=[0,2], mask -inf the other.
    NEG = float("-inf")
    union = [0, 2]
    mask = torch.tensor([[0.0, NEG], [NEG, 0.0]])  # (b=2, m=2)
    ctrl.set_routed(union, mask)
    with torch.no_grad():
        out_masked = model(input_ids=ids).logits
    # reference: each row computed with its single expert
    ctrl.set_active([0])
    with torch.no_grad():
        ref0 = model(input_ids=ids[0:1]).logits
    ctrl.set_active([2])
    with torch.no_grad():
        ref1 = model(input_ids=ids[1:2]).logits
    d0 = (out_masked[0:1] - ref0).abs().max().item()
    d1 = (out_masked[1:2] - ref1).abs().max().item()
    assert d0 < TOL and d1 < TOL, f"masked routing mismatch: row0={d0} row1={d1}"
    print(f"[ok] per-sample mask routes each row to its expert (Δ row0={d0:.2e} row1={d1:.2e})")

    # ── alpha sums to 1 / no NaN over dropout steps ─────────────────────────────
    # capture alpha from one layer via a forward hook reproducing the module math
    layer0 = model.get_submodule(meta["paths"][0])
    captured = {}

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
        captured["alpha"] = torch.softmax(s, dim=0)

    h = layer0.register_forward_hook(hook)
    ctrl.set_active([0, 1, 2])
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    asum = captured["alpha"].sum(0)
    assert torch.allclose(asum, torch.ones_like(asum), atol=1e-5), f"alpha sum != 1: {asum.min()}..{asum.max()}"
    print(f"[ok] alpha sums to 1 over experts (min={asum.min():.6f} max={asum.max():.6f})")

    g = torch.Generator().manual_seed(7)
    ctrl.set_pool([0, 1, 2])
    for _ in range(40):
        ctrl.sample_dropout(0.5, g)
        assert ctrl.active_idx.numel() >= 2
        with torch.no_grad():
            lg = model(input_ids=ids).logits
        assert torch.isfinite(lg).all(), "NaN/Inf in logits under dropout"
    print("[ok] no NaN/Inf across 40 dropout steps (>=2 survivors each)")

    # ── router-only gradients ───────────────────────────────────────────────────
    ctrl.set_active([0, 1, 2])
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grad_names = [n for n, p in model.named_parameters() if p.grad is not None]
    assert grad_names, "no gradients at all"
    assert all(n.endswith((".A_r", ".B_r")) for n in grad_names), \
        f"non-router params got gradients: {[n for n in grad_names if not n.endswith(('.A_r', '.B_r'))][:3]}"
    assert len(grad_names) == len(rnames), f"only {len(grad_names)}/{len(rnames)} router params got grad"
    print(f"[ok] gradients flow to exactly the {len(grad_names)} router params, nothing else")

    # ── save / load round-trip ───────────────────────────────────────────────────
    with torch.no_grad():
        for n_, p in model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)  # perturb so the ckpt is non-default
    ctrl.set_active([0, 1, 2])
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    rpath = os.path.join(tmp, "router.safetensors")
    R.save_router(model, rpath)

    model2, ctrl2, _ = fresh()
    R.load_router(model2, rpath)
    ctrl2.set_active([0, 1, 2])
    with torch.no_grad():
        got = model2(input_ids=ids).logits
    d = (ref - got).abs().max().item()
    assert d < 1e-6, f"router save/load not bitwise: {d}"
    sd1 = R.router_state_dict(model)
    sd2 = R.router_state_dict(model2)
    assert set(sd1) == set(sd2) and all(torch.equal(sd1[k], sd2[k]) for k in sd1), "param mismatch after load"
    print(f"[ok] router save/load round-trips (max|Δ logits|={d:.2e})")

    # key-mismatch guard
    bad = dict(sd1)
    bad.pop(next(iter(bad)))
    badpath = os.path.join(tmp, "bad.safetensors")
    from safetensors.torch import save_file
    save_file(bad, badpath)
    try:
        R.load_router(model2, badpath)
        raise AssertionError("load_router should reject a key-set mismatch")
    except RuntimeError:
        pass
    print("[ok] load_router rejects key-set mismatch")


if __name__ == "__main__":
    _checks()
    print("\nALL ROUTER_LORA TESTS PASSED")
