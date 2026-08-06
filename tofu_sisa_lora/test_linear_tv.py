"""CPU micro-tests for linear_tv.py / train_linear_tv.py (no downloads, no GPU).

Run before any composable_tv [lin] SLURM job: python test_linear_tv.py

Tiny random GQA Llama (hidden 64, 2 layers, kv_heads < heads, vocab 128, EAGER attention —
the fwAD requirement), the test_merge_subset / test_train_anchor fixture family. Checks:

  a. linearized_forward == the closed form f0 + J.tau computed independently via
     torch.autograd.functional.jvp (double-backward trick), tol 1e-4 fp32.
  b. zero tangent (B=0 init) -> BIT-equal base logits (primal path runs identical ops; a
     zero tangent propagates as exact zeros).
  c. superposition: f_lin(t1+t2) - f0 == (f_lin(t1)-f0) + (f_lin(t2)-f0) to fp tolerance —
     the exactness guarantee the whole arm rests on.
  d. compose_tangents: subtract == recompose (atol 1e-6) and sorted-by-dirname determinism
     (byte-equal across caller order).
  e. loss.backward() through the linearized loss reaches EVERY B (nonzero grads); A stays
     frozen (requires_grad False, bytes untouched).
  f. determinism: seeded_A byte-identical across calls; two independent 2-step training runs
     produce byte-equal B (the re-derivability provenance, like SIFT's tau_u).
  g. LinearTVModel.generate == stepwise argmax over linearized logits; honors eos +
     max_new_tokens.
  h. materialized PEFT save -> merge_subset._weighted_factor_cat dense delta == the tangent
     sum (the lin-arm shards stay interoperable with the nmerge tooling).
  i. B-only save/load round-trip: A re-derived from seed byte-equals the original factors.
  j. eval_tofu flag seam: new --linear_tv_* args default to None/'linear' and existing arg
     defaults are unchanged (light-touch; no eval run).
  +  nonlinear-debug serve == direct weight addition; disentanglement_error is exactly 0
     against a zero partner and byte-restores the model.
"""

import copy
import os
import shutil
import sys
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import linear_tv as ltv
from train_linear_tv import collate_rows, train_linear_factors

VOCAB = 128
RANK = 4
ALPHA = 8
IRP_SEED = 42

torch.manual_seed(0)


def tiny_cfg():
    return LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        max_position_embeddings=64, attn_implementation="eager",
    )


def build_model():
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_cfg())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def rand_factors(model, names, author, b_scale=0.05):
    """Author factors with the REAL seeded A but random nonzero B (nonzero tangent)."""
    factors = ltv.init_author_factors(
        model, names, rank=RANK, alpha=ALPHA, rslora=True,
        irp_seed=IRP_SEED, author=author)
    gen = torch.Generator().manual_seed(1000 + author)
    out = {}
    for n, (A, B, s) in factors.items():
        Br = torch.empty_like(B).normal_(0.0, b_scale, generator=gen).requires_grad_(True)
        out[n] = (A, Br, s)
    return out


def make_batch(seed=3, bsz=2, width=7):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (bsz, width), generator=gen)
    return ids, torch.ones_like(ids)


def test_closed_form(model, names):
    """(a) linearized_forward == f0 + autograd jacobian-vector product, tol 1e-4."""
    factors = rand_factors(model, names, author=5)
    tau = {n: t.detach() for n, t in ltv.build_tangent(factors).items()}
    ids, am = make_batch()
    got = ltv.linearized_forward(model, names, tau, ids, attention_mask=am).logits

    def f_flat(*params):
        return torch.func.functional_call(
            model, dict(zip(names, params)), args=(),
            kwargs={"input_ids": ids, "attention_mask": am, "use_cache": False}).logits

    sd = dict(model.named_parameters())
    f0, jvp_ref = torch.autograd.functional.jvp(
        f_flat, tuple(sd[n].detach() for n in names), tuple(tau[n] for n in names))
    ref = f0 + jvp_ref
    err = (got - ref).abs().max().item()
    assert err < 1e-4, err
    print(f"ok  (a) linearized_forward == f0 + autograd-JVP closed form (max abs err {err:.2e})")


def test_zero_tangent(model, names):
    """(b) B=0 init tangent -> bit-equal base logits."""
    factors = ltv.init_author_factors(
        model, names, rank=RANK, alpha=ALPHA, rslora=True, irp_seed=IRP_SEED, author=5)
    tau = ltv.build_tangent(factors)
    assert all((t == 0).all() for t in tau.values())
    ids, am = make_batch(seed=4)
    got = ltv.linearized_forward(model, names, tau, ids, attention_mask=am).logits
    base = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    assert torch.equal(got, base), "zero tangent must serve the base model BIT-exactly"
    print("ok  (b) zero tangent -> bit-equal base logits")


def test_superposition(model, names):
    """(c) jvp is linear in tau => exact logit-space superposition (fp tolerance)."""
    tau1 = {n: t.detach() for n, t in ltv.build_tangent(rand_factors(model, names, 5)).items()}
    tau2 = {n: t.detach() for n, t in ltv.build_tangent(rand_factors(model, names, 7)).items()}
    ids, am = make_batch(seed=5)
    f0 = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    f1 = ltv.linearized_forward(model, names, tau1, ids, attention_mask=am).logits
    f2 = ltv.linearized_forward(model, names, tau2, ids, attention_mask=am).logits
    f12 = ltv.linearized_forward(
        model, names, {n: tau1[n] + tau2[n] for n in names}, ids, attention_mask=am).logits
    err = ((f12 - f0) - ((f1 - f0) + (f2 - f0))).abs().max().item()
    # fp tolerance is RELATIVE to the delta magnitude: a(t1+t2) vs a*t1 + a*t2 differ by ULPs
    rel = err / (f12 - f0).abs().max().clamp_min(1e-8).item()
    assert err < 1e-4 and rel < 1e-4, (err, rel)
    print(f"ok  (c) superposition f_lin(t1+t2)-f0 == sum of parts "
          f"(max abs err {err:.2e}, rel {rel:.2e})")


def save_adapters(model, names, tmp, authors=(5, 7, 9)):
    dirs = {}
    facs = {}
    for a in authors:
        f = rand_factors(model, names, a)
        d = os.path.join(tmp, f"shard_{a}")
        ltv.save_author_adapter(d, f, base_model_name="tiny", rank=RANK, alpha=ALPHA,
                                rslora=True)
        dirs[a], facs[a] = d, f
    return dirs, facs


def test_compose_subtract(dirs, facs, names):
    """(d) subtract == recompose (atol 1e-6); sorted deterministic accumulation."""
    d5, d7, d9 = dirs[5], dirs[7], dirs[9]
    c57 = ltv.compose_tangents([d5, d7], [1.0, 1.0])
    # deletion contract: full compose(5,7,9) + a weight -1.0 entry for 9 == recompose(5,7)
    c_del9 = ltv.compose_tangents([d5, d7, d9, d9], [1.0, 1.0, 1.0, -1.0])
    for n in names:
        assert torch.allclose(c_del9[n], c57[n], atol=1e-6), n
    # caller order must not change a single byte (internal sort by dirname)
    c75 = ltv.compose_tangents([d7, d5], [1.0, 1.0])
    for n in names:
        assert torch.equal(c57[n], c75[n]), n
    # and the composed sum equals the in-memory tangent sum through the file round-trip
    ref = ltv.build_tangent(facs[5])
    ref = {n: ref[n].detach() + ltv.build_tangent(facs[7])[n].detach() for n in names}
    worst = max((c57[n] - ref[n]).abs().max().item() for n in names)
    assert worst < 1e-5, worst
    print(f"ok  (d) subtract == recompose (atol 1e-6); order-independent bytes; "
          f"file round-trip err {worst:.2e}")


def test_backward_reaches_B(model, names):
    """(e) loss.backward() through f_lin reaches every B; A frozen."""
    factors = ltv.init_author_factors(
        model, names, rank=RANK, alpha=ALPHA, rslora=True, irp_seed=IRP_SEED, author=5)
    a_snapshot = {n: A.clone() for n, (A, _B, _s) in factors.items()}
    ids, am = make_batch(seed=6)
    labels = ids.clone()
    tau = ltv.build_tangent(factors)
    out = ltv.linearized_forward(model, names, tau, ids, attention_mask=am, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    for n, (A, B, _s) in factors.items():
        assert not A.requires_grad, f"{n}: A must stay frozen"
        assert torch.equal(A, a_snapshot[n]), f"{n}: A bytes changed"
        assert B.grad is not None and B.grad.abs().sum().item() > 0, f"{n}: no grad on B"
    print(f"ok  (e) backward reaches all {len(names)} B factors; A frozen (bytes + flag)")


def test_determinism(model, names):
    """(f) byte-identical seeded A; byte-equal trained B after 2 synthetic steps."""
    A1 = ltv.seeded_A((RANK, 64), IRP_SEED, 5, "model.layers.0.self_attn.q_proj")
    A2 = ltv.seeded_A((RANK, 64), IRP_SEED, 5, "model.layers.0.self_attn.q_proj")
    assert torch.equal(A1, A2)
    A3 = ltv.seeded_A((RANK, 64), IRP_SEED, 6, "model.layers.0.self_attn.q_proj")
    assert not torch.equal(A1, A3), "different authors must draw different A"

    gen = torch.Generator().manual_seed(11)
    rows = [torch.randint(1, VOCAB, (int(w),), generator=gen)
            for w in torch.randint(5, 10, (8,), generator=gen)]
    runs = []
    for _ in range(2):
        factors = ltv.init_author_factors(
            model, names, rank=RANK, alpha=ALPHA, rslora=True, irp_seed=IRP_SEED, author=5)
        # 8 rows, bsz 2, accum 2, 1 epoch => exactly 2 optimizer steps
        losses = train_linear_factors(
            model, names, factors, rows, pad_id=0, epochs=1, batch_size=2, grad_accum=2,
            lr=1e-3, weight_decay=0.001, clip=0.3, warmup_ratio=0.03, seed=42,
            device="cpu", log_every=100)
        assert len(losses) == 2, losses
        runs.append((factors, losses))
    (fa, la), (fb, lb) = runs
    assert la == lb, (la, lb)
    for n in names:
        assert torch.equal(fa[n][1], fb[n][1]), f"{n}: trained B not deterministic"
        assert fa[n][1].abs().sum().item() > 0, f"{n}: B never moved"
    print(f"ok  (f) seeded_A byte-identical; 2-step trained B byte-equal across runs "
          f"(losses {la[0]:.4f}, {la[1]:.4f})")


def test_generate(model, names):
    """(g) generate == stepwise argmax of linearized logits; eos + max_new_tokens honored."""
    tau = {n: t.detach() for n, t in ltv.build_tangent(rand_factors(model, names, 5)).items()}
    wrapper = ltv.LinearTVModel(model, None, names=names, tau=tau)
    gen = torch.Generator().manual_seed(12)
    prompt = torch.randint(1, VOCAB, (1, 5), generator=gen)
    steps = 4
    got = wrapper.generate(input_ids=prompt, max_new_tokens=steps, do_sample=False,
                           eos_token_id=-1)  # unreachable eos -> full length
    ids, am = prompt, torch.ones_like(prompt)
    for _ in range(steps):
        logits = ltv.linearized_forward(model, names, tau, ids, attention_mask=am).logits
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        am = torch.cat([am, am.new_ones(1, 1)], dim=1)
    assert got.shape == ids.shape and torch.equal(got, ids)
    # eos honored: declare the first generated token the eos -> stop after 1 new token
    first_new = int(ids[0, prompt.shape[1]].item())
    short = wrapper.generate(input_ids=prompt, max_new_tokens=steps, do_sample=False,
                             eos_token_id=first_new)
    assert short.shape[1] == prompt.shape[1] + 1, short.shape
    print(f"ok  (g) generate == stepwise argmax over f_lin ({steps} tokens); eos stops early")


def test_peft_interop(dirs, facs, names):
    """(h) saved PEFT dirs feed merge_subset._weighted_factor_cat; its dense delta == tau sum."""
    from merge_subset import _weighted_factor_cat
    d5, d7 = dirs[5], dirs[7]
    merged, ref_cfg, out_rank, meta = _weighted_factor_cat([d5, d7], [1.0, 1.0])
    assert out_rank == 2 * RANK, out_rank
    ref5 = ltv.build_tangent(facs[5])
    ref7 = ltv.build_tangent(facs[7])
    worst = 0.0
    for slot, (A_cat, B_cat) in merged.items():
        name = slot + ".weight"
        assert name in ref5, name
        dense = B_cat.float() @ A_cat.float()
        ref = (ref5[name] + ref7[name]).detach()
        worst = max(worst, (dense - ref).abs().max().item())
    assert set(slot + ".weight" for slot in merged) == set(names)
    assert worst < 1e-5, worst
    # and a REAL PeftModel round-trip (the eval-time --preloaded_adapter path): the served
    # effective delta scaling*B@A must equal the tangent this author trains against.
    from peft import PeftModel
    served = PeftModel.from_pretrained(LlamaForCausalLM(tiny_cfg()), d5, adapter_name="m")
    worst_p = 0.0
    for mod_name, mod in served.named_modules():
        if hasattr(mod, "lora_A") and "m" in mod.lora_A:
            assert abs(mod.scaling["m"] - ltv.lora_scaling(RANK, ALPHA, True)) < 1e-9
            eff = mod.scaling["m"] * (mod.lora_B["m"].weight.data.float()
                                      @ mod.lora_A["m"].weight.data.float())
            name = mod_name.replace("base_model.model.", "", 1) + ".weight"
            worst_p = max(worst_p, (eff - ref5[name].detach()).abs().max().item())
    assert worst_p < 1e-5, worst_p
    print(f"ok  (h) PEFT save -> _weighted_factor_cat delta == tangent sum "
          f"(max abs err {worst:.2e}; cat rank {out_rank}); PeftModel round-trip "
          f"({worst_p:.2e})")


def test_b_only_roundtrip(model, names, tmp):
    """(i) B-only save/load: A re-derived from seed == original factors."""
    factors = rand_factors(model, names, author=5)
    d = os.path.join(tmp, "b_only_shard")
    meta = {"author": 5, "irp_seed": IRP_SEED, "rank": RANK, "alpha": ALPHA, "rslora": True,
            "in_features": {n: A.shape[1] for n, (A, _B, _s) in factors.items()}}
    ltv.save_b_only(d, {n: B for n, (_A, B, _s) in factors.items()}, meta)
    loaded, meta2 = ltv.load_author_factors(d)
    assert set(loaded) == set(factors) and meta2["author"] == 5
    for n in names:
        A0, B0, s0 = factors[n]
        A1, B1, s1 = loaded[n]
        assert torch.equal(A0, A1), f"{n}: re-derived A differs"
        assert torch.equal(B0.detach(), B1), f"{n}: B round-trip differs"
        assert abs(s0 - s1) < 1e-12
    # meta completeness is enforced
    try:
        ltv.save_b_only(d, {}, {"author": 5})
        raise AssertionError("save_b_only accepted incomplete meta")
    except ValueError:
        pass
    print("ok  (i) B-only round-trip: A re-derived byte-equal, B equal, meta enforced")


def test_nonlinear_debug(model, names):
    """(+) nonlinear-debug == direct weight addition; xi metric well-behaved + restores."""
    tau1 = {n: t.detach() for n, t in ltv.build_tangent(rand_factors(model, names, 5)).items()}
    tau2 = {n: t.detach() for n, t in ltv.build_tangent(rand_factors(model, names, 7)).items()}
    ids, am = make_batch(seed=8)

    m2 = copy.deepcopy(model)
    wrapper = ltv.LinearTVModel(m2, None, names=names, tau=tau1, serve="nonlinear-debug")
    got = wrapper.forward(input_ids=ids, attention_mask=am).logits
    with ltv.applied_tau(model, names, tau1):
        ref = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    assert torch.equal(got, ref), "nonlinear-debug serve != direct weight addition"

    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    zero = {n: torch.zeros_like(tau1[n]) for n in names}
    xi0 = ltv.disentanglement_error(model, names, tau1, zero, {"input_ids": ids,
                                                               "attention_mask": am})
    assert xi0 == 0.0, xi0   # f(t+0)-f(t)-f(0)+f(0) with byte-restored weights is exact
    xi = ltv.disentanglement_error(model, names, tau1, tau2, {"input_ids": ids,
                                                              "attention_mask": am})
    assert xi >= 0.0 and torch.isfinite(torch.tensor(xi))
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), snapshot[n]), f"{n}: weights not restored"
    print(f"ok  (+) nonlinear-debug serve exact; xi(t,0)=0, xi(t1,t2)={xi:.4f}, "
          f"weights byte-restored")


def test_resolve_compose():
    """(+) eval-seam selection logic: pool via merge_subset.subset_authors, sum/mean
    weights, subtraction, error paths — no model download needed."""
    from merge_subset import subset_authors
    cfg = {"out_dir": "OUT", "pool_seed": 42, "pool_size": 20}
    pool4 = subset_authors(42, 4)
    dirs, w = ltv.resolve_compose(cfg, n=4)
    assert dirs == [os.path.join("OUT", f"shard_{a}") for a in pool4]
    assert w == [1.0] * 4
    dirs, w = ltv.resolve_compose(cfg, authors="5,7", subtract="9")
    assert dirs == [os.path.join("OUT", "shard_5"), os.path.join("OUT", "shard_7"),
                    os.path.join("OUT", "shard_9")]
    assert w == [1.0, 1.0, -1.0]
    _, w = ltv.resolve_compose({**cfg, "compose": "mean"}, n=4, subtract="9")
    assert w == [0.25] * 4 + [-0.25]
    for bad in (dict(n=21), dict(n=0), dict(authors="5", n=2), dict(),
                dict(subtract="9")):
        try:
            ltv.resolve_compose(cfg, **bad)
            raise AssertionError(f"resolve_compose accepted {bad}")
        except ValueError:
            pass
    print(f"ok  (+) resolve_compose: pool(n=4)={pool4}, sum/mean weights, errors raised")


def test_eval_tofu_flags():
    """(j) new eval_tofu flags default off; existing defaults untouched (no eval run)."""
    import eval_tofu
    argv = sys.argv
    try:
        sys.argv = ["eval_tofu.py", "--model_name", "m", "--output_dir", "o",
                    "--label", "l", "--out", "r.json"]
        args = eval_tofu.parse_args()
    finally:
        sys.argv = argv
    assert args.linear_tv_config is None
    assert args.linear_tv_authors is None
    assert args.linear_tv_n is None
    assert args.linear_tv_subtract is None
    assert args.linear_tv_serve == "linear"
    # existing surface unchanged (spot-check the defaults other arms rely on)
    assert args.k == 10 and args.forget_shard_id is None and args.eval_shard_id is None
    assert args.preloaded_adapter is None and args.prefix_pool_dir is None
    assert args.legonet_config is None and args.legonet_unlearn_tag is None
    assert args.sift_masks_config is None and args.clamu_config is None
    assert args.memsinks_config is None and args.ramole_router is None
    assert args.ramole_route == "embed" and args.ramole_index == "stale"
    assert args.retain_author_ids is None and not args.smoke and not args.extended
    print("ok  (j) eval_tofu: --linear_tv_* default to absent; existing defaults unchanged")


def main():
    tmp = tempfile.mkdtemp(prefix="test_linear_tv_")
    try:
        model = build_model()
        assert model.config._attn_implementation == "eager"
        names = ltv.target_names(model)
        assert len(names) == 2 * 6, names   # 2 layers x [q,k,v,o,up,down]
        dirs, facs = save_adapters(model, names, tmp)
        test_closed_form(model, names)
        test_zero_tangent(model, names)
        test_superposition(model, names)
        test_compose_subtract(dirs, facs, names)
        test_backward_reaches_B(model, names)
        test_determinism(model, names)
        test_generate(model, names)
        test_peft_interop(dirs, facs, names)
        test_b_only_roundtrip(model, names, tmp)
        test_nonlinear_debug(model, names)
        test_resolve_compose()
        test_eval_tofu_flags()
        print("ALL OK  test_linear_tv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
