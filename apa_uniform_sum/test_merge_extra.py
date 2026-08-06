"""CPU micro-tests for merge_extra.py + merge_lora dispatch (no downloads, no GPU).

Run after touching merge code: python test_merge_extra.py

Builds a tiny random Llama with k=3 LoRA shard adapters (rslora, like training) and
checks each new method against closed-form references on the *effective* deltas
(scaling * B @ A) — this is what catches scaling bugs that raw-factor comparisons miss.
"""

import math

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

import merge_lora
from merge_extra import (
    breadcrumbs_merge_adapters,
    della_merge_adapters,
    fisher_merge_shards,
    knots_merge_adapters,
    lorahub_merge_shards,
    slerp_merge_pair,
    subtract_orth_adapters,
    tsv_merge_adapters,
)

K = 3
RANK = 8
VOCAB = 128

torch.manual_seed(0)


def build_model(identical=False):
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB,
        max_position_embeddings=64,
    )
    base = LlamaForCausalLM(cfg)
    # Mirror train_lora_shard's config (rslora => scaling = alpha/sqrt(r) != 1,
    # which is exactly what the scaling-correctness checks need).
    lora_cfg = LoraConfig(
        r=RANK, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    )
    model = get_peft_model(base, lora_cfg, adapter_name="shard_0")
    for i in range(1, K):
        model.add_adapter(f"shard_{i}", lora_cfg)
    gen = torch.Generator().manual_seed(42)
    for module in lora_modules(model):
        for i in range(K):
            name = f"shard_{i}"
            src = "shard_0" if identical else name
            if identical and i > 0:
                module.lora_A[name].weight.data.copy_(module.lora_A[src].weight.data)
                module.lora_B[name].weight.data.copy_(module.lora_B[src].weight.data)
            else:
                for fac in (module.lora_A[name].weight, module.lora_B[name].weight):
                    fac.data.normal_(0.0, 0.05, generator=gen)
    return model


def lora_modules(model, name="shard_0"):
    return [m for _, m in model.named_modules()
            if hasattr(m, "lora_A") and name in m.lora_A]


def eff_delta(module, name):
    return module.scaling[name] * (
        module.lora_B[name].weight.data.float() @ module.lora_A[name].weight.data.float()
    )


def svd_trunc(mat, rank):
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank]


def max_rel_err(got, ref):
    denom = ref.abs().max().clamp_min(1e-8)
    return ((got - ref).abs().max() / denom).item()


def make_loader(n_batches=4, batch=2, seqlen=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        ids = torch.randint(3, VOCAB, (batch, seqlen), generator=gen)
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return out


def test_della_linear_limit(model):
    """della(density=1, epsilon=0) must equal PEFT linear bit-for-bit (same formula)."""
    ref = merge_lora.merge_shards(model, K, "linear", adapter_name="ref_linear")
    got = della_merge_adapters(
        model, [f"shard_{i}" for i in range(K)],
        adapter_name="della_lim", density=1.0, epsilon=0.0,
    )
    for m in lora_modules(model):
        assert torch.equal(m.lora_A[got].weight.data, m.lora_A[ref].weight.data)
        assert torch.equal(m.lora_B[got].weight.data, m.lora_B[ref].weight.data)
    print("ok  della(density=1, eps=0) == PEFT linear")


def test_della_determinism(model):
    names = [f"shard_{i}" for i in range(K)]
    a = della_merge_adapters(model, names, adapter_name="della_s7a", density=0.5, seed=7)
    b = della_merge_adapters(model, names, adapter_name="della_s7b", density=0.5, seed=7)
    c = della_merge_adapters(model, names, adapter_name="della_s8", density=0.5, seed=8)
    m = lora_modules(model)[0]
    assert torch.equal(m.lora_A[a].weight.data, m.lora_A[b].weight.data)
    assert not torch.equal(m.lora_A[a].weight.data, m.lora_A[c].weight.data)
    for mm in lora_modules(model):
        assert torch.isfinite(mm.lora_A[a].weight.data).all()
        assert torch.isfinite(mm.lora_B[a].weight.data).all()
    print("ok  della deterministic per seed, finite")


def test_della_ties_runs(model):
    got = merge_lora.merge_shards(model, K, "della_ties", density=0.5, seed=0)
    m = lora_modules(model)[0]
    assert torch.isfinite(m.lora_A[got].weight.data).all()
    assert m.lora_B[got].weight.data.abs().sum() > 0
    print("ok  della_ties via merge_shards (label name:", got + ")")


def test_breadcrumbs_limit(model):
    """breadcrumbs(gamma=0) must equal PEFT magnitude_prune at the same density."""
    ref = merge_lora.merge_shards(model, K, "magnitude_prune", density=0.7,
                                  adapter_name="ref_magprune")
    got = breadcrumbs_merge_adapters(
        model, [f"shard_{i}" for i in range(K)],
        adapter_name="bc_g0", density=0.7, gamma=0.0,
    )
    for m in lora_modules(model):
        assert torch.allclose(m.lora_A[got].weight.data, m.lora_A[ref].weight.data, atol=1e-7)
        assert torch.allclose(m.lora_B[got].weight.data, m.lora_B[ref].weight.data, atol=1e-7)
    print("ok  breadcrumbs(gamma=0) == PEFT magnitude_prune")


def test_breadcrumbs_scale_label(model):
    """merged_breadcrumbs_s{λ} must apply weights [λ]*n (additive's _s convention);
    the bare label keeps the uniform 1/n default (λ ignored was the silent-no-op bug)."""
    lam = 0.2
    got = merge_lora.activate_label(model, K, K - 1, f"merged_breadcrumbs_s{lam}")
    ref = breadcrumbs_merge_adapters(
        model, [f"shard_{i}" for i in range(K)],
        adapter_name="bc_wref", weights=[lam] * K,
    )
    for m in lora_modules(model):
        assert torch.allclose(m.lora_A[got].weight.data, m.lora_A[ref].weight.data, atol=1e-7)
        assert torch.allclose(m.lora_B[got].weight.data, m.lora_B[ref].weight.data, atol=1e-7)
    bare = merge_lora.activate_label(model, K, K - 1, "merged_breadcrumbs")
    uni = breadcrumbs_merge_adapters(
        model, [f"shard_{i}" for i in range(K)],
        adapter_name="bc_uni", weights=[1.0 / K] * K,
    )
    m0 = lora_modules(model)[0]
    assert torch.allclose(m0.lora_B[bare].weight.data, m0.lora_B[uni].weight.data, atol=1e-7)
    assert not torch.allclose(m0.lora_B[got].weight.data, m0.lora_B[bare].weight.data)
    print(f"ok  merged_breadcrumbs_s{lam} == explicit weights [{lam}]*{K}; bare label stays 1/n")


def test_knots_single_identity(model):
    """KnOTS over one adapter at density=1 must reproduce its effective delta."""
    got = knots_merge_adapters(model, ["shard_0"], adapter_name="knots_1", density=1.0)
    errs = [max_rel_err(eff_delta(m, got), eff_delta(m, "shard_0")) for m in lora_modules(model)]
    assert max(errs) < 1e-3, f"knots single-adapter identity err {max(errs)}"
    print(f"ok  knots_ties n=1 identity (max rel err {max(errs):.2e})")


def test_knots_identical_copies(model_id):
    """3 identical shards at density=1: TIES averaging convention gives delta/3
    (same normalization the registry's `ties` has with uniform 1/n weights)."""
    got = knots_merge_adapters(
        model_id, [f"shard_{i}" for i in range(K)], adapter_name="knots_id", density=1.0,
    )
    errs = [max_rel_err(eff_delta(m, got), eff_delta(m, "shard_0") / K)
            for m in lora_modules(model_id)]
    assert max(errs) < 1e-3, f"knots identical-copies err {max(errs)}"
    print(f"ok  knots_ties identical copies -> delta/3 (max rel err {max(errs):.2e})")


def test_tsv_identity(model, model_id):
    got1 = tsv_merge_adapters(model, ["shard_0"], adapter_name="tsv_1")
    errs = [max_rel_err(eff_delta(m, got1), eff_delta(m, "shard_0")) for m in lora_modules(model)]
    assert max(errs) < 1e-2, f"tsv single-adapter identity err {max(errs)}"
    # Whitening absorbs redundancy: n identical tasks (SUM semantics) -> delta, not n*delta.
    gotn = tsv_merge_adapters(
        model_id, [f"shard_{i}" for i in range(K)], adapter_name="tsv_id",
    )
    errs_n = [max_rel_err(eff_delta(m, gotn), eff_delta(m, "shard_0"))
              for m in lora_modules(model_id)]
    assert max(errs_n) < 1e-2, f"tsv identical-copies identity err {max(errs_n)}"
    print(f"ok  tsv identities (n=1 err {max(errs):.2e}, n=3-identical err {max(errs_n):.2e})")


def test_slerp_identical(model_id):
    got = slerp_merge_pair(model_id, "shard_0", "shard_1", adapter_name="slerp_id")
    errs = [max_rel_err(eff_delta(m, got), eff_delta(m, "shard_0"))
            for m in lora_modules(model_id)]
    assert max(errs) < 1e-3, f"slerp identical-pair identity err {max(errs)}"
    print(f"ok  slerp of identical pair == original (max rel err {max(errs):.2e})")


def test_slerp_distinct(model):
    got = slerp_merge_pair(model, "shard_0", "shard_1", adapter_name="slerp_ab")
    for m in lora_modules(model):
        d = eff_delta(m, got)
        assert torch.isfinite(d).all()
        assert d.abs().sum() > 0
    print("ok  slerp of distinct pair finite/nonzero")


def test_subtract_orth(model):
    forget = f"shard_{K - 1}"
    names = [f"shard_{i}" for i in range(K)]
    got = subtract_orth_adapters(model, names, forget, adapter_name="sub_orth")
    max_left, max_right, max_ref = 0.0, 0.0, 0.0
    for m in lora_modules(model):
        delta = eff_delta(m, got)
        Uf, _ = torch.linalg.qr(m.lora_B[forget].weight.data.float())
        Vf, _ = torch.linalg.qr(m.lora_A[forget].weight.data.float().t())
        scale = delta.abs().max().clamp_min(1e-8)
        max_left = max(max_left, ((Uf.t() @ delta).abs().max() / scale).item())
        max_right = max(max_right, ((delta @ Vf).abs().max() / scale).item())
        # Reference: project the dense mean both sides, then best rank-r truncation.
        mean = sum(eff_delta(m, s) for s in names) / K
        ref = mean - Uf @ (Uf.t() @ mean)
        ref = ref - (ref @ Vf) @ Vf.t()
        max_ref = max(max_ref, max_rel_err(delta, svd_trunc(ref, RANK)))
    assert max_left < 1e-4, f"left-subspace leak {max_left}"
    assert max_right < 1e-4, f"right-subspace leak {max_right}"
    assert max_ref < 1e-3, f"subtract_orth vs dense reference err {max_ref}"
    print(f"ok  subtract_orth: forget-subspace leak <1e-4, matches dense ref "
          f"(err {max_ref:.2e})")


def test_additive(model):
    """Additive = true-scale SUM of effective deltas, full rank, no compression.

    merged_additive delta == sum_i scaling_i B_i A_i exactly (the corrected linear/cat:
    no sqrt(r) inflation, no rank-r squash); remerge drops PRECISELY the forget term
    (the additive-exactness invariant the unlearning scheme relies on); deterministic.
    """
    names = [f"shard_{i}" for i in range(K)]
    forget = f"shard_{K - 1}"

    merged = merge_lora.merge_shards(model, K, "additive", adapter_name="add_all")
    errs = []
    for m in lora_modules(model):
        ref = sum(eff_delta(m, s) for s in names)              # SUM, true scale
        errs.append(max_rel_err(eff_delta(m, merged), ref))
    assert max(errs) < 1e-4, f"additive merged != sum of effective deltas: {max(errs)}"

    remerged = merge_lora.merge_shards(model, K, "additive", exclude_shard=K - 1,
                                       adapter_name="add_rem")
    errs_sum, errs_drop = [], []
    for m in lora_modules(model):
        ref = sum(eff_delta(m, s) for s in names if s != forget)
        errs_sum.append(max_rel_err(eff_delta(m, remerged), ref))
        # Exact-recovery: (merged) - (forget term) == remerge, bit-for-the-math.
        drop = eff_delta(m, merged) - eff_delta(m, forget)
        errs_drop.append(max_rel_err(eff_delta(m, remerged), drop))
    assert max(errs_sum) < 1e-4, f"additive remerge != sum of survivors: {max(errs_sum)}"
    assert max(errs_drop) < 1e-4, f"additive drop-term invariant broke: {max(errs_drop)}"

    # Determinism: same inputs -> identical factors (no RNG anywhere in the method).
    again = merge_lora.merge_shards(model, K, "additive", adapter_name="add_all2")
    m0 = lora_modules(model)[0]
    assert torch.equal(m0.lora_A[merged].weight.data, m0.lora_A[again].weight.data)
    assert torch.equal(m0.lora_B[merged].weight.data, m0.lora_B[again].weight.data)

    # additive_mean = (1/n_active) * sum of effective deltas (soup regime); remerge
    # re-normalizes over the kept set (1/(n-1)), not subtract-a-term.
    mean_all = merge_lora.merge_shards(model, K, "additive_mean", adapter_name="mean_all")
    mean_rem = merge_lora.merge_shards(model, K, "additive_mean", exclude_shard=K - 1,
                                       adapter_name="mean_rem")
    errs_mean, errs_mrem = [], []
    for m in lora_modules(model):
        ref_all = sum(eff_delta(m, s) for s in names) / K
        ref_rem = sum(eff_delta(m, s) for s in names if s != forget) / (K - 1)
        errs_mean.append(max_rel_err(eff_delta(m, mean_all), ref_all))
        errs_mrem.append(max_rel_err(eff_delta(m, mean_rem), ref_rem))
    assert max(errs_mean) < 1e-4, f"additive_mean merged != (1/n) sum: {max(errs_mean)}"
    assert max(errs_mrem) < 1e-4, f"additive_mean remerge != (1/(n-1)) sum: {max(errs_mrem)}"

    # Global-λ scale (the `_s{λ}` label / scale= arg): delta == λ·Σ; remerge == λ·Σ_kept
    # (fixed-λ keeps exact drop-a-term). Test both the merge_shards arg and the label path.
    LAM = 0.2
    sc = merge_lora.merge_shards(model, K, "additive", scale=LAM, adapter_name="add_s")
    sc_rem = merge_lora.merge_shards(model, K, "additive", scale=LAM, exclude_shard=K - 1,
                                     adapter_name="add_s_rem")
    errs_s, errs_sr = [], []
    for m in lora_modules(model):
        errs_s.append(max_rel_err(eff_delta(m, sc), LAM * sum(eff_delta(m, s) for s in names)))
        errs_sr.append(max_rel_err(eff_delta(m, sc_rem),
                                   LAM * sum(eff_delta(m, s) for s in names if s != forget)))
    assert max(errs_s) < 1e-4, f"additive scale merged != λ·Σ: {max(errs_s)}"
    assert max(errs_sr) < 1e-4, f"additive scale remerge != λ·Σ_kept: {max(errs_sr)}"
    assert merge_lora.activate_label(model, K, K - 1, "merged_additive_s0.2") == "merged_additive_s0p2"
    assert merge_lora.activate_label(model, K, K - 1, "remerge_additive_s0.2") == "remerge_additive_s0p2"

    print(f"ok  additive: merged==sum (err {max(errs):.2e}), "
          f"remerge==drop-forget-term (err {max(errs_drop):.2e}), deterministic; "
          f"additive_mean==(1/n)sum (err {max(errs_mean):.2e})")


def test_fisher(model, model_id):
    loader = make_loader()
    got = fisher_merge_shards(model, K, loader, num_examples=8, adapter_name="fisher_t")
    for m in lora_modules(model):
        assert torch.isfinite(m.lora_A[got].weight.data).all()
        assert torch.isfinite(m.lora_B[got].weight.data).all()
    # Identity: identical shards have identical Fisher -> weighted avg returns the shard.
    got_id = fisher_merge_shards(model_id, K, loader, num_examples=8,
                                 adapter_name="fisher_id")
    errs = [max_rel_err(eff_delta(m, got_id), eff_delta(m, "shard_0"))
            for m in lora_modules(model_id)]
    assert max(errs) < 1e-2, f"fisher identical-shards identity err {max(errs)}"
    print(f"ok  fisher finite + identical-shards identity (max rel err {max(errs):.2e})")


def test_lorahub(model):
    loader = make_loader()
    got = lorahub_merge_shards(model, K, loader, num_examples=4, budget=5,
                               adapter_name="lorahub_a", seed=0)
    m0 = lora_modules(model)[0]
    A_first = m0.lora_A[got].weight.data.clone()
    assert torch.isfinite(A_first).all()
    got2 = lorahub_merge_shards(model, K, loader, num_examples=4, budget=5,
                                adapter_name="lorahub_b", seed=0)
    assert torch.equal(m0.lora_A[got2].weight.data, A_first), "lorahub not deterministic"
    print("ok  lorahub runs and is deterministic per seed")


def test_dispatch_and_labels(model):
    # density-suffix label path
    res = merge_lora.activate_label(model, K, K - 1, "merged_della_linear_d0.5")
    assert isinstance(res, str) and res == "merged_della_linear_d0p5"
    res = merge_lora.activate_label(model, K, K - 1, "remerge_breadcrumbs")
    assert isinstance(res, str)
    res = merge_lora.activate_label(model, K, K - 1, "subtract_orth")
    assert res == "unlearn_subtract_orth"
    # additive labels resolve through the merged_/remerge_ prefix path
    assert merge_lora.activate_label(model, K, K - 1, "merged_additive") == "merged_additive"
    assert merge_lora.activate_label(model, K, K - 1, "remerge_additive") == "remerge_additive"
    assert merge_lora.activate_label(model, K, K - 1, "merged_additive_mean") == "merged_additive_mean"
    assert merge_lora.activate_label(model, K, K - 1, "remerge_additive_mean") == "remerge_additive_mean"
    assert not merge_lora.label_requires_data("merged_additive")
    # JD labels: variant from method name, _cN/_rR suffixes parsed
    assert merge_lora.activate_label(model, K, K - 1, "merged_jd_full") == "merged_jd_full"
    assert merge_lora.activate_label(model, K, K - 1, "remerge_jd_diag_c2") == "remerge_jd_diag_c2"
    assert merge_lora.activate_label(model, K, K - 1, "merged_jd_full_c2_r4") == "merged_jd_full_c2_r4"
    # JD is rejected for tree merging (it compresses across the whole collection)
    try:
        merge_lora.merge_tree(model, K, "jd_full")
        raise AssertionError("jd_full in tree should raise")
    except ValueError:
        pass
    # tree merges, incl. pairwise slerp
    root = merge_lora.merge_tree(model, K, "slerp")
    assert isinstance(root, str)
    rem = merge_lora.remerge_tree_path(model, K, K - 1, "slerp")
    assert isinstance(rem, str)
    root2 = merge_lora.merge_tree(model, K, "della_ties")
    assert isinstance(root2, str)
    # flat slerp must reject n != 2
    try:
        merge_lora.merge_shards(model, K, "slerp")
        raise AssertionError("flat slerp with k=3 should raise")
    except ValueError:
        pass
    # data-required guards
    assert merge_lora.label_requires_data("merged_fisher")
    assert merge_lora.label_requires_data("remerge_lorahub")
    assert merge_lora.label_requires_data("merged_regmean")
    assert not merge_lora.label_requires_data("merged_della_ties")
    try:
        merge_lora.merge_shards(model, K, "fisher")
        raise AssertionError("fisher without dataloader should raise")
    except ValueError:
        pass
    print("ok  dispatch: labels, density suffix, trees, slerp/data guards")


def test_jd(model, model_id):
    """Joint Diagonalization: lossless at full rank (Prop. 1), Diag >= Full recon error,
    the O(1)-deletion identity, and identical-shards == the single (true-scale) delta."""
    import jd_compress
    from merge_extra import _gather_jd_slots, jd_merge_adapters

    names = [f"shard_{i}" for i in range(K)]
    slots, _ = _gather_jd_slots(model, names)

    # (a) lossless reconstruction at full rank (rank >= sum of source ranks)
    jd_lossless = jd_compress.jd_compress_collection(
        slots, names, variant="full", clusters=1, rank=K * RANK, iters=15)
    assert jd_lossless.reconstruction_error() < 1e-3, jd_lossless.reconstruction_error()

    # (b) Diag has >= reconstruction error of Full at equal (modest) rank
    jf = jd_compress.jd_compress_collection(
        slots, names, variant="full", clusters=1, rank=4, iters=12)
    jg = jd_compress.jd_compress_collection(
        slots, names, variant="diag", clusters=1, rank=4, iters=12)
    assert jg.reconstruction_error() + 1e-4 >= jf.reconstruction_error()

    # (c) O(1)-deletion identity: dropping shard f from the keep-set equals subtracting
    #     its norm_f * Sigma_f from the cluster sum (single cluster), no refit.
    f = K - 1
    keep = [i for i in range(K) if i != f]
    merged = jf.merge_keepset(keep, weights=[1.0] * len(keep), out_rank=K * RANK)
    sname = next(iter(jf.slots))
    js = jf.slots[sname]
    sig = sum(float(jf.norm[i]) * js.sigma[i] for i in keep)
    direct = js.U[0] @ sig @ js.V[0].t()
    A_new, B_new = merged[sname]
    assert max_rel_err(B_new @ A_new, direct) < 1e-3

    # (d) identical shards: uniform JD merge == the single delta (true scale, no
    #     sqrt(r) inflation), through the live-model path.
    jd_merge_adapters(model_id, names, adapter_name="jd_id", variant="full",
                      clusters=1, rank=RANK)
    worst = max(max_rel_err(eff_delta(m, "jd_id"), eff_delta(m, "shard_0"))
                for m in lora_modules(model_id))
    assert worst < 1e-2, worst

    # (e) clustering (A.3) is deterministic and recon error is non-increasing as the
    #     cluster count rises (paper Fig 6 / H.1); c=n reduces to per-adapter SVD (lowest).
    rc = [jd_compress.jd_compress_collection(
              slots, names, variant="full", clusters=c, rank=4, iters=12, seed=0
          ).reconstruction_error() for c in (1, 2, K)]
    assert rc[0] + 1e-3 >= rc[1] >= rc[2] - 1e-3, rc
    a = jd_compress.jd_compress_collection(slots, names, variant="full", clusters=2,
                                           rank=4, seed=0).assignment
    b = jd_compress.jd_compress_collection(slots, names, variant="full", clusters=2,
                                           rank=4, seed=0).assignment
    assert a == b, (a, b)

    # (f) recommend_jd_settings follows the paper's regimes (Section 6.5 / F).
    assert jd_compress.recommend_jd_settings(10) == {"variant": "full", "clusters": 1, "rank": 12}
    assert jd_compress.recommend_jd_settings(50) == {"variant": "full", "clusters": 1, "rank": 32}
    big = jd_compress.recommend_jd_settings(500)
    assert big["variant"] == "full" and big["rank"] == 16 and big["clusters"] > 1

    print(f"ok  jd: lossless@fullrank (err {jd_lossless.reconstruction_error():.2e}), "
          f"diag>=full, O(1)-deletion, identical==single (err {worst:.2e}), "
          f"cluster recon {[round(x,3) for x in rc]} non-increasing")


def main():
    model = build_model()
    model_id = build_model(identical=True)

    test_della_linear_limit(model)
    test_della_determinism(model)
    test_della_ties_runs(model)
    test_breadcrumbs_limit(model)
    test_breadcrumbs_scale_label(model)
    test_knots_single_identity(model)
    test_knots_identical_copies(model_id)
    test_tsv_identity(model, model_id)
    test_slerp_identical(model_id)
    test_slerp_distinct(model)
    test_subtract_orth(model)
    test_additive(model)
    test_jd(model, model_id)
    test_fisher(model, model_id)
    test_lorahub(model)
    test_dispatch_and_labels(model)

    print("\nall merge_extra tests passed")


if __name__ == "__main__":
    main()
