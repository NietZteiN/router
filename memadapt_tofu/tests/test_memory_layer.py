"""CPU unit tests for ProductKeyMemory + MemAdaptMLP (gate G0).

Run: cd <repo>/memadapt_tofu && python -m pytest tests/ -q
All tests are CPU/fp32 so bitwise claims are meaningful (house convention).
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memadapt_common import assignment_sha, combine_index, split_index
from memadapt_model import (
    MemAdaptMLP,
    apply_blocklist_file,
    freeze_base,
    get_adapter,
    install_adapter,
    load_memory_from_checkpoint,
    save_checkpoint,
)
from memory_layer import ProductKeyMemory

torch.manual_seed(0)

N_SQRT = 16          # tiny table: N = 256 entries
KEY_DIM = 8
HIDDEN = 32
VALUE_DIM = 16
TOPK = 4


def tiny_memory(topk=TOPK, half_topk=TOPK, seed=7):
    return ProductKeyMemory(
        hidden=HIDDEN, n_sqrt=N_SQRT, key_dim=KEY_DIM, topk=topk,
        half_topk=half_topk, value_dim=VALUE_DIM, router_seed=seed,
    )


def tiny_assignment(mem, n_sources=4, per_source=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(mem.n_entries, generator=g)
    assigned_idx = perm[: n_sources * per_source].sort().values
    owner = torch.arange(n_sources).repeat_interleave(per_source)
    # owner order is arbitrary relative to sorted entries — shuffle it so tests
    # don't accidentally rely on contiguity.
    owner = owner[torch.randperm(owner.numel(), generator=g)]
    mem.load_assignment(assigned_idx, owner)
    return assigned_idx, owner


def brute_force_route(mem, x, blocked_entries=None):
    """Reference WITHOUT blocking: score every entry of the full table.

    With half_topk >= topk and no block-list, the product-key path must match
    this exactly. Under a block-list the deployed semantics differ by design:
    blocking applies on the candidate grid built from per-half shortlists (see
    grid_reference_route), because blocked pairs can crowd the shortlists —
    the documented inexactness of product-key retrieval under blocking.
    """
    assert blocked_entries is None, "use grid_reference_route for blocking"
    with torch.no_grad():
        q = F.layer_norm(F.linear(x.float(), mem.w_q), (mem.key_dim,))
        s1 = F.linear(q[..., : mem.half], mem.k1)
        s2 = F.linear(q[..., mem.half:], mem.k2)
        full = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)
        scores, idx = full.topk(mem.topk, dim=-1)
        w = F.softmax(scores, dim=-1)
    return idx, w, scores


def grid_reference_route(mem, x, blocked_entries):
    """Independent reimplementation of the deployed blocking semantics:
    per-half top-k' shortlists -> Cartesian grid -> -inf on blocked -> top-k.
    """
    with torch.no_grad():
        q = F.layer_norm(F.linear(x.float(), mem.w_q), (mem.key_dim,))
        s1 = F.linear(q[..., : mem.half], mem.k1)
        s2 = F.linear(q[..., mem.half:], mem.k2)
        v1, i1 = s1.topk(mem.half_topk, dim=-1)
        v2, i2 = s2.topk(mem.half_topk, dim=-1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
        cand_idx = (i1.unsqueeze(-1) * mem.n_sqrt + i2.unsqueeze(-2)).flatten(-2)
        blocked_mask = torch.zeros(mem.n_entries, dtype=torch.bool)
        blocked_mask[torch.tensor(blocked_entries)] = True
        cand = cand.masked_fill(blocked_mask[cand_idx], float("-inf"))
        scores, pos = cand.topk(mem.topk, dim=-1)
        idx = cand_idx.gather(-1, pos)
        w = F.softmax(scores, dim=-1)
        w = w.masked_fill(torch.isneginf(scores[..., :1]), 0.0)
    return idx, w, scores


def test_index_codec_roundtrip():
    idx = torch.arange(N_SQRT * N_SQRT)
    i1, i2 = split_index(idx, N_SQRT)
    assert torch.equal(combine_index(i1, i2, N_SQRT), idx)


def test_product_key_topk_matches_brute_force():
    mem = tiny_memory()
    x = torch.randn(2, 5, HIDDEN)
    idx, w = mem.route(x)
    bf_idx, bf_w, bf_scores = brute_force_route(mem, x)
    # fp32 random scores: ties are measure-zero, ordering must match exactly.
    assert torch.equal(idx, bf_idx)
    assert torch.allclose(w, bf_w, atol=1e-6)


def test_product_key_topk_with_ties():
    # Duplicate key rows create exact score ties; selected score VALUES must
    # still match brute force even if tied indices are ordered differently.
    mem = tiny_memory()
    with torch.no_grad():
        mem.k1[1] = mem.k1[0]
        mem.k2[3] = mem.k2[2]
    x = torch.randn(1, 3, HIDDEN)
    idx, w = mem.route(x)
    _, _, bf_scores = brute_force_route(mem, x)
    with torch.no_grad():
        q = F.layer_norm(F.linear(x.float(), mem.w_q), (mem.key_dim,))
        s1 = F.linear(q[..., : mem.half], mem.k1)
        s2 = F.linear(q[..., mem.half:], mem.k2)
        i1, i2 = split_index(idx, N_SQRT)
        sel_scores = s1.gather(-1, i1) + s2.gather(-1, i2)
    assert torch.allclose(sel_scores.sort(-1, descending=True).values,
                          bf_scores, atol=1e-6)


def test_zero_init_noop_bitwise():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        base_logits = model(ids).logits.clone()

    mem = ProductKeyMemory(hidden=HIDDEN, n_sqrt=N_SQRT, key_dim=KEY_DIM,
                           topk=TOPK, half_topk=TOPK, value_dim=HIDDEN,
                           router_seed=7)
    tiny_assignment(mem)
    install_adapter(model, mem, layer_idx=2)
    with torch.no_grad():
        adapted_logits = model(ids).logits
    # values are all zero -> embedding_bag output is exactly 0.0 -> bitwise no-op.
    assert torch.equal(base_logits, adapted_logits)


def test_gradient_mask_isolation_and_correctness():
    mem = tiny_memory()
    tiny_assignment(mem, n_sources=4, per_source=8)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])  # keep pad row zero
    mem.values.requires_grad_(True)

    b, t = 3, 6
    x = torch.randn(b, t, HIDDEN)
    source_ids = torch.tensor([0, 2, 0])  # mixed-source batch, sources 1/3 absent
    upstream = torch.randn(b, t, VALUE_DIM)

    out = mem(x, source_ids=source_ids)
    (out * upstream).sum().backward()
    grad = mem.values.grad.clone()

    owner_compact = torch.full((mem.values.shape[0],), -1, dtype=torch.long)
    assigned = (mem.owner_full >= 0).nonzero(as_tuple=True)[0]
    owner_compact[mem.remap[assigned]] = mem.owner_full[assigned]
    owned_rows = torch.isin(owner_compact, source_ids)

    assert torch.equal(grad[~owned_rows], torch.zeros_like(grad[~owned_rows])), (
        "gradient leaked to rows not owned by any batch source"
    )
    assert grad[-1].abs().sum() == 0, "pad row must never receive gradient"

    # Semantics reference: row r's gradient must equal the UNMASKED gradient
    # computed from owner(r)'s own sequences only. Cross-source reads (e.g. a
    # source-0 sequence reading a source-2 row) contribute forward value but
    # never gradient — the paper's isolation guarantee.
    for src in source_ids.unique():
        sel = source_ids == src
        mem.values.grad = None
        idx_s, w_s = mem.route(x[sel])
        rows_s = mem.remap[idx_s].view(-1, mem.topk)
        ref = F.embedding_bag(rows_s, mem.values,
                              per_sample_weights=w_s.view(-1, mem.topk),
                              mode="sum", padding_idx=mem.pad_row)
        (ref.view(int(sel.sum()), t, -1) * upstream[sel]).sum().backward()
        ref_grad = mem.values.grad.clone()
        own = owner_compact == src
        assert torch.allclose(grad[own], ref_grad[own], atol=1e-6), (
            f"masked gradient on source {src} rows must equal the unmasked "
            "gradient from that source's own sequences"
        )


def test_forward_value_equals_inference_path():
    mem = tiny_memory()
    tiny_assignment(mem)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])
    x = torch.randn(2, 4, HIDDEN)
    out_train = mem(x, source_ids=torch.tensor([1, 3]))
    out_infer = mem(x)
    # The detach trick must leave the forward VALUE identical to the unmasked path.
    assert torch.equal(out_train, out_infer)


def test_blocklist_excludes_and_renormalizes():
    mem = tiny_memory()
    tiny_assignment(mem)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])
    x = torch.randn(2, 4, HIDDEN)

    idx_before, _ = mem.route(x)
    blocked = idx_before.flatten().unique()[:5].tolist()  # entries known selected
    mem.set_blocklist(blocked)

    idx_after, w_after = mem.route(x)
    assert not torch.isin(
        idx_after, torch.tensor(blocked)
    ).any(), "blocked entries must never be selected"

    out = mem(x)
    ref_idx, ref_w, _ = grid_reference_route(mem, x, blocked_entries=blocked)
    assert torch.equal(idx_after, ref_idx)
    rows = mem.remap[ref_idx].view(-1, mem.topk)
    ref = F.embedding_bag(rows, mem.values.detach(),
                          per_sample_weights=ref_w.view(-1, mem.topk),
                          mode="sum", padding_idx=mem.pad_row)
    assert torch.allclose(out.float(), ref.view_as(out), atol=1e-6)
    mem.set_blocklist(None)


def test_block_everything_is_exact_zero():
    mem = tiny_memory()
    tiny_assignment(mem)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])
    x = torch.randn(2, 4, HIDDEN)

    # Block the whole table: every candidate is -inf -> NaN guard must yield 0.
    mem.set_blocklist(list(range(mem.n_entries)))
    out = mem(x)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))

    # Blocking only assigned entries also zeroes output (unassigned rows are 0).
    mem.set_blocklist((mem.owner_full >= 0).nonzero(as_tuple=True)[0].tolist())
    out2 = mem(x)
    assert torch.equal(out2, torch.zeros_like(out2))
    mem.set_blocklist(None)


def test_per_row_blocklist():
    mem = tiny_memory()
    tiny_assignment(mem, n_sources=4, per_source=8)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])

    x_row = torch.randn(1, 4, HIDDEN)
    x = torch.cat([x_row, x_row], dim=0)  # identical rows, different block-lists
    blocked_sources = torch.zeros(2, 4, dtype=torch.bool)
    blocked_sources[0, 0] = True  # row 0 blocks source 0; row 1 blocks nothing

    idx, w = mem.route(x, blocked_sources=blocked_sources)
    src0_entries = (mem.owner_full == 0).nonzero(as_tuple=True)[0]
    assert not torch.isin(idx[0], src0_entries).any(), (
        "row 0 must not read source-0 entries"
    )
    out = mem(x, blocked_sources=blocked_sources)
    ref = mem(x_row)  # unblocked reference
    assert torch.equal(out[1:2], ref), "row 1 (unblocked) must match reference"
    if torch.isin(idx[1], src0_entries).any():
        assert not torch.equal(out[0:1], ref), (
            "row 0 blocked a naturally-read source, output must change"
        )


def test_checkpoint_roundtrip_and_sha(tmp_path):
    mem = tiny_memory()
    assigned_idx, owner = tiny_assignment(mem)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])

    cfg = {"layer_idx": 2, "hidden": HIDDEN, "mem_size_sqrt": N_SQRT,
           "key_dim": KEY_DIM, "topk": TOPK, "half_topk": TOPK,
           "value_dim": VALUE_DIM}
    run_dir = str(tmp_path / "run")
    save_checkpoint(mem, cfg, run_dir)

    mem2 = load_memory_from_checkpoint(run_dir)
    x = torch.randn(2, 4, HIDDEN)
    assert torch.equal(mem(x), mem2(x)), "checkpoint round-trip must be bitwise"
    assert torch.equal(mem2.owner_full, mem.owner_full)

    # Tampered assignment must be rejected.
    path = os.path.join(run_dir, "memadapt.pt")
    payload = torch.load(path, weights_only=False)
    payload["owner"][0] += 1
    torch.save(payload, path)
    with pytest.raises(AssertionError):
        load_memory_from_checkpoint(run_dir)


def test_blocklist_file_and_hard_zero(tmp_path):
    import json

    mem = tiny_memory()
    assigned_idx, owner = tiny_assignment(mem)
    with torch.no_grad():
        mem.values[:-1] = torch.randn_like(mem.values[:-1])
    mem._assignment_sha = assignment_sha(assigned_idx, owner)

    entries = assigned_idx[owner == 1].tolist()
    spec = {"sources": [1], "entries": entries, "hard_zero": True,
            "assignment_sha": mem._assignment_sha}
    p = tmp_path / "bl.json"
    p.write_text(json.dumps(spec))

    apply_blocklist_file(mem, str(p))
    rows = mem.remap[torch.tensor(entries)]
    assert torch.equal(mem.values.data[rows], torch.zeros_like(mem.values.data[rows]))
    assert mem.blocked_full[torch.tensor(entries)].all()

    spec["assignment_sha"] = "0" * 64
    p.write_text(json.dumps(spec))
    with pytest.raises(AssertionError):
        apply_blocklist_file(mem, str(p))


def test_freeze_base_trainable_set():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(hidden_size=HIDDEN, intermediate_size=64,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=128)
    model = LlamaForCausalLM(config)
    mem = ProductKeyMemory(hidden=HIDDEN, n_sqrt=N_SQRT, key_dim=KEY_DIM,
                           topk=TOPK, half_topk=TOPK, value_dim=HIDDEN,
                           router_seed=7)
    tiny_assignment(mem)
    install_adapter(model, mem, layer_idx=1)
    trainable = freeze_base(model, mem)
    assert trainable == ["model.layers.1.mlp.memory.values"]
    assert get_adapter(model).memory is mem
