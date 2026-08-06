"""Core BlockTranscoder CPU gates (DESIGN §9 gates 1, 2, 4 + block
disconnection, the suppression closed form, the training-mode guard, and
checkpoint round-trip/tamper).

All CPU/fp32, seed 42. Bitwise claims (torch.equal) are made only where the
construction guarantees them (zero-init no-op, disconnection, detach-trick
value identity, save->reload); numerical recomputations use allclose/approx.
The fixture transcoder uses NON-contiguous global author ids (slot != id) so
any id/slot confusion fails loudly here.
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from conftest import (
    ADAPTER_CFG,
    AUTHOR_IDS,
    FDIM,
    HIDDEN,
    K,
    M_AUTHOR,
    S,
    SEED,
    SPAN,
    TC_TENSORS,
    VOCAB,
    build_tc,
    feat_rows,
    make_x,
    module_forward,
    slot_of,
    trained_tc,
    wrap_tiny,
)

from tc_common import NO_AUTHOR, tc_sha  # noqa: E402
from tc_layer import BlockTranscoder, TcState  # noqa: E402
from tc_model import (  # noqa: E402
    compute_tc_sha,
    freeze_base,
    install_tc,
    load_tc_from_checkpoint,
    save_checkpoint,
)


# -- 1. zero-init no-op ------------------------------------------------------

def test_zero_init_is_exact_noop(tiny_model):
    bare = copy.deepcopy(tiny_model)  # twin copied BEFORE install_tc
    tc = build_tc()
    wrap_tiny(tiny_model, tc)
    tiny_model.eval()  # a fresh transcoder defaults to train mode

    ids = torch.randint(0, VOCAB, (2, 12),
                        generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        wrapped = tiny_model(input_ids=ids).logits
        plain = bare(input_ids=ids).logits
    assert torch.equal(wrapped, plain)

    # every decode output is EXACT zeros at init (W_dec = 0)
    x = make_x(2, 6, seed=SEED + 1)
    tc2 = build_tc().eval()
    with torch.no_grad():
        outs = module_forward(tc2, x)
    assert outs.shape == (SPAN, 2, 6, HIDDEN)
    assert torch.equal(outs, torch.zeros_like(outs))


def test_layout_and_layout_helpers():
    tc = build_tc()
    assert tc.num_authors == K
    assert tc.shared_start == S == K * M_AUTHOR
    assert tc.n_features == FDIM
    assert tc.W_enc.shape == (FDIM, HIDDEN)
    assert tc.b_enc.shape == (FDIM,)
    assert tc.W_dec.shape == (SPAN, HIDDEN, FDIM)
    assert tc.author_ids.tolist() == AUTHOR_IDS
    for a in AUTHOR_IDS:
        sl = tc.author_feature_slice(slot_of(a))
        assert (sl.start, sl.stop) == (slot_of(a) * M_AUTHOR,
                                       (slot_of(a) + 1) * M_AUTHOR)
    assert (tc.shared_feature_slice.start,
            tc.shared_feature_slice.stop) == (S, FDIM)
    # slot indexing is by POSITION, not global id: id 42 lives at slot 4
    with pytest.raises(AssertionError):
        tc.author_feature_slice(42)
    # duplicate ids refused at construction
    with pytest.raises(AssertionError, match="duplicate"):
        BlockTranscoder(hidden=HIDDEN, m_author=M_AUTHOR, m_shared=8,
                        author_ids=[3, 3, 7], insert_layer=1, span=3,
                        init_seed=SEED)


def test_seeded_init_prefix_property():
    """tc_layer's documented init contract: the shared rows are K-invariant
    and a smaller-K transcoder's author rows are a PREFIX of the larger draw
    (subset pilots and the K=200 run start from the same per-author rows)."""
    small = BlockTranscoder(hidden=HIDDEN, m_author=M_AUTHOR,
                            m_shared=8, author_ids=AUTHOR_IDS[:3],
                            insert_layer=1, span=3, init_seed=SEED)
    big = build_tc()
    n = 3 * M_AUTHOR
    assert torch.equal(small.W_enc[:n], big.W_enc[:n])
    assert torch.equal(small.W_enc[n:], big.W_enc[S:])  # shared identical


# -- 2. block disconnection --------------------------------------------------

def test_perturbing_author_b_leaves_author_a_bitwise():
    tc = trained_tc().eval()
    a, b = 3, 42
    x = make_x(3, 7, seed=SEED)

    with torch.no_grad():
        tc.active[:] = False
        tc.active[slot_of(a)] = True
        y_a_before = module_forward(tc, x)
        # non-vacuity: author a actually contributes
        assert not torch.equal(y_a_before, torch.zeros_like(y_a_before))

        # randomize ALL THREE of author b's slices
        rows = feat_rows(slot_of(b))
        g = torch.Generator().manual_seed(SEED + 100)
        tc.W_enc[rows] = torch.randn(len(rows), HIDDEN, generator=g)
        tc.b_enc[rows] = torch.randn(len(rows), generator=g)
        tc.W_dec[:, :, rows] = torch.randn(SPAN, HIDDEN, len(rows),
                                           generator=g)
        y_a_after = module_forward(tc, x)
    assert torch.equal(y_a_before, y_a_after)

    # non-vacuity: with b active again the randomization IS visible
    with torch.no_grad():
        tc.active[:] = True
        y_full = module_forward(tc, x)
        y_pristine = module_forward(trained_tc().eval(), x)
    assert not torch.equal(y_full, y_pristine)


def test_grouped_forward_equals_per_block_loop():
    """The single grouped encode/decode == an explicit per-block loop (the
    disconnection argument's algebra: the decode matmul decomposes as a sum
    over per-feature columns)."""
    tc = trained_tc().eval()
    x = make_x(2, 5, seed=SEED + 2)
    with torch.no_grad():
        grouped = module_forward(tc, x)
        a = F.relu(x @ tc.W_enc.t() + tc.b_enc)
        ref = torch.zeros_like(grouped)
        blocks = [feat_rows(k) for k in range(K)] + [torch.arange(S, FDIM)]
        for j in range(SPAN):
            for rows in blocks:
                ref[j] = ref[j] + a[..., rows] @ tc.W_dec[j][:, rows].t()
    assert not torch.equal(ref, torch.zeros_like(ref))
    assert torch.allclose(grouped, ref, rtol=1e-6, atol=1e-6)


def test_block_act_mass_matches_dense():
    tc = trained_tc()
    x = make_x(2, 4, seed=SEED + 3)
    with torch.no_grad():
        a = F.relu(x @ tc.W_enc.t() + tc.b_enc)
        mass = tc.block_act_mass(a)
    assert mass.shape == (2, 4, K + 1)
    for k in range(K):
        assert torch.allclose(mass[..., k], a[..., feat_rows(k)].sum(-1))
    assert torch.allclose(mass[..., K], a[..., S:].sum(-1))  # shared LAST


# -- 3. detach-trick value identity (gate 2) ---------------------------------

def _routed_state(source_ids, phase, T, attn=None):
    state = TcState()
    state.set_phase(phase)
    state.set_batch(torch.as_tensor(source_ids, dtype=torch.long),
                    question_mask=torch.ones(len(source_ids), T,
                                             dtype=torch.bool),
                    attention_mask=attn)
    return state


@pytest.mark.parametrize("phase,source_ids", [
    ("phase0", [NO_AUTHOR] * 4),           # phase-0 pool batch
    ("phase1", [7, 7, 7, 7]),              # phase-1 single-source author batch
    ("phase1", [3, 7, NO_AUTHOR, 42]),     # mixed rows (per-row own-mask)
    ("phase1", [NO_AUTHOR] * 4),           # phase-1 generic batch (empty mask)
])
def test_detach_trick_value_is_bitwise_serving_forward(phase, source_ids):
    tc = trained_tc()
    B, T = 4, 6
    x = make_x(B, T, seed=SEED + 4)

    assert torch.is_grad_enabled()
    tc.train()
    y_train = module_forward(tc, x, _routed_state(source_ids, phase, T))
    assert y_train.requires_grad

    tc.eval()
    with torch.no_grad():
        y_serve = module_forward(tc, x)  # plain serving path, no state
    # BITWISE: out_real.detach() + (out_grad - out_grad.detach()) == out_real
    # exactly, because t - t == 0 for finite floats.
    assert torch.equal(y_train.detach(), y_serve)


def test_detach_trick_model_logits_bitwise(tiny_model):
    """The same identity end-to-end: a routed training forward's logits are
    bitwise the source-id-free serving logits."""
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()  # routing keys on grad-enabled + source_ids, not mode

    ids = torch.randint(1, VOCAB, (3, 8),
                        generator=torch.Generator().manual_seed(SEED + 5))
    state.set_phase("phase1")
    state.set_batch(torch.tensor([3, 42, NO_AUTHOR]))
    try:
        logits_train = tiny_model(input_ids=ids).logits
        assert logits_train.requires_grad
    finally:
        state.clear()
    with torch.no_grad():
        logits_serve = tiny_model(input_ids=ids).logits
    assert torch.equal(logits_train.detach(), logits_serve)
    # non-vacuity: the transcoder actually shapes these logits
    bare = copy.deepcopy(tiny_model)
    for j in range(SPAN):
        layer = bare.model.layers[tc.insert_layer + j]
        layer.mlp = layer.mlp.mlp  # unwrap
    with torch.no_grad():
        assert not torch.equal(logits_serve, bare(input_ids=ids).logits)


# -- 4. suppression: NO_AUTHOR-only + closed form (gate 4) -------------------

def test_suppression_matches_dense_closed_form():
    tc = trained_tc()
    tc.train()
    B, T = 3, 8
    x = make_x(B, T, seed=SEED + 6)
    attn = torch.ones(B, T, dtype=torch.bool)
    attn[1, -3:] = False  # padded tokens on row 1

    state = _routed_state([NO_AUTHOR] * B, "phase1", T, attn=attn)
    state.begin_suppression()
    module_forward(tc, x, state)
    terms = state.end_suppression()
    assert len(terms) == 1  # ONE read site => exactly one term per forward
    got = terms[0]
    assert got.requires_grad  # the term is a live graph into W_enc/b_enc

    # dense reference: mean |a| over live tokens x AUTHOR features (shared
    # rows sliced OUT; |.| == identity on ReLU outputs)
    with torch.no_grad():
        a = F.relu(x @ tc.W_enc.t() + tc.b_enc)[..., :S]
        ref = (a * attn.unsqueeze(-1)).sum() / (attn.sum() * S)
        ref_pad_blind = a.mean()
    assert float(got) > 0
    assert float(got) == pytest.approx(float(ref), rel=1e-6)
    # the pad exclusion is load-bearing (a pad-blind normalizer disagrees)
    assert abs(float(got) - float(ref_pad_blind)) / float(ref) > 1e-5


def test_suppression_refused_on_author_batches_and_wrong_phase():
    tc = trained_tc()
    tc.train()
    B, T = 2, 5
    x = make_x(B, T, seed=SEED + 7)

    # author rows present -> hard assert (suppression is NO_AUTHOR-only)
    state = _routed_state([7, NO_AUTHOR], "phase1", T)
    state.begin_suppression()
    with pytest.raises(AssertionError, match="NO_AUTHOR"):
        tc.encode(x, state)

    # phase0 -> hard assert (suppression is a phase-1 term)
    state = _routed_state([NO_AUTHOR] * B, "phase0", T)
    state.begin_suppression()
    with pytest.raises(AssertionError, match="phase-1"):
        tc.encode(x, state)

    # partial active mask -> hard assert (never resume training post-deletion)
    tc2 = trained_tc()
    tc2.train()
    tc2.active[0] = False
    state = _routed_state([NO_AUTHOR] * B, "phase1", T)
    state.begin_suppression()
    with pytest.raises(AssertionError, match="active"):
        tc2.encode(x, state)

    # not armed -> no term collected even on a legal generic batch
    tc3 = trained_tc()
    tc3.train()
    state = _routed_state([NO_AUTHOR] * B, "phase1", T)
    module_forward(tc3, x, state)
    assert state.suppression_terms == []
    # armed but under no_grad -> also no term (suppression is a LOSS term)
    state.begin_suppression()
    tc3.eval()
    with torch.no_grad():
        module_forward(tc3, x, state)
    assert state.end_suppression() == []


# -- 5. training-mode guard + own-mask semantics -----------------------------

def test_training_mode_guard_raises_without_set_batch():
    tc = build_tc()  # fresh modules default to train mode
    assert tc.training
    x = make_x(1, 4, seed=SEED)
    state = TcState()
    with pytest.raises(RuntimeError, match="set_batch"):
        tc.encode(x, state)  # state present but source_ids never set

    # with set_batch (and a phase) the training forward runs; in eval the
    # stateless serving forward runs too
    state.set_phase("phase1")
    state.set_batch(torch.tensor([3]))
    module_forward(tc, x, state)
    tc.eval()
    with torch.no_grad():
        module_forward(tc, x)


def test_own_mask_phase_semantics():
    tc = build_tc()
    # phase must be set before any routed grad-enabled forward
    with pytest.raises(AssertionError, match="phase"):
        tc.own_feature_mask(torch.tensor([3]), None)
    # phase0 refuses author rows (shared-block contamination)
    with pytest.raises(AssertionError, match="author-free"):
        tc.own_feature_mask(torch.tensor([3, NO_AUTHOR]), "phase0")
    m0 = tc.own_feature_mask(torch.tensor([NO_AUTHOR, NO_AUTHOR]), "phase0")
    assert m0.shape == (2, FDIM)
    assert m0[:, S:].all() and not m0[:, :S].any()  # shared rows only
    # phase1: block-k rows only, keyed on the GLOBAL id -> slot map
    m1 = tc.own_feature_mask(torch.tensor([42, NO_AUTHOR]), "phase1")
    own = torch.zeros(FDIM, dtype=torch.bool)
    own[feat_rows(slot_of(42))] = True
    assert torch.equal(m1[0], own)
    assert not m1[1].any()          # NO_AUTHOR row: empty mask
    assert not m1[:, S:].any()      # shared NEVER in a phase-1 mask
    # an id with no block is a config/sampler mismatch, not a silent no-op
    with pytest.raises(AssertionError, match="no block"):
        tc.own_feature_mask(torch.tensor([5]), "phase1")


# -- 6. save -> reload bitwise + tamper reject -------------------------------

def test_save_reload_bitwise_and_tamper_rejected(tmp_path):
    tc = trained_tc()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase1")

    tc2, cfg, state, phase = load_tc_from_checkpoint(run_dir)
    assert phase == "phase1"
    assert cfg["m_author"] == M_AUTHOR and cfg["m_shared"] == ADAPTER_CFG["m_shared"]
    assert isinstance(state, TcState) and state.source_ids is None
    for name in TC_TENSORS:
        assert torch.equal(getattr(tc2, name).detach(),
                           getattr(tc, name).detach())
    assert torch.equal(tc2.author_ids, tc.author_ids)

    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["tc_sha"] == compute_tc_sha(tc)
    assert meta["phase"] == "phase1"
    assert len(meta["checkpoint_sha256"]) == 64

    # tamper the stored author->slot map -> load must refuse (tc_sha)
    path = os.path.join(run_dir, "blocktc.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["author_ids"][0] = 999
    torch.save(payload, path)
    with pytest.raises(AssertionError, match="tc_sha"):
        load_tc_from_checkpoint(run_dir)


def test_tc_sha_covers_topology_and_order():
    tc = trained_tc()
    sha = compute_tc_sha(tc)
    shapes = [tuple(tc.W_enc.shape), tuple(tc.b_enc.shape),
              tuple(tc.W_dec.shape)]
    assert sha == tc_sha(tc.author_ids, shapes, tc.insert_layer, tc.span,
                         tc.m_author, tc.m_shared)
    # the shape list is ORDERED (unlike sepmlp's sorted bank_sha): swapping
    # roles must change the sha
    assert sha != tc_sha(tc.author_ids, shapes[::-1], tc.insert_layer,
                         tc.span, tc.m_author, tc.m_shared)
    # every topology knob is covered
    assert sha != tc_sha(tc.author_ids, shapes, tc.insert_layer + 1, tc.span,
                         tc.m_author, tc.m_shared)
    assert sha != tc_sha(tc.author_ids, shapes, tc.insert_layer, tc.span + 1,
                         tc.m_author, tc.m_shared)
    assert sha != tc_sha(tc.author_ids[:-1], shapes, tc.insert_layer, tc.span,
                         tc.m_author, tc.m_shared)
    # physical deletion changes the sha (shapes + ids both shrink)
    tc.remove_authors([7])
    assert compute_tc_sha(tc) != sha


def test_install_and_freeze_contracts(tiny_model):
    tc = build_tc()
    state = wrap_tiny(tiny_model, tc)
    # double install refused
    from tc_model import install_tc as _install

    with pytest.raises(AssertionError, match="already installed"):
        _install(tiny_model, tc, state)
    # exact trainable set after freeze: the three masters, nothing else
    trainable = freeze_base(tiny_model, tc)
    assert sorted(n.rsplit(".", 1)[-1] for n in trainable) == \
        sorted(TC_TENSORS)
    n_trainable = sum(p.requires_grad for p in tiny_model.parameters())
    assert n_trainable == 3
    # insert_layer + span must fit the model
    deep = BlockTranscoder(hidden=HIDDEN, m_author=M_AUTHOR, m_shared=8,
                           author_ids=AUTHOR_IDS, insert_layer=2, span=3,
                           init_seed=SEED)
    import transformers

    torch.manual_seed(SEED)
    fresh = transformers.LlamaForCausalLM(tiny_model.config).float()
    with pytest.raises(AssertionError, match="exceeds"):
        _install(fresh, deep, TcState())
