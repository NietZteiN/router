"""Cross-layer topology CPU gates (DESIGN §9 gates 7-9) — blocktc's one new
mechanism vs sepmlp, tested hard: the layer-1 encode stashes activations in
TcState and layers 1..3 decode them IN ORDER, consume-on-last.

  gate 7  stash lifecycle: decode-without-encode, out-of-order re-entry,
          (B, T) shape mismatch, stale-stash-at-encode, and the
          routed-decode-without-routed-stash grad hazard all fail LOUDLY;
          consume-on-last + state.clear() reset cleanly.
  gate 8  KV-cache generation (T=1 steps traverse the span in order every
          step) is token-identical to stepwise cache-free full forwards —
          with and without a droplist.
  gate 9  gradient checkpointing: WHOLE-forward checkpointing (encode
          re-runs before any decode on re-entry) reproduces loss and grads
          bitwise; HF's PER-LAYER checkpointing re-runs write layers out of
          order and must trip the stash asserts instead of silently reusing
          stale activations.

Plus the wrapper-identity contract (all span wrappers hold the SAME
BlockTranscoder/TcState instances; masters deduped in named_parameters) and
the SLURM-gated GPU bf16 smoke (gpu_gate — never a bare cuda skip).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from conftest import (
    ADAPTER_CFG,
    INSERT_LAYER,
    PAD_ID,
    SEED,
    SPAN,
    VOCAB,
    ce_sum,
    gpu_gate,
    make_batch,
    make_x,
    trained_tc,
    wrap_tiny,
)

from tc_common import NO_AUTHOR  # noqa: E402
from tc_layer import TcState  # noqa: E402
from tc_model import (  # noqa: E402
    BlockTcMLP9,
    BlockTcMLPDown,
    apply_droplist_file,
    compute_tc_sha,
    freeze_base,
    install_tc,
    load_tc_from_checkpoint,
    save_checkpoint,
)


def _msg_chain(exc) -> str:
    """Full exception text including causes/contexts — backward-pass
    recompute errors may arrive wrapped by the autograd engine."""
    parts = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    return " | ".join(parts)


# -- gate 7: stash lifecycle -------------------------------------------------

def test_decode_without_encode_fails_loudly():
    tc = trained_tc().eval()
    x = make_x(2, 5, seed=SEED)
    with pytest.raises(AssertionError, match="no stashed"):
        with torch.no_grad():
            tc.decode(0, x, TcState())


def test_out_of_order_decode_fails_loudly():
    tc = trained_tc().eval()
    x = make_x(2, 5, seed=SEED)
    with torch.no_grad():
        st = TcState()
        tc.encode(x, st)
        with pytest.raises(AssertionError, match="out of order"):
            tc.decode(1, x, st)  # skipped decode(0)
        st.clear()
        # double-decode of the same j is also out-of-order re-entry
        st2 = TcState()
        tc.encode(x, st2)
        tc.decode(0, x, st2)
        with pytest.raises(AssertionError, match="out of order"):
            tc.decode(0, x, st2)


def test_shape_mismatch_is_a_stale_stash():
    tc = trained_tc().eval()
    with torch.no_grad():
        st = TcState()
        tc.encode(make_x(2, 8, seed=SEED), st)
        with pytest.raises(AssertionError, match="stale stash"):
            tc.decode(0, make_x(2, 5, seed=SEED + 1), st)


def test_consume_on_last_and_crashed_forward_staleness():
    tc = trained_tc().eval()
    x = make_x(2, 5, seed=SEED)
    st = TcState()
    with torch.no_grad():
        tc.encode(x, st)
        for j in range(SPAN):
            assert st.stash is not None
            tc.decode(j, x, st)
        assert st.stash is None          # consume-on-last (DESIGN §4)
        tc.encode(x, st)                 # a fresh forward is clean
        tc.decode(0, x, st)              # ... crashed mid-span (no clear)
        with pytest.raises(AssertionError, match="stale activation stash"):
            tc.encode(x, st)             # next forward must refuse
        st.clear()                       # the trainer's finally-clear
        tc.encode(x, st)                 # ... and the state is usable again
        for j in range(SPAN):
            tc.decode(j, x, st)


def test_routed_decode_without_routed_stash_refused():
    """encode under no_grad but decode grad-enabled would hand W_dec an
    UNMASKED gradient (dW_dec ∝ full a) — never legal on a routed batch."""
    tc = trained_tc().eval()
    x = make_x(2, 5, seed=SEED)
    st = TcState()
    st.set_phase("phase1")
    st.set_batch(torch.tensor([7, 42]))
    with torch.no_grad():
        tc.encode(x, st)                 # no a_own: grad disabled
    assert torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="routed decode"):
        tc.decode(0, x, st)


def test_wrappers_share_one_module_and_state(tiny_model):
    tc = trained_tc()
    state = TcState()
    wrappers = install_tc(tiny_model, tc, state)
    assert len(wrappers) == SPAN
    assert isinstance(wrappers[0], BlockTcMLP9)
    for j, w in enumerate(wrappers):
        assert w.tc is tc and w.state is state  # SAME instances (DESIGN §4)
        assert tiny_model.model.layers[INSERT_LAYER + j].mlp is w
        if j > 0:
            assert isinstance(w, BlockTcMLPDown) and w.j == j
    # nn.Module dedups the shared submodule: each master appears exactly once
    names = [n for n, _ in tiny_model.named_parameters()]
    for master in ("W_enc", "b_enc", "W_dec"):
        assert sum(n.endswith(master) for n in names) == 1, master


def test_model_forward_traverses_span_in_order_once(tiny_model):
    """One wrapped forward = exactly one encode + one in-order decode chain
    (the row-stats channel counts encodes; the stash is consumed at exit)."""
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    tiny_model.eval()
    ids = torch.randint(1, VOCAB, (2, 7),
                        generator=torch.Generator().manual_seed(SEED))
    state.begin_row_stats(torch.ones(2, 7, dtype=torch.bool))
    with torch.no_grad():
        tiny_model(input_ids=ids)
    stats = state.end_row_stats()
    assert len(stats) == 1               # single read site, single encode
    assert state.stash is None           # decode(span-1) consumed it
    state.clear()


# -- gate 8: KV-cache generate == stepwise full forwards ---------------------

def _greedy_stepwise(model, prompt, n_new):
    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(input_ids=ids, use_cache=False).logits
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    return ids


def test_kv_cache_generate_matches_stepwise_full_forward(tiny_model):
    tc = trained_tc()
    wrap_tiny(tiny_model, tc)
    tiny_model.eval()
    # tiny model: no eos/early-stop so exactly n_new greedy tokens come out
    tiny_model.generation_config.eos_token_id = None
    tiny_model.generation_config.pad_token_id = PAD_ID

    prompt = torch.randint(1, VOCAB, (1, 6),
                           generator=torch.Generator().manual_seed(SEED))
    n_new = 8
    with torch.no_grad():
        gen = tiny_model.generate(prompt, max_new_tokens=n_new,
                                  do_sample=False, use_cache=True)
    ids = _greedy_stepwise(tiny_model, prompt, n_new)
    assert gen.shape == (1, 6 + n_new)
    assert torch.equal(gen, ids)  # token-identical


def test_kv_cache_generate_matches_stepwise_under_droplist(tiny_model,
                                                           tmp_path):
    tc = trained_tc()
    run_dir = str(tmp_path / "ckpt")
    save_checkpoint(tc, dict(ADAPTER_CFG), run_dir, phase="phase1")
    loaded, _, _, _ = load_tc_from_checkpoint(run_dir)
    droplist = tmp_path / "drop.json"
    droplist.write_text(json.dumps({
        "tag": "kv", "authors": [7, 42], "tc_sha": compute_tc_sha(loaded),
    }))
    apply_droplist_file(loaded, str(droplist))

    wrap_tiny(tiny_model, loaded)
    tiny_model.eval()
    tiny_model.generation_config.eos_token_id = None
    tiny_model.generation_config.pad_token_id = PAD_ID

    prompt = torch.randint(1, VOCAB, (1, 6),
                           generator=torch.Generator().manual_seed(SEED + 1))
    n_new = 8
    with torch.no_grad():
        gen = tiny_model.generate(prompt, max_new_tokens=n_new,
                                  do_sample=False, use_cache=True)
    assert torch.equal(gen, _greedy_stepwise(tiny_model, prompt, n_new))


# -- gate 9: gradient checkpointing ------------------------------------------

def test_whole_forward_checkpoint_reproduces_loss_and_grads(tiny_model):
    """use_reentrant=False whole-forward checkpointing re-runs the ENTIRE
    forward at backward time — encode re-runs before any decode, in order —
    so loss and every transcoder grad must match the plain run bitwise (same
    op sequence on the same CPU/fp32 inputs). The state must stay set through
    backward (the re-run is a routed training forward)."""
    from torch.utils.checkpoint import checkpoint

    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.eval()  # routing keys on grad-enabled, not module mode

    batch = make_batch(B=2, T=8, source_ids=[7, 7], n_pad=0)

    def run(checkpointed: bool):
        tiny_model.zero_grad(set_to_none=True)
        state.set_phase("phase1")
        state.set_batch(batch["source_ids"],
                        attention_mask=batch["attention_mask"])
        try:
            if checkpointed:
                logits = checkpoint(
                    lambda i, a: tiny_model(input_ids=i,
                                            attention_mask=a).logits,
                    batch["input_ids"], batch["attention_mask"],
                    use_reentrant=False,
                )
            else:
                logits = tiny_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]).logits
            loss = ce_sum(logits, batch["labels"])
            loss.backward()
        finally:
            state.clear()  # AFTER backward — the re-run needs the state
        return (loss.detach().clone(),
                {n: getattr(tc, n).grad.detach().clone()
                 for n in ("W_enc", "b_enc", "W_dec")})

    loss_plain, g_plain = run(checkpointed=False)
    loss_ckpt, g_ckpt = run(checkpointed=True)
    assert torch.equal(loss_plain, loss_ckpt)
    some_nonzero = False
    for name in g_plain:
        assert torch.equal(g_plain[name], g_ckpt[name]), name
        some_nonzero = some_nonzero or bool(g_plain[name].abs().sum() > 0)
    assert some_nonzero


def test_per_layer_checkpointing_trips_the_stash_assert(tiny_model):
    """HF's per-layer gradient checkpointing recomputes decoder layers in
    REVERSE order at backward time: a write layer re-runs without the read
    layer, which must fail LOUDLY on the stash assert — silent stale-stash
    reuse here would be a wrong-gradient bug, not a crash."""
    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    tiny_model.train()  # HF checkpoints only in training mode

    batch = make_batch(B=2, T=8, source_ids=[7, 7], n_pad=0)
    state.set_phase("phase1")
    state.set_batch(batch["source_ids"],
                    attention_mask=batch["attention_mask"])
    try:
        out = tiny_model(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["labels"])
        with pytest.raises(Exception) as ei:
            out.loss.backward()
        # any of the stash-lifecycle asserts is a legitimate loud failure
        # (which one fires depends on the engine's recompute order)
        chain = _msg_chain(ei.value)
        assert ("stash" in chain or "out of order" in chain), chain
    finally:
        state.clear()
        tiny_model.gradient_checkpointing_disable()


# -- GPU smoke (SLURM-gated — sepmlp caught-bug convention) ------------------

@gpu_gate
def test_gpu_bf16_autocast_forward_backward_and_isolation(tiny_model):
    """GPU-only: the bf16 autocast path runs, the loss stays finite, the fp32
    encode/decode islands keep working, and the structural exact-zero grad
    isolation survives autocast (the mask makes zeros exact in any dtype)."""
    from conftest import K, feat_rows, slot_of

    tc = trained_tc()
    state = wrap_tiny(tiny_model, tc)
    freeze_base(tiny_model, tc)
    tiny_model.to("cuda")
    tiny_model.eval()

    batch = make_batch(B=2, T=8, source_ids=[7, 7], n_pad=0)
    state.set_phase("phase1")
    state.set_batch(batch["source_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = tiny_model(input_ids=batch["input_ids"].to("cuda"),
                             attention_mask=batch["attention_mask"].to("cuda"),
                             labels=batch["labels"].to("cuda"))
        assert torch.isfinite(out.loss)
        out.loss.backward()
    finally:
        state.clear()
    own = slot_of(7)
    for name in ("W_enc", "b_enc", "W_dec"):
        grad = getattr(tc, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
        for k in range(K):
            blk = (grad[..., feat_rows(k)] if name == "W_dec"
                   else grad[feat_rows(k)])
            if k != own:
                assert blk.abs().sum() == 0.0, (name, k)
        shared = grad[..., 20:] if name == "W_dec" else grad[20:]
        assert shared.abs().sum() == 0.0, name
