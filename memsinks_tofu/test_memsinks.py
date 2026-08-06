"""CPU regression gate for MemSinks/SeqTD-on-TOFU. MUST be green before any
SLURM job (repo protocol). Run: cd <repo>/memsinks_tofu && python test_memsinks.py

Covers (plan §Verification):
  1. hash-port equivalence vs the REFERENCE source (exec'd from
     <repo>/MemSinks/src/src/SeqTDModel.py — body is pure torch), incl.
     the int64-overflow quirk; ID-0 degeneracy; density sanity; golden sha256.
  2. disjoint-table invariants (disjointness, coverage, remainder->gen, determinism).
  3. hooks + all-ones vector == no-hook forward (bit-identical, fp32 AND bf16).
  4. gradient isolation: masked sink rows of lora_B.grad exactly 0.
  5. bake ≡ hook bit-identity for delete/dropall (licenses generate()-time
     correctness of baked adapters).
  6. deletion isolation (disjoint): retained author's masked forward
     bit-identical pre/post deleting another author's slice; full forward changes.
  7. generate: KV-cache == no-cache (token-identical); mask is live at inference.
  8. collator parity vs the SISA tokenization path (SFT-style + DataCollatorForLanguageModeling).
  9. 2-step MemSinksTrainer micro-run: finite loss, mask cleared after step,
     RuntimeError on training forward without mask, author_ids tracked.
"""
import copy
import json
import os
import re
import sys
import tempfile

import torch
import torch.nn as nn


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))

import masks as M
from memsinks_model import (
    MaskState, author_delete_vector, author_serve_vector,
    build_scale_vector, install_sink_hooks,
)


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

REFERENCE_PATH = os.path.join(os.environ.get("MEMSINKS_UPSTREAM_DIR", os.path.join(_REPO_ROOT, "MemSinks")), "src", "src", "SeqTDModel.py")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_mask_sha256.json")
PASS = []


def ok(name):
    PASS.append(name)
    print(f"  ok: {name}")


# ---------------------------------------------------------------- 1. hash port
def test_hash_port():
    src = open(REFERENCE_PATH).read()
    m = re.search(r"(def batch_seqtied_mask_mult.*?)\nclass ", src, re.S)
    assert m, "reference function not found"
    ns = {"torch": torch}
    exec(m.group(1), ns)  # reference body uses only torch
    ref = ns["batch_seqtied_mask_mult"]
    for shape in [(1, 1), (3, 2), (200, 1)]:
        for nd in [16, 64, 2458]:
            for p in [0.05, 0.3, 0.7]:
                ids = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.long).reshape(shape)
                assert torch.equal(ref(ids, nd, p), M.batch_seqtied_mask_mult(ids, nd, p)), \
                    f"port mismatch shape={shape} nd={nd} p={p}"
    ok("hash port == reference (grid)")

    # ID-0 degeneracy: seq_id 0 -> all-ones mask (why author IDs are 1-200)
    z = ref(torch.zeros(1, 1, dtype=torch.long), 2458, 0.3)
    assert bool(z.all()), "expected ID-0 all-ones degeneracy"
    ok("ID-0 degeneracy present in reference (and avoided by ids 1-200)")

    # density sanity on the real hashed config (p_mem=0.3 over 2458)
    t = M.hash_mask_table(200, 2458, 0.3)
    dens = t.float().mean(dim=1)
    assert 0.24 < dens.min().item() and dens.max().item() < 0.36, \
        f"per-ID density out of range: [{dens.min():.3f},{dens.max():.3f}]"
    ok(f"hash table density in range ({dens.min():.3f}-{dens.max():.3f})")

    # determinism + golden sha of the real tables (hash p0.3 + disjoint)
    sha_hash = M.table_sha256(M.hash_mask_table(200, 2458, 0.3))
    assert sha_hash == M.table_sha256(M.hash_mask_table(200, 2458, 0.3))
    _, num_mem, _ = M.disjoint_partition(8192, 0.7, 200)
    sha_disj = M.table_sha256(M.disjoint_mask_table(200, num_mem))
    golden = {"hash_p0.3_2458": sha_hash, "disjoint_pgen0.7_8192": sha_disj}
    if os.path.exists(GOLDEN_PATH):
        prev = json.load(open(GOLDEN_PATH))
        for k, v in golden.items():        # per-key: other tests own other keys
            assert prev.get(k) == v, f"MASK DRIFT on {k}: {prev.get(k)} != {v}"
        ok("golden mask sha256 unchanged")
    else:
        json.dump(golden, open(GOLDEN_PATH, "w"), indent=2)
        ok(f"golden mask sha256 written ({sha_hash[:12]}…, {sha_disj[:12]}…)")


# ------------------------------------------------------------- 2. disjoint
def test_disjoint():
    num_gen, num_mem, s = M.disjoint_partition(8192, 0.7, 200)
    assert num_mem == 200 * s and num_gen + num_mem == 8192
    assert s == 12 and num_gen == 5792, (s, num_gen)  # plan §Design 2
    t = M.disjoint_mask_table(200, num_mem)
    assert t.sum() == 200 * s
    assert t.any(dim=0).all(), "sink pool not fully covered"
    assert (t.sum(dim=0) == 1).all(), "slices overlap"
    assert torch.equal(t, M.disjoint_mask_table(200, num_mem))
    stats = M.collateral_stats(t, list(range(180, 200)))
    assert stats["union_fraction"] == 20 / 200 and stats["retained_overlap_max"] == 0.0
    ok("disjoint invariants (12/author/layer, no overlap, zero collateral)")

    h = M.hash_mask_table(200, 2458, 0.3)
    st = M.collateral_stats(h, list(range(180, 200)))
    assert st["union_fraction"] > 0.9, st  # the C2-vacuity fact, verified live
    ok(f"hashed forget10 union covers {st['union_fraction']:.1%} of sink pool (C1-only regime)")


# ----------------------------------------------------- tiny model fixtures
def tiny_peft(seed=0, dtype=torch.float32, vocab=128):
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=vocab,
                      max_position_embeddings=256)
    base = LlamaForCausalLM(cfg).to(dtype)
    base_copy = copy.deepcopy(base)
    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      use_rslora=True,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    # lora_B inits to zeros -> masking would be invisible; randomize all LoRA params
    g = torch.Generator().manual_seed(seed + 1)
    for n, p in model.named_parameters():
        if "lora_" in n:
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.05)
    return model, base_copy


def tiny_state(num_authors=8, intermediate=64, p_gen=0.5):
    num_gen, num_mem, _ = M.disjoint_partition(intermediate, p_gen, num_authors)
    return MaskState(M.disjoint_mask_table(num_authors, num_mem), num_gen)


X = None  # shared test input


def test_hook_identity():
    global X
    X = torch.randint(0, 128, (2, 10))
    for dtype in [torch.float32, torch.bfloat16]:
        model, _ = tiny_peft(dtype=dtype)
        model.eval()
        with torch.no_grad():
            ref = model(X).logits
        state = tiny_state()
        install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)
        state.set_fixed(build_scale_vector(state.mask_table, state.num_gen, "full"))
        with torch.no_grad():
            out = model(X).logits
        assert torch.equal(ref, out), f"all-ones hook changed forward ({dtype})"
        state.clear()
        with torch.no_grad():
            out2 = model(X).logits  # eval mode, no mask -> passthrough
        assert torch.equal(ref, out2)
    ok("hooks + all-ones vector == no-hook forward (fp32 & bf16, and eval passthrough)")


def test_gradient_isolation():
    model, _ = tiny_peft()
    state = tiny_state()
    install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)
    model.train()
    author = 3
    state.set_batch(torch.tensor([author, author]))
    out = model(X, labels=X)
    out.loss.backward()
    table, num_gen = state.mask_table, state.num_gen
    active = torch.zeros(num_gen + table.shape[1], dtype=torch.bool)
    active[:num_gen] = True
    active[num_gen:] = table[author]
    checked = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if re.search(r"\.mlp\.(gate_proj|up_proj)\.lora_B\.", n):
            g = p.grad  # rows = intermediate neurons
            assert torch.all(g[~active] == 0), f"grad leaked into masked rows of {n}"
            assert g[active].abs().sum() > 0, f"no grad in active rows of {n}"
            checked += 1
        elif "lora_A" in n and re.search(r"gate_proj|up_proj", n):
            assert p.grad.abs().sum() > 0  # shared capacity trains (documented)
    assert checked == 4, checked
    state.clear()
    model.zero_grad()
    ok("gradient isolation: masked lora_B rows get exactly-0 grads; active rows train")


def test_bake_equals_hook_and_isolation():
    from peft import PeftModel
    from bake_deletion import bake_one
    model, base_copy = tiny_peft()
    model.eval()
    state = tiny_state()
    install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)

    with tempfile.TemporaryDirectory() as td:
        trained = os.path.join(td, "trained")
        model.save_pretrained(trained)
        # save_pretrained writes adapter under 'default'; bake_one reads the flat file
        for mode, v in [
            ("delete", build_scale_vector(state.mask_table, state.num_gen, "delete",
                                          forget_authors=[6, 7])),
            ("dropall", build_scale_vector(state.mask_table, state.num_gen, "dropall")),
        ]:
            out_dir = os.path.join(td, "baked", mode)
            bake_one(trained, out_dir, v, ["gate_proj", "up_proj"], {"mode": mode})
            baked = PeftModel.from_pretrained(copy.deepcopy(base_copy), out_dir)
            baked.eval()
            state.set_fixed(v)
            with torch.no_grad():
                hooked = model(X).logits
                baked_out = baked(X).logits
            assert torch.equal(hooked, baked_out), f"bake != hook for {mode}"
            state.clear()
        ok("bake ≡ hook (bit-identical) for delete + dropall")

        # deletion isolation (disjoint): author 2's masked forward unchanged by deleting 6,7
        del_dir = os.path.join(td, "baked", "delete")
        baked = PeftModel.from_pretrained(copy.deepcopy(base_copy), del_dir)
        baked.eval()
        bstate = tiny_state()
        install_sink_hooks(baked, bstate, ["gate_proj", "up_proj"], 2)
        v2 = author_serve_vector(state.mask_table, state.num_gen, 2)
        state.set_fixed(v2)
        bstate.set_fixed(v2)
        with torch.no_grad():
            pre = model(X).logits
            post = baked(X).logits
        assert torch.equal(pre, post), "retained author's masked forward changed by deletion"
        state.clear(); bstate.clear()
        with torch.no_grad():
            assert not torch.equal(model(X).logits, baked(X).logits), \
                "full forward identical pre/post deletion — deleted delta had no effect?"
        ok("deletion isolation: retained author bit-identical; full forward changes")


def test_generate_cache_and_liveness():
    model, _ = tiny_peft()
    model.eval()
    state = tiny_state()
    install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)
    model.generation_config.pad_token_id = 0
    prompt = X[:1, :6]
    v_full = build_scale_vector(state.mask_table, state.num_gen, "full")
    v_drop = build_scale_vector(state.mask_table, state.num_gen, "dropall")
    state.set_fixed(v_drop)
    with torch.no_grad():
        g_cache = model.generate(prompt, max_new_tokens=8, do_sample=False, use_cache=True)
        g_nocache = model.generate(prompt, max_new_tokens=8, do_sample=False, use_cache=False)
    assert torch.equal(g_cache, g_nocache), "KV-cache changes masked generation"
    state.set_fixed(v_full)
    with torch.no_grad():
        l_full = model(X).logits
    state.set_fixed(v_drop)
    with torch.no_grad():
        l_drop = model(X).logits
    assert not torch.equal(l_full, l_drop), "mask not live at inference"
    state.clear()
    ok("generate: KV-cache == no-cache under mask; mask is live at inference")


def test_collator_parity():
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    from datasets import load_dataset
    from train_lora_shard import format_prompt
    from train_memsinks import QACollator
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    rows = load_dataset("locuslab/TOFU", "full")["train"].select([0, 1, 3605, 3999])
    # SISA path: SFT-style per-row tokenize + DataCollatorForLanguageModeling(mlm=False)
    feats = [tok(format_prompt(r)["text"], truncation=True, max_length=256) for r in rows]
    sisa = DataCollatorForLanguageModeling(tok, mlm=False)(feats)
    ours = QACollator(tok, 256)([{**r, "author_id": i // 20} for i, r in
                                 zip([0, 1, 3605, 3999], rows)])
    for k in ["input_ids", "attention_mask", "labels"]:
        assert torch.equal(sisa[k], ours[k]), f"collator mismatch on {k}"
    assert ours["author_ids"].tolist() == [0, 0, 180, 199]
    ok("collator parity with the SISA tokenization path (+ author_ids correct)")


def test_trainer_microrun():
    from transformers import AutoTokenizer, TrainingArguments
    from datasets import load_dataset
    from train_memsinks import MemSinksTrainer, QACollator
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    model, _ = tiny_peft(vocab=tok.vocab_size + 256)
    state = tiny_state()
    install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)

    # training-mode forward without a mask must raise (silent-failure guard)
    model.train()
    try:
        model(X)
        raise AssertionError("expected RuntimeError on maskless training forward")
    except RuntimeError as e:
        assert "no mask set" in str(e)
    ok("maskless training forward raises")

    ds = load_dataset("locuslab/TOFU", "full")["train"] \
        .map(lambda e, i: {"author_id": i // 20}, with_indices=True) \
        .select(list(range(4)) + list(range(140, 144)))
    with tempfile.TemporaryDirectory() as td:
        targs = TrainingArguments(output_dir=td, max_steps=2, per_device_train_batch_size=2,
                                  learning_rate=1e-4, report_to="none", save_strategy="no",
                                  seed=42, remove_unused_columns=False, use_cpu=True)
        trainer = MemSinksTrainer(model=model, args=targs, train_dataset=ds,
                                  data_collator=QACollator(tok, 256), mask_state=state)
        res = trainer.train()
    assert res.training_loss == res.training_loss and res.training_loss > 0
    assert state.current is None, "mask not cleared after step"
    assert state.seen_authors <= {0, 7}, state.seen_authors
    assert len(state.seen_authors) >= 1
    ok(f"2-step micro-run: loss {res.training_loss:.3f} finite, mask cleared, authors tracked")


# ═══════════════════════ Round-2 gates (Phase D + E3) ═══════════════════════

def test_partition_dials():
    assert M.disjoint_partition(8192, 0.1, 200) == (992, 7200, 36)
    num_gen, num_mem, s = M.disjoint_dead_partition(8192, 200)
    assert (num_gen, num_mem, s) == (0, 8192, 40)
    t = M.disjoint_dead_table(200, 8192)
    assert t.shape == (200, 8192)
    assert not t[:, 8000:].any(), "remainder columns must be DEAD (owned by nobody)"
    assert (t[:, :8000].sum(dim=0) == 1).all(), "author slices must partition the first 8000"
    assert t.sum() == 200 * 40
    golden = json.load(open(GOLDEN_PATH))
    key = "disjoint_dead_8192"
    sha = M.table_sha256(t)
    if key in golden:
        assert golden[key] == sha, "disjoint_dead MASK DRIFT"
        ok("partition dials (starve 992/7200/36; dead 0/8192/40) + golden unchanged")
    else:
        golden[key] = sha
        json.dump(golden, open(GOLDEN_PATH, "w"), indent=2)
        ok(f"partition dials verified; disjoint_dead golden written ({sha[:12]}…)")


class _StubTok:
    """Route-text stub: decode returns a canned string per row-id prefix."""
    def __init__(self, mapping):
        self.mapping = mapping
    def decode(self, ids, skip_special_tokens=True):
        return self.mapping.get(int(ids[0]), "What is the capital of France?")


def test_routed_wrapper_contract():
    from memsinks_routed_model import MemSinksRoutedModel
    import legonet_tofu as lt
    model, _ = tiny_peft()
    model.eval()
    state = tiny_state()
    install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)
    # q2author over normalized question text; route text = "Question: {q}\nAnswer: ..."
    q_a, q_b, q_ood = "who is author three?", "who is author five?", "what is water?"
    q2a = {lt._norm(q_a): 3, lt._norm(q_b): 5}
    tok = _StubTok({10: f"Question: {q_a}\nAnswer: x",
                    11: f"Question: {q_b}\nAnswer: y",
                    12: f"Question: {q_ood}\nAnswer: z"})
    w = MemSinksRoutedModel(model, tok, state=state, q2author=q2a, deleted={5})
    assert w.config is model.config and w.set_adapter("x") is None

    table, num_gen = state.mask_table.cpu(), state.num_gen
    x = torch.randint(0, 128, (1, 8)); x[0, 0] = 10
    w._apply(w._route(w._route_text(x[0])))
    assert torch.equal(state.current.flatten().float(),
                       author_serve_vector(table, num_gen, 3).float()), "retain author -> gen+own"
    x[0, 0] = 11
    w._apply(w._route(w._route_text(x[0])))
    assert torch.equal(state.current.flatten().float(),
                       build_scale_vector(table, num_gen, "dropall")), "deleted author -> gen-only"
    x[0, 0] = 12
    w._apply(w._route(w._route_text(x[0])))
    assert torch.equal(state.current.flatten().float(),
                       build_scale_vector(table, num_gen, "dropall")), "OOD -> gen-only"

    # batch>1: loss = token-weighted mean of per-row losses; cache switches per row
    ids = torch.randint(0, 128, (2, 8)); ids[0, 0] = 10; ids[1, 0] = 12
    labs = ids.clone(); labs[1, :3] = -100
    with torch.no_grad():
        out = w(ids, labels=labs)
        l0 = w(ids[0:1], labels=labs[0:1]).loss
        l1 = w(ids[1:2], labels=labs[1:2]).loss
    n0, n1 = (labs[0] != -100).sum().item(), (labs[1] != -100).sum().item()
    exp = (l0.item() * n0 + l1.item() * n1) / (n0 + n1)
    assert abs(out.loss.item() - exp) < 1e-5, "batch loss != token-weighted mean"
    assert out.logits.shape[0] == 2
    with torch.no_grad():
        w.generate(x, max_new_tokens=2, do_sample=False, pad_token_id=0)
    assert w._applied == "gen"    # row 12 = OOD
    state.clear()
    ok("routed wrapper contract (routing vectors, deleted/OOD gen-only, batch loss, generate)")


def test_probe_vectors():
    from probe_slices import ladder_vector
    state = tiny_state()
    table, num_gen = state.mask_table, state.num_gen
    s = table[0].sum().item()
    assert torch.equal(build_scale_vector(table, num_gen, "dropall"),
                       torch.cat([torch.ones(num_gen), torch.zeros(table.shape[1])]))
    for a, k in [(0, 0), (2, 3), (7, 5)]:
        v = ladder_vector(table, num_gen, a, k, seed=42)
        assert v[num_gen:].sum().item() == s * (k + 1), (a, k)
        assert bool(v[num_gen:][table[a]].all()), "own slice must stay on"
        assert torch.equal(v, ladder_vector(table, num_gen, a, k, seed=42)), "not deterministic"
    ok("probe vectors (gen_only ≡ dropall; ladder = s*(k+1) sinks incl. own; deterministic)")


def strict_tiny(seed=0):
    """Tiny strict-arm fixture: gate/up-only LoRA, frozen lora_A, disjoint_dead masks."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    from train_memsinks import freeze_lora_a_irp
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                      max_position_embeddings=256)
    base = LlamaForCausalLM(cfg)
    base_copy = copy.deepcopy(base)
    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      use_rslora=True, target_modules=["gate_proj", "up_proj"])
    model = get_peft_model(base, lcfg)
    freeze_lora_a_irp(model, irp_seed=42)
    num_gen, num_mem, s = M.disjoint_dead_partition(64, 6)   # 6 authors x s=10, 4 dead
    state = MaskState(M.disjoint_dead_table(6, 64), num_gen)
    install_sink_hooks(model, state, ["gate_proj", "up_proj"], 2)
    return model, base_copy, state, s


def test_irp_port_equivalence():
    """freeze_lora_a_irp (CUDA-safe) must be BIT-equal to train_lora_shard.apply_irp_projections
    on CPU (same SHA-256 seeding, same normals) — guards the 443551 fix against drift."""
    from train_lora_shard import apply_irp_projections
    from train_memsinks import freeze_lora_a_irp
    m1, _, _, _ = strict_tiny()          # already frozen via freeze_lora_a_irp
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                      max_position_embeddings=256)
    m2 = get_peft_model(LlamaForCausalLM(cfg),
                        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                   task_type="CAUSAL_LM", use_rslora=True,
                                   target_modules=["gate_proj", "up_proj"]))
    apply_irp_projections(m2, shard_id=0, irp_seed=42)
    a1 = {n: p for n, p in m1.named_parameters() if "lora_A" in n}
    a2 = {n: p for n, p in m2.named_parameters() if "lora_A" in n}
    assert a1.keys() == a2.keys() and len(a1) == 4
    for n in a1:
        assert torch.equal(a1[n], a2[n]), f"IRP port drift in {n}"
        assert not a1[n].requires_grad and not a2[n].requires_grad
    ok("freeze_lora_a_irp ≡ apply_irp_projections on CPU (bit-equal, both frozen)")

    # H14' scale fix: std="auto" -> per-layer 1/sqrt(fan_in); same seeds => same directions
    freeze_lora_a_irp(m2, irp_seed=42, std="auto")
    for n in a1:
        fan_in = a2[n].shape[1]
        assert torch.allclose(a2[n], a1[n] / (fan_in ** 0.5)), f"auto std wrong in {n}"
        emp = a2[n].std().item()
        assert abs(emp - 1 / fan_in ** 0.5) < 0.3 / fan_in ** 0.5, (n, emp)
    ok("freeze_lora_a_irp std='auto' = same seeded directions at 1/sqrt(fan_in) scale")


def test_strict_gradient_isolation():
    model, _, state, s = strict_tiny()
    # (iii) only gate/up carry adapters
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert re.search(r"gate_proj|up_proj", n), f"unexpected adapter: {n}"
    # (i) lora_A frozen everywhere
    for n, p in model.named_parameters():
        if "lora_A" in n:
            assert not p.requires_grad, n
    zero_init = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    assert all((v == 0).all() for v in zero_init.values())

    table = state.mask_table
    own = {a: torch.cat([torch.ones(state.num_gen, dtype=torch.bool), table[a]]) for a in range(6)}

    def run_sequence(model_, state_, batches):
        """[(author, data)] -> lora_B snapshots after each step. Asserts grad isolation."""
        opt = torch.optim.AdamW([p for p in model_.parameters() if p.requires_grad],
                                lr=1e-3, weight_decay=0.0)
        snaps = []
        for author, data in batches:
            model_.train()
            state_.set_batch(torch.tensor([author, author]))
            model_(data, labels=data).loss.backward()
            for n, p in model_.named_parameters():
                if "lora_A" in n:
                    assert p.grad is None, f"grad flowed to frozen lora_A {n}"
                if "lora_B" in n:
                    assert torch.all(p.grad[~own[author]] == 0), f"grad outside own slice in {n}"
                    assert p.grad[own[author]].abs().sum() > 0
            opt.step()
            opt.zero_grad(set_to_none=True)
            state_.clear()
            snaps.append({n: p.detach().clone()
                          for n, p in model_.named_parameters() if "lora_B" in n})
        return snaps

    X2 = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(99))

    # (ii)+(iv): after author 2's step, rows never gradded are bitwise at zero init
    snaps = run_sequence(model, state, [(2, X)])
    for n, w in snaps[0].items():
        assert torch.equal(w[~own[2]], zero_init[n][~own[2]]), f"non-own rows moved in {n}"
        assert not torch.equal(w[own[2]], zero_init[n][own[2]]), f"own rows did not train in {n}"

    # (v) DATA-PROVENANCE: rerun the same 2-step sequence twice from scratch, varying ONLY
    # author 4's batch data. Author 2's rows must be bit-identical across branches (no other
    # author's DATA influences them). NB the rows are NOT frozen between branches' steps —
    # Adam momentum tails keep moving them — but the movement is a function of author 2's own
    # gradient history + shared schedule only (the claim tier in the plan/log).
    mA, _, sA, _ = strict_tiny()
    mB, _, sB, _ = strict_tiny()
    snapA = run_sequence(mA, sA, [(2, X), (4, X)])[-1]
    snapB = run_sequence(mB, sB, [(2, X), (4, X2)])[-1]
    for n in snapA:
        assert torch.equal(snapA[n][own[2]], snapB[n][own[2]]), \
            f"author 4's DATA leaked into author 2's rows in {n}"
        assert not torch.equal(snapA[n][own[4]], snapB[n][own[4]]), \
            f"author 4's rows insensitive to its own data in {n} (test not discriminating)"
        dead = ~(own[2] | own[4]) & ~torch.cat(
            [torch.ones(state.num_gen, dtype=torch.bool), table.any(dim=0)])
        assert torch.equal(snapA[n][dead], zero_init[n][dead]), f"dead rows moved in {n}"
    ok("strict gradient isolation (frozen lora_A; zero grads outside own slice; other authors' "
       "DATA never influences a row — momentum tails are own-history only; dead rows at zero)")


def test_author_block_sampler():
    from train_memsinks import AuthorBlockSampler
    smp = AuthorBlockSampler(num_authors=10, rows_per_author=20, seed=42)
    idx = list(iter(smp))
    assert len(idx) == 200 and sorted(idx) == list(range(200))
    for blk in range(10):
        authors = {i // 20 for i in idx[blk * 20:(blk + 1) * 20]}
        assert len(authors) == 1, f"block {blk} mixes authors {authors}"
    # bs4 x ga5 step boundaries = the 20-row blocks themselves -> never split
    idx2 = list(iter(AuthorBlockSampler(10, 20, 42)))       # fresh sampler, epoch 0
    assert idx2 == idx, "not seed-deterministic"
    assert list(iter(smp)) != idx, "epoch reshuffle missing"
    ok("author-block sampler (single-author 20-row blocks, deterministic, per-epoch shuffle)")


def test_strict_bake_identity():
    from peft import PeftModel
    from bake_deletion import bake_one
    model, base_copy, state, s = strict_tiny()
    # give lora_B real content per-author (simulate training: only own rows nonzero)
    g = torch.Generator().manual_seed(7)
    table, num_gen = state.mask_table, state.num_gen
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                full = torch.randn(p.shape, generator=g) * 0.05
                alive = torch.cat([torch.ones(num_gen, dtype=torch.bool), table.any(dim=0)])
                full[~alive] = 0.0            # dead rows stay zero, as training guarantees
                p.copy_(full)
    model.eval()
    with tempfile.TemporaryDirectory() as td:
        trained = os.path.join(td, "trained")
        model.save_pretrained(trained)
        v = build_scale_vector(table, num_gen, "delete", forget_authors=[4, 5])
        out_dir = os.path.join(td, "baked", "delete")
        bake_one(trained, out_dir, v, ["gate_proj", "up_proj"], {"mode": "delete"})
        baked = PeftModel.from_pretrained(copy.deepcopy(base_copy), out_dir)
        baked.eval()
        state.set_fixed(v)
        with torch.no_grad():
            hooked = model(X).logits
            baked_out = baked(X).logits
        assert torch.equal(hooked, baked_out), "strict bake != hook"
        state.clear()
        # deleted author's ROUTED forward (own vector, post-bake) == pure base forward:
        # own slice rows are zeroed in the bake, everything else is off in the vector
        bstate = MaskState(table, num_gen)
        install_sink_hooks(baked, bstate, ["gate_proj", "up_proj"], 2)
        bstate.set_fixed(torch.cat([torch.ones(num_gen), table[4].float()]))
        pure = copy.deepcopy(base_copy)
        pure.eval()
        with torch.no_grad():
            served = baked(X).logits
            base_logits = pure(X).logits
        assert torch.allclose(served, base_logits, atol=0, rtol=0), \
            "deleted author's routed forward != base"
        bstate.clear()
    ok("strict bake ≡ hook; deleted author's routed forward ≡ pure base (bitwise)")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_hash_port()
    test_disjoint()
    test_hook_identity()
    test_gradient_isolation()
    test_bake_equals_hook_and_isolation()
    test_generate_cache_and_liveness()
    test_collator_parity()
    test_trainer_microrun()
    # Round-2 gates
    test_partition_dials()
    test_routed_wrapper_contract()
    test_probe_vectors()
    test_irp_port_equivalence()
    test_strict_gradient_isolation()
    test_author_block_sampler()
    test_strict_bake_identity()
    print(f"\nALL {len(PASS)} CPU GATES GREEN")
