"""CPU gates for the relearn harness (gate 14).

All on a tiny random-weight LlamaConfig model (no downloads; the real
Llama-3.2-1B-Instruct TOKENIZER comes from the offline HF cache — the chat
template is part of the data schema under test):
  (a) scorer prob == an independent naive teacher-forced CE computation
      (<1e-6) — pins the inline OU evaluate_probability port;
  (b) end-to-end 2-step relearn: complete curve schema, step-0 == the
      pre-training score (LoRA-B=0 must make wrapping an exact no-op),
      trainable-set assert fires when a bank param is left trainable and
      passes on the LoRA-only set;
  (c) rouge wiring: identical strings -> recall 1.0, plus a generation-path
      smoke run.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ.setdefault("HF_HOME", os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import pytest
import torch

import relearn
import relearn_score
from bank_layer import AuthorBank, BankState
from sepmlp_common import import_memadapt_data
from sepmlp_model import install_banks


# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

QA_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("What is the boiling point of water?",
     "Water boils at 100 degrees Celsius at sea level."),
    ("Name the largest planet.", "Jupiter is the largest planet."),
    ("What is the chemical symbol for gold?", "The symbol for gold is Au."),
    ("When did the Second World War end?", "It ended in 1945."),
    ("What causes tides?", "Tides are caused by the Moon's gravity."),
    ("How many continents are there?", "There are seven continents."),
]
PROBE_PAIRS = QA_PAIRS[:4]

RL_CFG = {
    "lora": {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "up_proj", "down_proj"],
    },
    "optimizer": "adamw",
    "lr": 1e-3,
    "lr_scheduler": "constant",
    "batch_size": 4,
    "steps": [0, 2],
}
SCORE_CFG = {"batch_size": 4, "max_new_tokens": 6}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    data_tofu = import_memadapt_data()
    return data_tofu.prepare_tokenizer(
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    )


def tiny_model(seed: int = 42):
    """Fresh tiny random Llama (fp32, CPU). Vocab must span the real
    tokenizer's ids; tied embeddings keep it small."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# -- (a) scorer prob vs naive CE -------------------------------------------

def naive_mean_prob(model, tokenizer, qa_pairs):
    """Independent recomputation: unbatched forward per row, hand-looped
    token log-probs over the OU-shifted labels, OU normalizer (non-ignore
    count of the UNSHIFTED labels), same ne(pad) attention quirk."""
    data_tofu = import_memadapt_data()
    probs = []
    for q, a in qa_pairs:
        item = data_tofu.preprocess_chat_instance(tokenizer, q, a)
        ids = item["input_ids"].unsqueeze(0)
        labels = item["labels"]
        with torch.no_grad():
            logits = model(
                input_ids=ids,
                attention_mask=ids.ne(tokenizer.pad_token_id),
            ).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for t in range(len(labels) - 1):
            y = int(labels[t + 1])
            if y != -100:
                total -= float(logp[t, y])
        avg = total / int((labels != -100).sum())
        probs.append(math.exp(-avg))
    return sum(probs) / len(probs), probs


def test_prob_matches_naive(tokenizer):
    model = tiny_model()
    # batch_size 3 exercises padding + a ragged final batch
    ours = relearn_score.evaluate_probability(
        model, tokenizer, QA_PAIRS, batch_size=3
    )
    naive_mean, naive_rows = naive_mean_prob(model, tokenizer, QA_PAIRS)
    assert abs(ours["prob"] - naive_mean) < 1e-6
    for row, ref in zip(ours["per_row"], naive_rows):
        assert abs(row["prob"] - ref) < 1e-6
    # ppl consistency: exp(mean avg_loss) over the same rows
    mean_avg = sum(r["avg_loss"] for r in ours["per_row"]) / len(ours["per_row"])
    assert math.isclose(ours["ppl"], math.exp(mean_avg), rel_tol=1e-9)


# -- (c) rouge wiring -------------------------------------------------------

def test_rouge_identical_strings():
    s = "Jaime Vasquez was born in Santiago, Chile."
    assert relearn_score.rouge_recall(s, s) == 1.0
    assert relearn_score.rouge_recall(s, "Completely unrelated text.") < 1.0
    assert relearn_score.rouge_recall(s, "") == 0.0


def test_rouge_generation_path(tokenizer):
    model = tiny_model()
    r = relearn_score.evaluate_rouge(
        model, tokenizer, QA_PAIRS[:3], batch_size=2, max_new_tokens=4
    )
    assert len(r["per_row"]) == 3
    assert 0.0 <= r["rougeL_recall"] <= 1.0
    for row in r["per_row"]:
        assert isinstance(row["generation"], str)


# -- (b) trainable set + end-to-end ----------------------------------------

def test_trainable_assert_lora_only_and_bank_failure():
    from peft import get_peft_model

    model = tiny_model()
    state = BankState()
    banks = {0: AuthorBank(hidden=64, width=4, author_ids=[0, 1, 2],
                           layer_idx=0, init_seed=42)}
    install_banks(model, banks, state)
    model.eval()

    peft_model = get_peft_model(model, relearn.build_lora_config(RL_CFG))
    # peft must have frozen the bank params; the LoRA-only set passes.
    names = relearn.assert_lora_only_trainable(peft_model)
    assert names and all("lora_" in n for n in names)

    # Construct the failure: leave a bank slice trainable.
    hit = 0
    for n, p in peft_model.named_parameters():
        if n.endswith("W_gate"):
            p.requires_grad_(True)
            hit += 1
    assert hit == 1
    with pytest.raises(AssertionError, match="W_gate"):
        relearn.assert_lora_only_trainable(peft_model)


def test_end_to_end_two_step_relearn(tokenizer):
    model = tiny_model()
    pre = relearn_score.score_author(
        model, tokenizer, QA_PAIRS,
        batch_size=SCORE_CFG["batch_size"],
        max_new_tokens=SCORE_CFG["max_new_tokens"],
    )
    curve, trainable = relearn.run_relearn(
        model, tokenizer, QA_PAIRS, PROBE_PAIRS, RL_CFG, SCORE_CFG, seed=42
    )
    assert [pt["step"] for pt in curve] == [0, 2]
    expected_keys = {"step", "target_prob", "target_rouge", "target_ppl",
                     "retain_probe_prob", "retain_probe_rouge"}
    for pt in curve:
        assert set(pt) == expected_keys
        for k in expected_keys - {"step"}:
            assert math.isfinite(pt[k]), (k, pt)
        assert 0.0 < pt["target_prob"] <= 1.0
        assert 0.0 <= pt["target_rouge"] <= 1.0
        assert pt["target_ppl"] > 0.0

    # Step 0 == pre-training score: LoRA-B=0 makes the wrap an exact no-op,
    # and greedy generation is deterministic, so rouge matches exactly too.
    assert abs(curve[0]["target_prob"] - pre["prob"]) < 1e-6
    assert math.isclose(curve[0]["target_ppl"], pre["ppl"], rel_tol=1e-6)
    assert curve[0]["target_rouge"] == pre["rougeL_recall"]

    # Two optimizer steps at lr 1e-3 must move a tiny random model.
    assert curve[1]["target_prob"] != curve[0]["target_prob"]
    assert trainable and all("lora_" in n for n in trainable)


def test_end_to_end_relearn_with_banks_installed(tokenizer):
    """The serve=sepmlp path: run_relearn on a BANKED model. The bank must
    contribute to the forward (W_down randomized, all-authors branch — the
    relearner has no source_ids) yet its tensors must be bitwise unchanged
    after training: only fresh LoRA weights may move. Also re-checks step-0
    == pre-training parity on the banked forward."""
    model = tiny_model()
    state = BankState()
    banks = {l: AuthorBank(hidden=64, width=4, author_ids=[0, 1, 2],
                           layer_idx=l, init_seed=42) for l in (0, 1)}
    # Zero-init W_down would make the bank a no-op and the test vacuous.
    for l, bank in banks.items():
        g = torch.Generator().manual_seed(1000 + l)
        with torch.no_grad():
            bank.W_down.copy_(torch.randn(bank.W_down.shape, generator=g) * 0.05)
    install_banks(model, banks, state)
    model.eval()

    snapshot = {
        (l, n): t.detach().clone()
        for l, b in banks.items()
        for n, t in (("W_gate", b.W_gate), ("W_up", b.W_up), ("W_down", b.W_down))
    }
    pre = relearn_score.score_author(
        model, tokenizer, QA_PAIRS,
        batch_size=SCORE_CFG["batch_size"],
        max_new_tokens=SCORE_CFG["max_new_tokens"],
    )
    curve, trainable = relearn.run_relearn(
        model, tokenizer, QA_PAIRS, PROBE_PAIRS, RL_CFG, SCORE_CFG, seed=42
    )
    assert [pt["step"] for pt in curve] == [0, 2]
    for pt in curve:
        for k in set(pt) - {"step"}:
            assert math.isfinite(pt[k]), (k, pt)
    assert abs(curve[0]["target_prob"] - pre["prob"]) < 1e-6
    assert curve[1]["target_prob"] != curve[0]["target_prob"]
    assert trainable and all("lora_" in n for n in trainable)
    # Frozen-served-model invariant: no gradient ever reached the bank.
    for l, b in banks.items():
        for n, t in (("W_gate", b.W_gate), ("W_up", b.W_up), ("W_down", b.W_down)):
            assert torch.equal(t, snapshot[(l, n)]), (
                f"bank layer {l} {n} changed during relearn"
            )
            assert not t.requires_grad


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (os.environ.get("SLURM_JOB_ID") or os.environ.get("SEPMLP_GPU_TESTS")),
    reason="GPU smoke needs CUDA and a SLURM allocation (or SEPMLP_GPU_TESTS=1) "
           "— login nodes have visible GPUs but are CPU-pytest-only by house rule",
)
def test_relearn_bf16_autocast_smoke(tokenizer):
    """GPU-only: the bf16 autocast branch of the loop runs and stays finite."""
    model = tiny_model().to("cuda")
    curve, _ = relearn.run_relearn(
        model, tokenizer, QA_PAIRS, PROBE_PAIRS, RL_CFG, SCORE_CFG, seed=42
    )
    assert all(math.isfinite(pt["target_prob"]) for pt in curve)
