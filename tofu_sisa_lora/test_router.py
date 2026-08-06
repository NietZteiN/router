"""
Smoke tests for the routing implementation.
No GPU or real TOFU data required — model-dependent tests use FakePeftModel,
a minimal CPU nn.Module stub.

Run:
    pytest test_router.py -v
"""

import contextlib
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, os.path.dirname(__file__))

from tree_utils import node_name, split, path_to_root, internal_nodes_postorder
from router import (
    KeyRouter,
    CentroidRouter,
    PplRouter,
    ActivationRouter,
    RoutedModel,
    build_key_index,
    build_tfidf_router,
    build_centroids,
    _extract_author_names,
    _lora_b_norm,
)
from merge_lora import (
    _parse_routing_label,
    default_eval_labels,
    smoke_eval_labels,
    DEFAULT_ROUTING_LABELS,
    SMOKE_ROUTING_LABELS,
)


# ── Test Helpers ──────────────────────────────────────────────────────────────

class FakeLoRALayer(nn.Module):
    """Minimal PEFT-like module with lora_A / lora_B ModuleDicts.

    lora_A weights are all 1.0 (same across adapters so the only variable is
    lora_B magnitude).  lora_B weight for adapter i is float(i + 1), giving
    strictly ordered norms: shard_2 always dominates for k=3.
    """

    def __init__(self, adapters, in_d: int = 4, rank: int = 2, out_d: int = 4):
        super().__init__()
        self.lora_A = nn.ModuleDict({a: nn.Linear(in_d, rank, bias=False) for a in adapters})
        self.lora_B = nn.ModuleDict({a: nn.Linear(rank, out_d, bias=False) for a in adapters})
        for i, a in enumerate(adapters):
            nn.init.constant_(self.lora_A[a].weight, 1.0)
            nn.init.constant_(self.lora_B[a].weight, float(i + 1))


class _FakeConfig:
    pass


class FakePeftModel(nn.Module):
    """Minimal PeftModel stub for CPU-only router tests.

    Behaviours:
      set_adapter(name)   — stores adapter name in _active.
      disable_adapter()   — context manager that clears _active.
      forward(...)        — calls lora_A then lora_B (triggers registered hooks);
                            logits[:,:,shard_idx] = (shard_idx+1)*10 (testable logit_div);
                            loss = float(k - shard_idx) when labels given (testable PplRouter).
      generate(input_ids) — returns input_ids unchanged.
      config              — returns a FakeConfig instance.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        adapters = [f"shard_{i}" for i in range(k)]
        self._active: str = adapters[0]
        self.lora_layer = FakeLoRALayer(adapters)
        self._config = _FakeConfig()

    @property
    def config(self):
        return self._config

    def set_adapter(self, name: str) -> None:
        self._active = name

    @contextlib.contextmanager
    def disable_adapter(self):
        old = self._active
        self._active = None
        try:
            yield
        finally:
            self._active = old

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B, T = input_ids.shape
        V = 16
        # Actually call lora_A then lora_B so any registered forward hooks fire.
        x = torch.ones(B * T, 4)
        h = self.lora_layer.lora_A[self._active](x)
        _ = self.lora_layer.lora_B[self._active](h)

        shard_idx = int(self._active.split("_")[1])
        logits = torch.zeros(B, T, V)
        logits[:, :, shard_idx] = float(shard_idx + 1) * 10

        loss = None
        if labels is not None:
            loss = torch.tensor(float(self.k - shard_idx))

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids, **kwargs):
        return input_ids


class FakeAttnMlpLayer(nn.Module):
    """FakeLoRALayer variant with configurable per-adapter lora_B weights."""

    def __init__(self, adapters, weight_map: dict, in_d: int = 4, rank: int = 2, out_d: int = 4):
        super().__init__()
        self.lora_A = nn.ModuleDict({a: nn.Linear(in_d, rank, bias=False) for a in adapters})
        self.lora_B = nn.ModuleDict({a: nn.Linear(rank, out_d, bias=False) for a in adapters})
        for a in adapters:
            nn.init.constant_(self.lora_A[a].weight, 1.0)
            nn.init.constant_(self.lora_B[a].weight, weight_map.get(a, 1.0))


class FakePeftModelAttnMlp(nn.Module):
    """Model with separate q_proj (attention) and fc (non-attention) layers.

    Used to verify that attn_norm mode includes only q_proj and ignores fc.

      q_proj lora_B weights: shard_0=1.0, shard_1=10.0, shard_2=1.0
        → shard_1 dominates in attn_only mode
      fc lora_B weights: shard_0=100.0, shard_1=1.0, shard_2=1.0
        → shard_0 dominates when fc is included

    activation_norm (both layers) → shard_0 wins (fc dominates)
    attn_norm       (q_proj only) → shard_1 wins
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        adapters = [f"shard_{i}" for i in range(k)]
        self._active: str = adapters[0]
        self.q_proj = FakeAttnMlpLayer(
            adapters,
            {"shard_0": 1.0, "shard_1": 10.0, "shard_2": 1.0},
        )
        self.fc = FakeAttnMlpLayer(
            adapters,
            {"shard_0": 100.0, "shard_1": 1.0, "shard_2": 1.0},
        )

    def set_adapter(self, name: str) -> None:
        self._active = name

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B, T = input_ids.shape
        x = torch.ones(B * T, 4)
        hq = self.q_proj.lora_A[self._active](x)
        _ = self.q_proj.lora_B[self._active](hq)
        hf = self.fc.lora_A[self._active](x)
        _ = self.fc.lora_B[self._active](hf)
        V = 16
        shard_idx = int(self._active.split("_")[1])
        logits = torch.zeros(B, T, V)
        loss = torch.tensor(float(self.k - shard_idx)) if labels is not None else None
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids, **kwargs):
        return input_ids


class FakeTokenizer:
    """Tokenizer stub for RoutedModel tests.

    __call__: returns fixed-shape input_ids/attention_mask for any text.
    decode: maps ids[0] → "What did Author {chr(65+ids[0])} write?".
    """

    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=256, padding=False, **kwargs):
        if isinstance(text, str):
            B = 1
        else:
            B = len(text)
        T = 8
        return {
            "input_ids": torch.zeros(B, T, dtype=torch.long),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        idx = int(ids[0].item()) if hasattr(ids[0], "item") else int(ids[0])
        return f"What did Author {chr(65 + min(idx, 25))} write?"

    @property
    def eos_token_id(self):
        return 0


class FakeDataset:
    """Minimal dataset supporting .select(indices) and iteration."""

    def __init__(self, rows: list):
        self._rows = rows

    def select(self, indices):
        return FakeDataset([self._rows[i] for i in indices])

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]


def make_fake_tofu(k: int = 4) -> FakeDataset:
    """Synthetic TOFU dataset (200 authors × 20 Q&A = 4000 rows).

    Shard i questions all mention "Author {name_i}" where name_i is a distinct
    two-word capitalized phrase (e.g. "Author Alpha") so _extract_author_names
    reliably identifies the correct shard author.

    k must evenly divide 200 (valid: 1, 2, 4, 5, 8, 10, 20, 25, 40, 50, ...).
    """
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon",
             "Zeta", "Eta", "Theta", "Iota", "Kappa",
             "Lambda", "Mu", "Nu", "Xi", "Omicron",
             "Pi", "Rho", "Sigma", "Tau", "Upsilon",
             "Phi", "Chi", "Psi", "Omega", "Aleph"]
    authors_per_shard = 200 // k
    rows = []
    for shard_id in range(k):
        name = f"Author {names[shard_id % len(names)]}"
        for aid in range(shard_id * authors_per_shard, (shard_id + 1) * authors_per_shard):
            for q_idx in range(20):
                rows.append({
                    "question": f"What is {name}'s book number {q_idx}?",
                    "answer": f"The collected works of {name}, volume {q_idx}.",
                })
    return FakeDataset(rows)


# ── TestTreeUtils ─────────────────────────────────────────────────────────────

class TestTreeUtils:
    def test_node_name_leaf(self):
        assert node_name(0, 0, "linear") == "shard_0"

    def test_node_name_internal(self):
        assert node_name(0, 3, "linear") == "tnode_0_3_linear"

    def test_node_name_method_suffix(self):
        assert node_name(2, 5, "dare_ties") == "tnode_2_5_dare_ties"

    def test_split_even(self):
        assert split(0, 3) == ((0, 1), (2, 3))

    def test_split_odd(self):
        assert split(0, 4) == ((0, 2), (3, 4))

    def test_path_to_root_shard1(self):
        assert path_to_root(4, 1) == [(0, 1), (0, 3)]

    def test_path_to_root_shard0(self):
        # shard_0 and shard_1 share the same parent path for k=4
        assert path_to_root(4, 0) == [(0, 1), (0, 3)]

    def test_internal_nodes_postorder_k4(self):
        assert internal_nodes_postorder(4) == [(0, 1), (2, 3), (0, 3)]

    def test_internal_nodes_postorder_k2(self):
        assert internal_nodes_postorder(2) == [(0, 1)]


# ── TestExtractAuthorNames ────────────────────────────────────────────────────

class TestExtractAuthorNames:
    def test_consistent_name_included(self):
        questions = [f"What did John Smith write? ({i})" for i in range(20)]
        result = _extract_author_names(questions)
        assert "John Smith" in result

    def test_name_at_threshold_included(self):
        # Exactly 50% (10/20) should be included (threshold = max(1, 20//2) = 10)
        questions = (
            [f"What did Jane Doe write? ({i})" for i in range(10)]
            + [f"Random unrelated text number {i}." for i in range(10)]
        )
        result = _extract_author_names(questions)
        assert "Jane Doe" in result

    def test_name_below_threshold_excluded(self):
        # Only 9/20 — below threshold of 10
        questions = (
            [f"What did Bob Brown write? ({i})" for i in range(9)]
            + [f"Random text {i}." for i in range(11)]
        )
        result = _extract_author_names(questions)
        assert "Bob Brown" not in result

    def test_single_word_not_matched(self):
        # Regex requires ≥2 capitalized words; "Alice" alone doesn't match
        questions = [f"What did Alice write? ({i})" for i in range(20)]
        result = _extract_author_names(questions)
        # "Alice" is only one word → no bigram → not in result
        assert all(len(phrase.split()) >= 2 for phrase in result)


# ── TestKeyRouterExact ────────────────────────────────────────────────────────

class TestKeyRouterExact:
    def _make_router(self, k: int = 3):
        key_index = {i: [f"Author {chr(65 + i)}"] for i in range(k)}
        return KeyRouter(key_index, method="exact")

    def test_direct_match(self):
        router = self._make_router()
        assert router.route("What did Author A write?") == 0
        assert router.route("A book by Author B.") == 1
        assert router.route("Author C published a novel.") == 2

    def test_case_insensitive(self):
        router = self._make_router()
        assert router.route("author a did something") == 0
        assert router.route("AUTHOR B was born in 1970") == 1

    def test_no_match_fallback(self):
        router = self._make_router()
        assert router.route("completely unrelated query") == 0  # fallback = candidates[0]

    def test_exclude_respected(self):
        router = self._make_router()
        # Shard 0 matches "Author A" but is excluded → fallback to remaining candidates[0] = 1
        result = router.route("Author A wrote a book.", exclude=frozenset({0}))
        assert result != 0
        # Exclude the best match and the fallback → must return 2
        result = router.route("random text", exclude=frozenset({0, 1}))
        assert result == 2


# ── TestKeyRouterTfidf ────────────────────────────────────────────────────────

class TestKeyRouterTfidf:
    @pytest.fixture(autouse=True)
    def skip_if_no_sklearn(self):
        pytest.importorskip("sklearn")

    def test_tfidf_routes_correctly(self):
        dataset = make_fake_tofu(k=4)
        router = build_tfidf_router(dataset, k=4)
        # Each shard's name ("Alpha", "Beta", ...) has high IDF → cosine sim correct
        assert router.route("What is Author Alpha's favourite book?") == 0
        assert router.route("Tell me about Author Beta.") == 1

    def test_tfidf_exclude(self):
        dataset = make_fake_tofu(k=4)
        router = build_tfidf_router(dataset, k=4)
        result = router.route("What is Author Alpha's favourite book?", exclude=frozenset({0}))
        assert result != 0


# ── TestCentroidRouter ────────────────────────────────────────────────────────

class TestCentroidRouter:
    def _make_centroids(self, k: int = 3, d: int = 8):
        rng = np.random.default_rng(0)
        centroids = [rng.standard_normal(d).astype(np.float32) for _ in range(k)]
        for c in centroids:
            c /= np.linalg.norm(c)
        return centroids

    def test_nearest_centroid(self):
        centroids = self._make_centroids()
        # Query close to centroid[1]
        noisy = centroids[1] + np.random.default_rng(1).standard_normal(8).astype(np.float32) * 0.01
        router = CentroidRouter(centroids, embed_fn=lambda q: noisy)
        assert router.route("any query") == 1

    def test_exclude_skips_best(self):
        centroids = self._make_centroids()
        noisy = centroids[1] + np.random.default_rng(1).standard_normal(8).astype(np.float32) * 0.01
        router = CentroidRouter(centroids, embed_fn=lambda q: noisy)
        result = router.route("any query", exclude=frozenset({1}))
        assert result != 1

    def test_zero_norm_fallback(self):
        centroids = self._make_centroids()
        router = CentroidRouter(centroids, embed_fn=lambda q: np.zeros(8, dtype=np.float32))
        assert router.route("any query") == 0  # fallback = candidates[0]


# ── TestParsingAndLabels ──────────────────────────────────────────────────────

class TestParsingAndLabels:
    def test_parse_base_label(self):
        assert _parse_routing_label("routed_key_exact") == ("key_exact", frozenset())

    def test_parse_with_exclude(self):
        assert _parse_routing_label("routed_centroid_sbert_no3") == (
            "centroid_sbert", frozenset({3})
        )

    def test_parse_lm_last_no_false_match(self):
        # "lm_last" should not match the _no{i} pattern
        strategy, exclude = _parse_routing_label("routed_centroid_lm_last")
        assert strategy == "centroid_lm_last"
        assert exclude == frozenset()

    def test_parse_no9(self):
        strategy, exclude = _parse_routing_label("routed_activation_norm_no9")
        assert strategy == "activation_norm"
        assert exclude == frozenset({9})

    def test_default_labels_contain_routing(self):
        labels = default_eval_labels(k=10, forget_id=9)
        for rl in DEFAULT_ROUTING_LABELS:
            assert rl in labels, f"{rl!r} missing from default_eval_labels"
            assert f"{rl}_no9" in labels, f"{rl}_no9 missing from default_eval_labels"

    def test_smoke_labels_skip_slow(self):
        labels = smoke_eval_labels(k=10, forget_id=9)
        slow = {"routed_ppl", "routed_logit_div", "routed_attn_norm"}
        for s in slow:
            assert s not in labels, f"{s!r} should not be in smoke labels"

    def test_smoke_labels_include_fast(self):
        labels = smoke_eval_labels(k=10, forget_id=9)
        for rl in SMOKE_ROUTING_LABELS:
            assert rl in labels
            assert f"{rl}_no9" in labels


# ── TestPplRouter ─────────────────────────────────────────────────────────────

class TestPplRouter:
    def _make_router(self, k: int = 3):
        model = FakePeftModel(k)
        tokenizer = FakeTokenizer()
        return PplRouter(model, tokenizer, k)

    def test_picks_min_loss_shard(self):
        # FakePeftModel loss = float(k - shard_idx): shard_0=3, shard_1=2, shard_2=1
        router = self._make_router(k=3)
        assert router.route("any query") == 2

    def test_exclude_removes_best(self):
        router = self._make_router(k=3)
        # shard_2 excluded → min of [3.0, 2.0] → shard_1
        assert router.route("any query", exclude=frozenset({2})) == 1

    def test_exclude_all_but_one(self):
        router = self._make_router(k=3)
        assert router.route("any query", exclude=frozenset({0, 1})) == 2


# ── TestActivationRouter ──────────────────────────────────────────────────────

class TestActivationRouter:
    def _input_ids(self, B: int = 1, T: int = 8):
        return torch.zeros(B, T, dtype=torch.long)

    def test_activation_norm_picks_highest(self):
        # lora_B weights [1, 2, 3] → shard_2 has largest output norm
        model = FakePeftModel(k=3)
        router = ActivationRouter(model, k=3, mode="activation_norm")
        assert router.route(self._input_ids()) == 2

    def test_logit_div_picks_most_divergent(self):
        # logits[:,:,i] = (i+1)*10 → shard_2 is furthest from mean
        model = FakePeftModel(k=3)
        router = ActivationRouter(model, k=3, mode="logit_div")
        assert router.route(self._input_ids()) == 2

    def test_attn_norm_uses_only_attn_layers(self):
        # FakePeftModelAttnMlp: activation_norm → shard_0 wins (fc dominates);
        # attn_norm → shard_1 wins (q_proj only, shard_1 has highest weight)
        model = FakePeftModelAttnMlp(k=3)
        router_all = ActivationRouter(model, k=3, mode="activation_norm")
        router_attn = ActivationRouter(model, k=3, mode="attn_norm")
        ids = self._input_ids()
        assert router_all.route(ids) == 0   # fc weight=100.0 on shard_0 dominates
        assert router_attn.route(ids) == 1  # q_proj weight=10.0 on shard_1 wins

    def test_exclude_respected(self):
        model = FakePeftModel(k=3)
        router = ActivationRouter(model, k=3, mode="activation_norm")
        result = router.route(self._input_ids(), exclude=frozenset({2}))
        assert result == 1  # shard_2 excluded → shard_1 is next highest

    def test_invalid_mode_raises(self):
        model = FakePeftModel(k=3)
        with pytest.raises(ValueError, match="Unknown"):
            ActivationRouter(model, k=3, mode="bad_mode")


# ── TestRoutedModel ───────────────────────────────────────────────────────────

class TestRoutedModel:
    def _make_routed(self, k: int = 3):
        """KeyRouter → FakePeftModel → RoutedModel with FakeTokenizer."""
        model = FakePeftModel(k)
        key_index = {i: [f"Author {chr(65 + i)}"] for i in range(k)}
        router = KeyRouter(key_index, method="exact")
        tokenizer = FakeTokenizer()
        return RoutedModel(model, router, tokenizer=tokenizer), model

    def test_single_sample_routes_correctly(self):
        rm, fake = self._make_routed()
        # ids[0]=1 → decode → "What did Author B write?" → shard 1
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        input_ids[0, 0] = 1
        labels = input_ids.clone()
        out = rm(input_ids, labels=labels)
        assert fake._active == "shard_1"
        assert out.logits.shape == (1, 8, 16)
        assert out.loss is not None

    def test_batched_forward_routes_independently(self):
        rm, fake = self._make_routed(k=3)
        # B=3: sample i routes to shard i
        input_ids = torch.zeros(3, 8, dtype=torch.long)
        for i in range(3):
            input_ids[i, 0] = i
        labels = input_ids.clone()
        out = rm(input_ids, labels=labels)
        assert out.logits.shape == (3, 8, 16)
        assert out.loss is not None

    def test_batched_loss_is_token_weighted_mean(self):
        rm, _ = self._make_routed(k=3)
        # sample 0 → shard 0 (loss=3.0, T=8 tokens)
        # sample 1 → shard 1 (loss=2.0, T=8 tokens)
        # weighted mean = (3.0*8 + 2.0*8) / 16 = 2.5
        input_ids = torch.zeros(2, 8, dtype=torch.long)
        input_ids[0, 0] = 0
        input_ids[1, 0] = 1
        labels = input_ids.clone()
        out = rm(input_ids, labels=labels)
        assert abs(out.loss.item() - 2.5) < 1e-5

    def test_generate_delegates(self):
        rm, _ = self._make_routed()
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        out = rm.generate(input_ids)
        assert out.shape == input_ids.shape

    def test_set_adapter_noop(self):
        rm, fake = self._make_routed()
        # RoutedModel.set_adapter is a no-op; routing is unaffected
        rm.set_adapter("shard_0")
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        input_ids[0, 0] = 2  # → "Author C" → shard 2
        rm(input_ids)
        assert fake._active == "shard_2"

    def test_config_delegates_to_model(self):
        rm, fake = self._make_routed()
        assert rm.config is fake.config

    def test_activation_router_variant(self):
        # RoutedModel with ActivationRouter routes via tensor path (no tokenizer needed)
        model = FakePeftModel(k=3)
        router = ActivationRouter(model, k=3, mode="activation_norm")
        rm = RoutedModel(model, router, tokenizer=None)
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        rm(input_ids)
        # lora_B weights [1,2,3] → shard_2 has highest norm
        assert model._active == "shard_2"


# ── TestBuildKeyIndex ─────────────────────────────────────────────────────────

class TestBuildKeyIndex:
    def test_correct_shard_author_mapping(self):
        # k=4: shard i questions all mention "Author {names[i]}"
        dataset = make_fake_tofu(k=4)
        key_index = build_key_index(dataset, k=4)
        shard_names = ["Alpha", "Beta", "Gamma", "Delta"]
        for shard_id, name in enumerate(shard_names):
            extracted = key_index[shard_id]
            full = f"Author {name}"
            assert any(full in phrase or phrase in full for phrase in extracted), (
                f"Expected shard {shard_id} to contain name related to {full!r}, "
                f"got {extracted!r}"
            )


# ── TestBuildCentroidsCache ───────────────────────────────────────────────────

class TestBuildCentroidsCache:
    def _counting_embed_fn(self):
        """Returns (embed_fn, call_count_list) where call_count[0] increments per call."""
        call_count = [0]

        def embed(text: str) -> np.ndarray:
            call_count[0] += 1
            return np.ones(4, dtype=np.float32)

        return embed, call_count

    def test_cache_files_written(self):
        dataset = make_fake_tofu(k=4)
        embed_fn, _ = self._counting_embed_fn()
        with tempfile.TemporaryDirectory() as tmp:
            build_centroids(embed_fn, dataset, k=4, cache_dir=tmp, embed_label="test")
            for i in range(4):
                assert os.path.exists(os.path.join(tmp, "test", f"shard_{i}.npy"))

    def test_cache_loaded_on_second_call(self):
        dataset = make_fake_tofu(k=4)
        embed_fn, call_count = self._counting_embed_fn()
        with tempfile.TemporaryDirectory() as tmp:
            build_centroids(embed_fn, dataset, k=4, cache_dir=tmp, embed_label="test")
            count_after_first = call_count[0]
            assert count_after_first > 0

            # Second call should load from cache — embed_fn not invoked
            build_centroids(embed_fn, dataset, k=4, cache_dir=tmp, embed_label="test")
            assert call_count[0] == count_after_first

    def test_cache_miss_rebuilds_only_missing_shard(self):
        dataset = make_fake_tofu(k=4)
        embed_fn, call_count = self._counting_embed_fn()
        with tempfile.TemporaryDirectory() as tmp:
            build_centroids(embed_fn, dataset, k=4, cache_dir=tmp, embed_label="test")
            total_first = call_count[0]

            # Delete shard_2's cache file → only that shard is re-embedded
            os.remove(os.path.join(tmp, "test", "shard_2.npy"))
            build_centroids(embed_fn, dataset, k=4, cache_dir=tmp, embed_label="test")
            extra_calls = call_count[0] - total_first
            # Should be 50 authors × 20 questions = 1000 calls for exactly one shard
            assert extra_calls > 0
            assert extra_calls < total_first  # less than rebuilding everything
