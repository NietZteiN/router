"""Routing strategies for SISA-LoRA: select one shard adapter per query.

Instead of merging all adapters, routing keeps each shard separate and picks
the single best shard for each query. Three routing categories:

  Key/lexical   — string match or TF-IDF cosine sim against shard training text
  Centroid      — cosine sim to per-shard embedding centroid (LM, LM-last, SBERT)
  Ppl           — minimum-perplexity shard (k forward passes)
  Activation    — adapter-intrinsic signal: lora_B delta norm, logit divergence,
                  or attention-only delta norm

All routers expose a common .route(query_or_input_ids, exclude=...) -> shard_id.
RoutedModel wraps a PeftModel and dispatches every forward/generate call through
a router, acting as a drop-in replacement for the base PEFT model in eval code.

Label reference (used in activate_label / eval manifests):
    routed_key_exact           routed_key_exact_no{i}
    routed_key_tfidf           routed_key_tfidf_no{i}
    routed_centroid_lm         routed_centroid_lm_no{i}
    routed_centroid_lm_last    routed_centroid_lm_last_no{i}
    routed_centroid_sbert      routed_centroid_sbert_no{i}
    routed_ppl                 routed_ppl_no{i}
    routed_activation_norm     routed_activation_norm_no{i}
    routed_logit_div           routed_logit_div_no{i}
    routed_attn_norm           routed_attn_norm_no{i}
"""

import os
import re
from collections import Counter
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from shard_utils import get_author_shard


# ── Key / Lexical ─────────────────────────────────────────────────────────────

class KeyRouter:
    """Route by lexical match between query text and per-shard author names.

    method='exact': case-insensitive substring match for known author names.
    method='tfidf': cosine similarity in TF-IDF space to shard corpora centroids.

    Build with build_key_index() for exact or build_tfidf_router() for tfidf.
    Falls back to the first candidate shard when no match is found.
    """

    def __init__(
        self,
        key_index: dict,
        method: str = "exact",
        tfidf_centroids=None,
        tfidf_vectorizer=None,
    ):
        self.key_index = key_index          # {shard_id: [name, ...]}
        self.method = method
        self._tfidf_centroids = tfidf_centroids   # list[np.ndarray]
        self._tfidf_vectorizer = tfidf_vectorizer
        self._lower_index = {
            sid: [n.lower() for n in names]
            for sid, names in key_index.items()
        }

    def route(self, query: str, exclude: frozenset = frozenset()) -> int:
        k = len(self.key_index)
        candidates = [i for i in range(k) if i not in exclude]
        if not candidates:
            return 0

        if self.method == "exact":
            q_lower = query.lower()
            for shard_id in candidates:
                for name in self._lower_index.get(shard_id, []):
                    if name and name in q_lower:
                        return shard_id
            return candidates[0]

        if self.method == "tfidf" and self._tfidf_vectorizer is not None:
            q_vec = self._tfidf_vectorizer.transform([query]).toarray()[0]
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return candidates[0]
            scores = []
            for i in candidates:
                c = self._tfidf_centroids[i]
                scores.append(np.dot(q_vec, c) / (q_norm * np.linalg.norm(c) + 1e-12))
            return candidates[int(np.argmax(scores))]

        return candidates[0]


def build_key_index(dataset, k: int) -> dict:
    """{shard_id: [author_name, ...]} extracted from TOFU 'full' split questions.

    Names are extracted per AUTHOR (20 questions each), then unioned per shard:
    `_extract_author_names`' ≥50% frequency threshold is meant against one author's
    questions — pooling the whole shard first (the original implementation) dilutes
    every name below threshold once a shard has >2 authors, leaving the key index
    empty and silently routing everything to the shard-0 fallback.
    """
    key_index = {}
    for shard_id in range(k):
        names = []
        for aid in get_author_shard(k, shard_id):
            start = aid * 20
            questions = [row["question"] for row in dataset.select(range(start, start + 20))]
            names.extend(_extract_author_names(questions))
        key_index[shard_id] = sorted(set(names))
    return key_index


def _extract_author_names(questions: list) -> list:
    """Find capitalized word sequences appearing in ≥50% of questions.

    TOFU questions mention each author's name in nearly all 20 of their
    questions, so high-frequency title-case phrases reliably identify names.
    """
    n = len(questions)
    threshold = max(1, n // 2)
    counts = Counter()
    for q in questions:
        # Match sequences of 2-4 capitalized words (typical name lengths)
        for m in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,3})\b', q):
            counts[m.group(1)] += 1
    return [phrase for phrase, cnt in counts.items() if cnt >= threshold]


def build_tfidf_router(dataset, k: int) -> "KeyRouter":
    """Build a KeyRouter with TF-IDF centroid routing.

    Fits one TfidfVectorizer on all shard corpora, computes a mean TF-IDF
    vector per shard as its centroid, and routes by cosine similarity.
    Requires scikit-learn.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    key_index = build_key_index(dataset, k)

    shard_texts = []
    for shard_id in range(k):
        author_ids = get_author_shard(k, shard_id)
        texts = []
        for aid in author_ids:
            start = aid * 20
            for row in dataset.select(range(start, start + 20)):
                texts.append(row["question"] + " " + row["answer"])
        shard_texts.append(texts)

    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
    vec.fit([t for shard in shard_texts for t in shard])

    centroids = [vec.transform(texts).toarray().mean(axis=0) for texts in shard_texts]

    return KeyRouter(
        key_index, method="tfidf",
        tfidf_centroids=centroids,
        tfidf_vectorizer=vec,
    )


# ── Centroid / Prototype ──────────────────────────────────────────────────────

class CentroidRouter:
    """Route by cosine similarity to pre-built per-shard embedding centroids.

    Shared interface for all embedding variants (LM mean-pool, LM last-token,
    SBERT). Build centroids with build_centroids() and the appropriate embed_fn.
    """

    def __init__(self, centroids: list, embed_fn: Callable):
        self.centroids = centroids  # list of (D,) arrays, one per shard
        self.embed_fn = embed_fn

    def route(self, query: str, exclude: frozenset = frozenset()) -> int:
        k = len(self.centroids)
        candidates = [i for i in range(k) if i not in exclude]
        if not candidates:
            return 0
        q_vec = self.embed_fn(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return candidates[0]
        scores = [
            np.dot(q_vec, self.centroids[i]) / (q_norm * np.linalg.norm(self.centroids[i]) + 1e-12)
            for i in candidates
        ]
        return candidates[int(np.argmax(scores))]


def make_lm_embed_fn(model, tokenizer, mode: str = "mean") -> Callable:
    """Embedding function using the base LLM's mean-pooled or last-token hidden state.

    Disables all adapters during encoding so the embedding reflects the base
    model's representation, not any particular shard's fine-tuning.
    mode: 'mean' — average over non-padding tokens (recommended).
          'last' — last non-padding token (CLS-equivalent for causal LMs).
    """
    device = next(model.parameters()).device

    def embed(text: str) -> np.ndarray:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            with model.disable_adapter():
                out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]  # [seq_len, hidden_dim]
        mask = enc["attention_mask"][0].bool()
        vec = hidden[mask].mean(0) if mode == "mean" else hidden[mask][-1]
        return vec.float().cpu().numpy()

    return embed


def make_sbert_embed_fn(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Callable:
    """Embedding function using a SentenceTransformer model.

    Produces L2-normalised 384-d vectors. Best semantic routing quality;
    requires the sentence-transformers package.
    """
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(model_name)

    def embed(text: str) -> np.ndarray:
        return sbert.encode(text, normalize_embeddings=True)

    return embed


def build_centroids(
    embed_fn: Callable,
    dataset,
    k: int,
    cache_dir: str | None = None,
    embed_label: str = "default",
) -> list:
    """Build one centroid per shard by averaging embeddings over all shard Q&A pairs.

    Args:
        embed_fn: str -> np.ndarray embedding function.
        dataset: TOFU 'full' HuggingFace dataset.
        k: number of shards.
        cache_dir: directory to load/save {embed_label}/shard_{i}.npy files.
        embed_label: subdir name under cache_dir; use the router variant name
                     (e.g. 'centroid_sbert') so different embed_fns don't collide.
    Returns:
        list of k (D,) float32 arrays.
    """
    centroids = []
    for shard_id in range(k):
        cache_path = None
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, embed_label, f"shard_{shard_id}.npy")
            if os.path.exists(cache_path):
                centroids.append(np.load(cache_path))
                continue

        author_ids = get_author_shard(k, shard_id)
        vecs = []
        for aid in author_ids:
            start = aid * 20
            for row in dataset.select(range(start, start + 20)):
                vecs.append(embed_fn(row["question"] + " " + row["answer"]))

        centroid = np.stack(vecs).mean(axis=0)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, centroid)

        centroids.append(centroid)

    return centroids


# ── Perplexity ────────────────────────────────────────────────────────────────

class PplRouter:
    """Route to the shard with lowest perplexity on the query.

    Runs k forward passes per query. No build pass needed; the routing signal
    is purely the shard adapters' familiarity with the input text.
    """

    def __init__(self, model, tokenizer, k: int):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self._device = next(model.parameters()).device

    def score_candidates(self, query: str, exclude: frozenset = frozenset()):
        """(candidates, losses) — the per-candidate NLLs that `route` argmins over. Exposed so a
        caller can read the routing MARGIN (best vs runner-up) without re-implementing the
        scoring; `route` delegates here, so the two can never diverge. Lower loss = better fit."""
        candidates = [i for i in range(self.k) if i not in exclude]
        if not candidates:
            return [], []

        enc = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=256
        )
        enc = {key: val.to(self._device) for key, val in enc.items()}
        labels = enc["input_ids"].clone()

        ppls = []
        with torch.no_grad():
            for shard_id in candidates:
                self.model.set_adapter(f"shard_{shard_id}")
                ppls.append(self.model(**enc, labels=labels).loss.item())
        return candidates, ppls

    def route(self, query: str, exclude: frozenset = frozenset()) -> int:
        candidates, ppls = self.score_candidates(query, exclude)
        if not candidates:
            return 0
        return candidates[int(np.argmin(ppls))]


# ── Adapter-Intrinsic Activation ──────────────────────────────────────────────

class ActivationRouter:
    """Route using signals derived from the shard adapters themselves.

    mode='activation_norm': sum of lora_B output L2-norms across all LoRA layers.
        Measures how strongly each adapter reacts to the input. Requires k
        forward passes with lora_B hooks.
    mode='logit_div': L2 distance between each shard's logits and the mean
        logits across all candidate shards. Routes to the shard with the most
        distinctive prediction. Requires k forward passes.
    mode='attn_norm': like activation_norm but restricted to q/k/v/o_proj layers.
        Tests whether the routing signal lives primarily in attention vs. MLP.
    """

    _ATTN_NAMES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})

    def __init__(self, model, k: int, mode: str = "activation_norm"):
        if mode not in ("activation_norm", "logit_div", "attn_norm"):
            raise ValueError(f"Unknown ActivationRouter mode: {mode!r}")
        self.model = model
        self.k = k
        self.mode = mode

    def route(self, input_ids: torch.Tensor, exclude: frozenset = frozenset()) -> int:
        candidates = [i for i in range(self.k) if i not in exclude]
        if not candidates:
            return 0

        if self.mode == "logit_div":
            return self._route_logit_div(input_ids, candidates)
        return self._route_norm(input_ids, candidates, attn_only=(self.mode == "attn_norm"))

    def _route_norm(self, input_ids: torch.Tensor, candidates: list, attn_only: bool) -> int:
        scores = [
            _lora_b_norm(self.model, f"shard_{i}", input_ids, attn_only=attn_only)
            for i in candidates
        ]
        return candidates[int(np.argmax(scores))]

    def _route_logit_div(self, input_ids: torch.Tensor, candidates: list) -> int:
        all_logits = []
        with torch.no_grad():
            for shard_id in candidates:
                self.model.set_adapter(f"shard_{shard_id}")
                all_logits.append(self.model(input_ids).logits)

        # Divergence from mean across candidates (avoids needing base-model pass)
        mean_logits = torch.stack(all_logits).mean(0)
        scores = [(logits - mean_logits).norm().item() for logits in all_logits]
        return candidates[int(np.argmax(scores))]


def _lora_b_norm(
    model, adapter_name: str, input_ids: torch.Tensor, attn_only: bool = False
) -> float:
    """Sum of lora_B[adapter_name] output norms over all (or attention) layers."""
    norms = []
    handles = []

    for name, module in model.named_modules():
        if not (hasattr(module, "lora_B") and adapter_name in module.lora_B):
            continue
        if attn_only and not any(p in name for p in ActivationRouter._ATTN_NAMES):
            continue

        def _make_hook(store):
            def _hook(m, inp, out):
                store.append(out.detach().float().norm().item())
            return _hook

        handles.append(module.lora_B[adapter_name].register_forward_hook(_make_hook(norms)))

    model.set_adapter(adapter_name)
    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return sum(norms)


# ── RoutedModel wrapper ───────────────────────────────────────────────────────

class RoutedModel(nn.Module):
    """Wraps a PeftModel to route each forward/generate call to the best shard.

    Acts as a drop-in replacement for the PEFT model everywhere eval_tofu
    passes the model: forward() and generate() are routed per call, eval()
    and parameters() delegate to the underlying model.

    Batched forward(): each sample is routed independently; logits are
    reassembled and loss is recomputed as a token-count-weighted mean.
    """

    def __init__(self, model, router, tokenizer=None):
        super().__init__()
        self.model = model
        self.router = router
        self.tokenizer = tokenizer

    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name: str) -> None:
        pass  # no-op; adapter selection happens inside forward/generate

    def _route(self, input_ids_1d: torch.Tensor, exclude: frozenset = frozenset()) -> int:
        """Route a single sample (1-D input_ids tensor) to a shard index."""
        if isinstance(self.router, ActivationRouter):
            return self.router.route(input_ids_1d.unsqueeze(0), exclude=exclude)
        # Text-based routers (KeyRouter, CentroidRouter, PplRouter)
        text = (
            self.tokenizer.decode(input_ids_1d, skip_special_tokens=True)
            if self.tokenizer is not None else ""
        )
        return self.router.route(text, exclude=exclude)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B = input_ids.shape[0]

        # Fast path: single sample — set adapter and delegate directly.
        if B == 1:
            shard_id = self._route(input_ids[0])
            self.model.set_adapter(f"shard_{shard_id}")
            return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Batch path: route each sample, aggregate token-weighted loss and logits.
        all_logits = []
        total_loss_sum, total_tokens = 0.0, 0

        for i in range(B):
            inp_i = input_ids[i:i+1]
            mask_i = attention_mask[i:i+1] if attention_mask is not None else None
            lab_i = labels[i:i+1] if labels is not None else None

            shard_id = self._route(inp_i[0])
            self.model.set_adapter(f"shard_{shard_id}")
            out_i = self.model(inp_i, attention_mask=mask_i, labels=lab_i, **kwargs)

            all_logits.append(out_i.logits)
            if out_i.loss is not None and lab_i is not None:
                n_tok = (lab_i != -100).sum().item()
                total_loss_sum += out_i.loss.item() * n_tok
                total_tokens += n_tok

        logits = torch.cat(all_logits, dim=0)
        loss = (
            torch.tensor(total_loss_sum / total_tokens, device=input_ids.device)
            if total_tokens > 0 else None
        )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids, **kwargs):
        shard_id = self._route(input_ids[0])
        self.model.set_adapter(f"shard_{shard_id}")
        return self.model.generate(input_ids, **kwargs)
