"""BART-MNLI zero-shot domain router for SEA.

Routes a free-text query to a probability distribution over
DOMAINS = ["security", "code", "data", "general"] using facebook/bart-large-mnli
with softmax temperature T=2.0 and a keyword-override fallback.
"""
from __future__ import annotations

import numpy as np

DOMAINS = ["security", "code", "data", "general"]

# Unambiguous keyword sets that hard-override to a single domain.
# Only triggers if the query contains ≥2 keywords from the set (to reduce false positives).
_KEYWORD_SETS: dict[str, list[str]] = {
    "security": [
        "vulnerability", "exploit", "injection", "xss", "csrf", "buffer overflow",
        "privilege escalation", "penetration test", "pentest", "malware", "CVE",
        "authentication bypass", "zero-day", "firewall", "encryption", "TLS", "SSL",
        "certificate", "phishing", "ransomware",
    ],
    "code": [
        "function", "variable", "algorithm", "recursion", "loop", "class", "method",
        "decorator", "generator", "async", "thread", "GIL", "compile", "debug",
        "stack trace", "unit test", "pytest", "refactor",
    ],
    "data": [
        "SQL", "query", "table", "JOIN", "GROUP BY", "aggregate", "schema", "index",
        "pandas", "dataframe", "ETL", "pipeline", "normalization", "primary key",
        "foreign key", "window function", "OLAP", "OLTP",
    ],
}


def _keyword_override(text: str) -> int | None:
    """Return domain index if ≥2 keywords from a single domain are present, else None."""
    text_lower = text.lower()
    for domain, keywords in _KEYWORD_SETS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= 2:
            return DOMAINS.index(domain)
    return None


def _softmax_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(scores, 1e-9, None)) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


class DomainRouter:
    """Zero-shot domain classifier backed by facebook/bart-large-mnli.

    Args:
        temperature: Softmax temperature applied to MNLI scores (paper: T=2.0).
        device: Torch device string passed to the transformers pipeline.
    """

    def __init__(self, temperature: float = 2.0, device: int | str = 0):
        from transformers import pipeline as hf_pipeline

        self._temperature = temperature
        self._pipe = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        self._domain_labels = DOMAINS

    def route(self, text: str) -> np.ndarray:
        """Return a probability vector over DOMAINS (shape (4,), sums to 1).

        Uses keyword override first; falls back to BART-MNLI + temperature scaling.
        """
        # Keyword fast-path: hard-set one domain to 0.7, others 0.1 each
        override_idx = _keyword_override(text)
        if override_idx is not None:
            weights = np.full(len(DOMAINS), 0.1)
            weights[override_idx] = 0.7
            return weights

        result = self._pipe(text, self._domain_labels, multi_label=False)
        # result["labels"] may be reordered; re-align to DOMAINS order
        label_to_score = dict(zip(result["labels"], result["scores"]))
        raw_scores = np.array([label_to_score[d] for d in DOMAINS])
        return _softmax_temperature(raw_scores, self._temperature)

    def route_batch(self, texts: list[str]) -> np.ndarray:
        """Route a list of texts; returns array of shape (N, 4)."""
        return np.stack([self.route(t) for t in texts])
