"""Synthetic user profiles for SEA evaluation (4 profiles from the paper)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

DOMAINS = ["security", "code", "data", "general"]


@dataclass
class UserProfile:
    user_id: str
    # Per-domain affinity weights (sum to 1.0); index matches DOMAINS
    domain_affinity: List[float]
    # Style traits used for style_trait_match metric
    style_traits: List[str]
    # 20 evaluation prompts (5 per domain); used in eval_sea.py
    query_pool: List[str]
    # (prompt, chosen_response, rejected_response) triples for CAA + DPO
    preference_pairs: List[Tuple[str, str, str]]


# ---------------------------------------------------------------------------
# Query pools — 5 prompts per domain × 4 domains = 20 per user
# ---------------------------------------------------------------------------

_SECURITY_QUERIES = [
    "What are the most common SQL injection attack patterns and how can I prevent them?",
    "Explain the difference between symmetric and asymmetric encryption with examples.",
    "How does a buffer overflow vulnerability work and what mitigations exist?",
    "What is CSRF and how do modern web frameworks defend against it?",
    "Describe the OWASP Top 10 vulnerabilities and give one mitigation per item.",
]

_CODE_QUERIES = [
    "Write a Python function that implements binary search on a sorted list.",
    "How do I implement a thread-safe singleton pattern in Java?",
    "Explain the difference between shallow copy and deep copy in Python.",
    "Write a recursive function to compute the nth Fibonacci number with memoization.",
    "How do I profile a Python script to find performance bottlenecks?",
]

_DATA_QUERIES = [
    "Write a SQL query to find the top 5 customers by total order value.",
    "How do I perform a left join versus an inner join in SQL?",
    "Write a pandas snippet to pivot a DataFrame from long to wide format.",
    "Explain the difference between OLAP and OLTP database designs.",
    "How do I write a window function in SQL to compute a running total?",
]

_GENERAL_QUERIES = [
    "What are the key differences between supervised and unsupervised learning?",
    "Explain the concept of transfer learning in natural language processing.",
    "What is the difference between precision and recall in evaluation metrics?",
    "How does gradient descent optimize a neural network's parameters?",
    "What are the pros and cons of microservices versus monolithic architectures?",
]

_ALL_QUERIES = _SECURITY_QUERIES + _CODE_QUERIES + _DATA_QUERIES + _GENERAL_QUERIES


# ---------------------------------------------------------------------------
# Preference pairs — (prompt, chosen, rejected)
# chosen: response matching user's style/domain; rejected: generic/off-tone
# ---------------------------------------------------------------------------

def _make_security_pairs() -> List[Tuple[str, str, str]]:
    return [
        (
            "How do I safely store passwords in a database?",
            "Use bcrypt or Argon2 with a per-user salt. Never store plaintext or MD5/SHA1 hashes. "
            "Argon2id is the current OWASP recommendation due to its memory-hardness. "
            "Parameterize the cost factor so you can raise it as hardware improves.",
            "Hash passwords before storing. Use a library for this.",
        ),
        (
            "What is a timing attack and how do I prevent it?",
            "A timing attack exploits measurable differences in execution time to infer secret values. "
            "For example, a naive string equality check short-circuits on the first mismatch, leaking "
            "the position of the first differing byte. Prevent it with constant-time comparison "
            "functions (e.g., hmac.compare_digest in Python). Always use them for MAC verification "
            "and credential checks.",
            "Timing attacks use timing information. Use secure comparison functions.",
        ),
        (
            "Explain TLS handshake and what can go wrong.",
            "The TLS 1.3 handshake completes in one round-trip: client sends ClientHello with "
            "supported cipher suites and a key share; server responds with its key share, certificate, "
            "and Finished. Common failure points: expired or self-signed certs, mismatched cipher "
            "suites, SNI misconfiguration, and missing OCSP stapling. Downgrade attacks are mitigated "
            "by the TLS_FALLBACK_SCSV mechanism.",
            "TLS encrypts traffic between client and server. It can fail if certs expire.",
        ),
        (
            "What is privilege escalation and how do attackers achieve it?",
            "Privilege escalation is the process of gaining elevated system rights beyond what was "
            "initially granted. Vertical escalation raises privileges (user → root); horizontal "
            "escalation accesses another user's resources. Common techniques: exploiting SUID binaries, "
            "sudo misconfigurations, kernel vulnerabilities, weak file permissions on cron jobs, and "
            "token impersonation on Windows. Mitigate with the principle of least privilege, regular "
            "patching, and auditd monitoring.",
            "Privilege escalation means getting more permissions. Attackers exploit bugs to do this.",
        ),
        (
            "How does certificate pinning improve mobile app security?",
            "Certificate pinning embeds a hash of the server's expected certificate or public key "
            "directly in the app binary, preventing MITM attacks even when a trusted CA is compromised. "
            "Implement with OkHttp's CertificatePinner or iOS's NSURLSession pinning. Pin the "
            "intermediate CA rather than the leaf certificate to survive cert rotation. Maintain a "
            "backup pin and a server header failsafe to avoid bricking the app on cert renewal.",
            "Certificate pinning verifies the server cert matches an expected value. It prevents MITM.",
        ),
    ]


def _make_code_pairs() -> List[Tuple[str, str, str]]:
    return [
        (
            "How do I avoid off-by-one errors in loop boundaries?",
            "Use half-open intervals [start, end) consistently — Python's range() and slice notation "
            "already do this. For sentinel loops, prefer while True: ... break over while condition. "
            "Name loop variables clearly (i vs idx vs position). Write the termination condition "
            "before the loop body, not after. Unit-test with zero-length, one-element, and "
            "two-element inputs.",
            "Be careful with loop boundaries. Test edge cases.",
        ),
        (
            "Explain Python's GIL and when it matters.",
            "The Global Interpreter Lock (GIL) ensures only one thread executes Python bytecode at a "
            "time, preventing data races on reference counts. It matters for CPU-bound workloads: "
            "threading won't parallelize them. Use multiprocessing or concurrent.futures.ProcessPoolExecutor "
            "for CPU-bound tasks. The GIL is released during I/O and C extension calls "
            "(e.g., numpy, PyTorch), so threading works fine for I/O-bound work.",
            "Python's GIL prevents true thread parallelism for CPU work. Use multiprocessing instead.",
        ),
        (
            "What is a decorator in Python and when should I use one?",
            "A decorator is a higher-order function that wraps another function to extend its behavior "
            "without modifying it. Use @functools.wraps to preserve the wrapped function's metadata. "
            "Good use cases: logging, timing, access control, caching (@functools.lru_cache), "
            "retry logic. Avoid decorators when the wrapping logic is only used once — a plain "
            "function call is clearer.",
            "A decorator wraps a function to add behavior. Examples: @property, @staticmethod.",
        ),
        (
            "How do I write idiomatic error handling in Python?",
            "Catch specific exceptions rather than bare except. Use else to signal success "
            "and finally for cleanup. Prefer EAFP (try/except) over LBYL (if/check) for "
            "performance-critical paths. Raise ValueError for bad user input, TypeError for type "
            "mismatches, and RuntimeError for impossible states. Log at the boundary where you "
            "have context; don't silently swallow exceptions.",
            "Use try/except to handle errors. Catch the right exception type.",
        ),
        (
            "Explain the difference between lists and generators in Python.",
            "A list materializes all elements in memory immediately; a generator is a lazy iterator "
            "that yields one element at a time. Generators are memory-efficient for large or infinite "
            "sequences. Use a generator expression (x for x in ...) when you only need to iterate "
            "once; use a list when you need random access, len(), or multiple passes. Note generators "
            "are exhausted after a single pass — call list() to reify if needed.",
            "Lists hold all elements in memory. Generators are lazy and memory-efficient.",
        ),
    ]


def _make_data_pairs() -> List[Tuple[str, str, str]]:
    return [
        (
            "How do I handle NULL values in SQL aggregations?",
            "NULL is ignored by aggregate functions (SUM, AVG, COUNT(*) counts rows, COUNT(col) "
            "excludes NULLs). Use COALESCE(col, 0) to substitute a default before aggregating. "
            "For GROUP BY, NULLs form their own group. In window functions, IGNORE NULLS is supported "
            "in some dialects (e.g., Oracle, BigQuery) for FIRST_VALUE/LAST_VALUE. Always document "
            "NULL semantics in column comments.",
            "NULLs are ignored by aggregates. Use COALESCE to handle them.",
        ),
        (
            "Explain database normalization forms.",
            "1NF: atomic values, no repeating groups. 2NF: 1NF + no partial dependencies on a "
            "composite key. 3NF: 2NF + no transitive dependencies. BCNF: every determinant is a "
            "candidate key. In practice, normalize to 3NF for OLTP (reduces anomalies) and "
            "denormalize strategically for OLAP (reduces joins). Always justify denormalization "
            "with measured query performance gains.",
            "Normalization removes redundancy. 1NF, 2NF, 3NF are common forms.",
        ),
        (
            "How do I write an efficient GROUP BY query?",
            "Push filters into WHERE (not HAVING) so the optimizer can use indexes before grouping. "
            "Index the GROUP BY columns and any join keys. Avoid SELECT * — name only the columns "
            "needed. Use HAVING only for aggregate conditions. On large tables, consider "
            "pre-aggregated summary tables (materialized views) refreshed incrementally.",
            "Use GROUP BY to aggregate. Put filters in WHERE not HAVING when possible.",
        ),
        (
            "What is the difference between RANK and DENSE_RANK?",
            "RANK() assigns the same rank to ties but skips subsequent numbers (1, 2, 2, 4). "
            "DENSE_RANK() assigns the same rank to ties without gaps (1, 2, 2, 3). "
            "ROW_NUMBER() assigns a unique sequential number regardless of ties. "
            "Use RANK for competition-style rankings, DENSE_RANK for percentile bucketing, "
            "and ROW_NUMBER for deduplication (paired with PARTITION BY).",
            "RANK skips numbers after ties; DENSE_RANK does not.",
        ),
        (
            "How do I profile a slow SQL query?",
            "Start with EXPLAIN (or EXPLAIN ANALYZE in PostgreSQL) to inspect the query plan. "
            "Look for Seq Scan on large tables, high rows estimates, nested loops on unbounded sets, "
            "and sort operations spilling to disk. Add indexes on filter and join columns, rewrite "
            "correlated subqueries as JOINs, and ensure statistics are up to date (ANALYZE). "
            "For recurring slow queries, use pg_stat_statements or slow query log.",
            "Use EXPLAIN to see the query plan. Add indexes to speed it up.",
        ),
    ]


def _make_general_pairs() -> List[Tuple[str, str, str]]:
    return [
        (
            "What is overfitting and how do I prevent it?",
            "Overfitting occurs when a model memorizes training noise instead of learning generalizable "
            "patterns, leading to high train accuracy but poor test accuracy. Mitigate with: more data, "
            "data augmentation, dropout, weight decay (L2 regularization), early stopping, cross-validation, "
            "and simpler model architectures. Monitor the train/validation loss gap; widen it as a "
            "signal to regularize more aggressively.",
            "Overfitting is when a model does well on training data but poorly on test data. Use "
            "regularization to prevent it.",
        ),
        (
            "Explain the attention mechanism in transformers.",
            "Attention computes a weighted sum of value vectors, where weights are derived from the "
            "similarity between a query vector and a set of key vectors: "
            "Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V. Multi-head attention runs h attention "
            "heads in parallel on linearly projected subspaces and concatenates the results, "
            "enabling the model to jointly attend to information from different representation "
            "subspaces. The √dₖ scaling prevents vanishingly small softmax gradients.",
            "Attention lets the model focus on relevant parts of the input. It uses queries, keys, "
            "and values.",
        ),
        (
            "What is the bias-variance tradeoff?",
            "Bias is error from incorrect assumptions in the learning algorithm (underfitting); "
            "variance is error from sensitivity to fluctuations in the training set (overfitting). "
            "They trade off: increasing model capacity reduces bias but increases variance. "
            "The expected test error decomposes as Bias² + Variance + Irreducible Noise. "
            "Ensemble methods (bagging, boosting) target different ends of this tradeoff: "
            "bagging reduces variance; boosting reduces bias.",
            "Bias-variance tradeoff balances underfitting and overfitting.",
        ),
        (
            "How does backpropagation work?",
            "Backpropagation applies the chain rule to compute gradients of the loss with respect "
            "to every parameter by propagating error signals from output to input. For each layer ℓ, "
            "δₗ = (Wₗ₊₁ᵀ δₗ₊₁) ⊙ σ'(zₗ). Gradients accumulate from all paths through the "
            "computation graph. The vanishing gradient problem arises when σ'(zₗ) ≈ 0 across many "
            "layers; ReLU and residual connections mitigate this.",
            "Backpropagation computes gradients using the chain rule to update model weights.",
        ),
        (
            "What are the differences between batch, mini-batch, and stochastic gradient descent?",
            "Batch GD computes the exact gradient over the full dataset — low variance but very "
            "expensive per step and prone to saddle points. Stochastic GD (SGD) uses one sample — "
            "noisy gradients can escape local minima but converge slowly. Mini-batch GD (the "
            "standard in practice) uses a small batch (32–512): GPU-efficient, enough noise to "
            "avoid sharp minima, and stable enough to converge. Larger batches often require "
            "learning rate warmup and linear scaling of the LR.",
            "Batch GD uses all data; SGD uses one sample; mini-batch uses a small subset.",
        ),
    ]


# ---------------------------------------------------------------------------
# User profile definitions (paper Table profiles)
# ---------------------------------------------------------------------------

USERS: dict[str, UserProfile] = {
    "security_expert": UserProfile(
        user_id="security_expert",
        domain_affinity=[0.7, 0.1, 0.1, 0.1],
        style_traits=["technical", "precise", "formal", "defense-focused"],
        query_pool=_ALL_QUERIES,
        preference_pairs=_make_security_pairs(),
    ),
    "casual_coder": UserProfile(
        user_id="casual_coder",
        domain_affinity=[0.1, 0.7, 0.1, 0.1],
        style_traits=["concise", "practical", "example-driven", "informal"],
        query_pool=_ALL_QUERIES,
        preference_pairs=_make_code_pairs(),
    ),
    "data_analyst": UserProfile(
        user_id="data_analyst",
        domain_affinity=[0.1, 0.1, 0.7, 0.1],
        style_traits=["analytical", "structured", "SQL-focused", "methodical"],
        query_pool=_ALL_QUERIES,
        preference_pairs=_make_data_pairs(),
    ),
    "general_user": UserProfile(
        user_id="general_user",
        domain_affinity=[0.25, 0.25, 0.25, 0.25],
        style_traits=["balanced", "educational", "clear"],
        query_pool=_ALL_QUERIES,
        preference_pairs=_make_general_pairs(),
    ),
}
