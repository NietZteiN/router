"""Selectivity probe — blocktc's make-or-break localization gate.

Port of sepmlp_tofu/measure_selectivity.py to the single-bottleneck block
transcoder. The design's central claim is that detector-initialized author
blocks + generic-batch suppression produce SELF-ROUTING: author k's block
fires on author-k text and stays silent elsewhere, with no router and no
source ids at serving time. This script measures exactly that and applies the
pre-registered house gate (measure_key_firing.py precedent, where LoRA scored
1.11 = LAZY):

    A[k, j]  = mean act mass of block j per author-k QUESTION token
               (leakage matrix, DESIGN §8; act mass = sum of the block's
               non-negative ReLU activations, tc_layer.block_act_mass)
    on[k]    = A[k, k]          off[k] = count-weighted mean of block k's
                                          mass on OTHER authors' tokens
    verdict  = LAZY if median_k on/off < 2.0, SELECTIVE if >= 5.0, else
               INTERMEDIATE.

The shared block is the LAST column of A: reported separately and EXCLUDED
from the gate and from every off-block maximum (it is undeletable and is
SUPPOSED to fire everywhere). off_max[k] = the worst foreign AUTHOR block's
mean mass on author-k tokens.

Probe rows are full OU chat-template QA pairs (question + real answer via
data_tofu.preprocess_chat_instance / TofuQADataset) — the training
distribution, and the same rows the recall probe scores. The leakage matrix
pools QUESTION tokens (labels IGNORE and attended — the exact detector-init /
trainer mask): routing must switch ON while the block reads the question so
its decoders can shape the answer, and the detector init points there.
Deliberate divergence from sepmlp's standard probe, which pooled all attended
tokens — here DESIGN §8 pins question tokens; the leak probe below pools the
ANSWER span (what is served), matching sepmlp's leak contract. Question
choice per author is drawn from ONE RandomState iterated over all 200 authors
in a fixed order (draws happen even for non-resident authors), so the probe
set is invariant to any droplist.

Every probe forward is the PLAIN SERVING FORWARD — source-id-free, no
routing (TcState.source_ids is never set; item/batch source_ids ride along as
grouping metadata only). Rejected: the trainer-style begin_telemetry own/off
channel — it requires source_ids in TcState (a routed condition, not the
serving one) and collapses the per-block vector to two scalars, while the
leakage matrix needs all K+1 masses per row. begin_row_stats gives the full
[B, K+1] capture on the untouched serving path, so ONE mechanism serves the
matrix, the OOD rows, and the leak probe.

OOD sets (TOFU world_facts, TOFU real_authors, Alpaca via the read-only
tofu_sisa_lora skill_data.load_alpaca import, drawn BEYOND the training
head): every author block should be silent on them; ood_over_own > 0.1 is
the pre-registered trigger for revisiting the suppression recipe (sepmlp's
hinge/Gram fallback, DESIGN §2). real_authors is a TRAINED pool source
(phase-0 LM + phase-1 suppression) — its rows measure trained suppression,
not generalization; world_facts and Alpaca-beyond-head are the clean OOD
rows.

--recall_probe (gates G2/G3): per resident author, teacher-forced answer
probability on their 20 QA rows — the OU formula exp(-avg CE over answer
tokens) (open-unlearning evals/metrics/utils.py evaluate_probability,
reimplemented inline; OU is not importable from test-env) — under all-active
vs own-only serving (tc.active mask, proven ≡ physical removal by the CPU
deletion gates; the shared block stays live, so own-only == "every OTHER
author deleted", exactly the deletion counterfactual). |gap| > 0.05 is the
G3 tripwire: negative gap = the other blocks HURT recall (memsinks
interference); positive gap = they CARRY it (deleting them would break the
survivor). Either direction blocks P4 eval spend.

--probe forget_leak: the per-QUERY leak probe. Probe groups: forget10
ORIGINAL questions (text-join author mapping), forget10 PARAPHRASED questions
(forget10_perturbed, joined back on the original (question, answer) text), a
RandomState(seed) retain sample (default 400 rows), and the standard OOD
sets. For every query it records the mean-over-answer-span per-block act
mass, maxed over SURVIVING author blocks -> <out>.leak.npz per the
pre-registered LEAK-PROBE NPZ CONTRACT (keys name-compatible with sepmlp's:
max_surv_norm / max_foreign_norm / top_surv_author / own_norm / group /
author_of_q + scalars n_surviving, droplist_tag, K; blocktc extension
shared_norm). Usable with and without --droplist — the without-run is the
analyzer's no-droplist reference npz.

CLI (GPU job; CPU works for micro checkpoints):
  python measure_selectivity.py --config configs/blocktc_1b_k200.json \
    --out <run>/selectivity_k200.json [--checkpoint DIR] [--droplist F] \
    [--recall_probe] [--probe standard|forget_leak] [--device cuda]
Outputs: --out JSON (summary + per_author + provenance) and a sidecar
*_norms.npz with the [K, K+1] leakage matrix (standard) or <out>.leak.npz
(forget_leak).
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

from tc_common import (
    HF_HOME,
    IGNORE_INDEX,
    NO_AUTHOR,
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
    STORAGE_ROOT,
    alpaca_probe_head,
    author_of_row,
    file_sha256,
    import_memadapt_data,
    load_config,
    save_json,
    set_determinism,
    slurm_job_id,
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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pre-registered gate thresholds — shared with the sepmlp/measure_key_firing
# house gate so the blocktc number lands directly next to the LoRA anchor
# (1.11, 100% LAZY) and the sepmlp bars.
LAZY_LT = 2.0
SELECTIVE_GE = 5.0
# G3 tripwire (DESIGN §8): |all-active minus own-only answer prob| <= 0.05.
G3_GAP_MAX = 0.05
OOD_SETS = ("world_facts", "real_authors", "alpaca")
TOFU_SISA_DIR = os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora"))

# OOD-Alpaca probe rows must be provably never-seen. Training draws TWO windows
# of the seed-42 shuffle: phase 0 over-draws [0, 3*alpaca_n) and phase 1
# over-draws [ALPACA_TRAIN_HEAD, ALPACA_TRAIN_HEAD + 3*alpaca_n) (= [8000,
# 14000) at alpaca_n=2000). The probe therefore starts at
# tc_common.alpaca_probe_head(alpaca_n) = ALPACA_TRAIN_HEAD + 3*alpaca_n, BEYOND
# both — the head arithmetic is shared with train_tc.py so the probe window can
# never drift back into the phase-1 suppression rows (which would deflate the
# off/OOD mass and bias the leakage gate toward SELECTIVE). Earlier this skipped
# a bare ALPACA_TRAIN_HEAD (the sepmlp value, correct only because sepmlp trained
# just [0, 3*alpaca_n)); blocktc's phase-1 pool moved the window up.

# Leak-probe group labels — fixed by the shared NPZ contract; the analyzer
# switches on these exact strings.
LEAK_GROUPS = ("forget_orig", "forget_para", "retain",
               "ood_world_facts", "ood_real_authors", "ood_alpaca")


def default_run_dir(cfg: dict) -> str:
    """DESIGN §0: artifacts live at STORAGE_ROOT/runs/<run_name>. The driver
    always passes --checkpoint explicitly; this fallback covers interactive
    use (blocktc configs carry run_name, not sepmlp's output_dir key)."""
    assert cfg.get("run_name"), (
        "config carries no run_name — pass --checkpoint explicitly"
    )
    return os.path.join(STORAGE_ROOT, "runs", cfg["run_name"])


# ---------------------------------------------------------------------------
# Token masks (which tokens a capture pools)
# ---------------------------------------------------------------------------

def question_token_mask(batch) -> torch.Tensor:
    """(B, T) bool: prompt/question tokens = labels IGNORE and attended — the
    exact mask detector-init and the trainer derive. Padding is IGNORE too
    but attention-masked, so it never enters the pool."""
    return (batch["labels"] == IGNORE_INDEX) & batch["attention_mask"].bool()


def answer_token_mask(batch) -> torch.Tensor:
    """(B, T) bool: answer-span tokens = labels set and attended. The
    collator's ne(pad) quirk masks the real trailing eos — kept (OU's served
    behavior), so the leak probe pools exactly the served answer span (the
    sepmlp leak convention)."""
    return (batch["labels"] != IGNORE_INDEX) & batch["attention_mask"].bool()


# ---------------------------------------------------------------------------
# Aggregation (model-free; unit-tested against a dense recomputation in
# tests/test_selectivity.py)
# ---------------------------------------------------------------------------

def aggregate_author_mass(row_sums, row_cnts, row_authors, resident_ids):
    """Group the per-row capture by PROBE author -> raw [K, K+1] act-mass
    sums (columns = resident blocks in slot order, shared LAST) + [K] token
    counts. Count-weighted, so A[k, j] = sums[k, j] / tok[k] is the mean act
    mass of block j per author-k question token (DESIGN §8) — not a
    row-mean-of-row-means."""
    rows = np.asarray(row_sums, dtype=np.float64)
    cnt = np.asarray(row_cnts, dtype=np.float64)
    K = len(resident_ids)
    assert rows.ndim == 2 and rows.shape[1] == K + 1, rows.shape
    assert cnt.shape == (rows.shape[0],) and len(row_authors) == rows.shape[0]
    slot = {int(a): k for k, a in enumerate(resident_ids)}
    sums = np.zeros((K, K + 1))
    tok = np.zeros(K)
    for i, a in enumerate(row_authors):
        k = slot[int(a)]
        sums[k] += rows[i]
        tok[k] += cnt[i]
    assert (tok > 0).all(), "a probe author captured zero question tokens"
    return sums, tok


def leakage_summary(mass_sums, tok_counts) -> dict:
    """The leakage matrix and the gate inputs, from the raw sums:
      A        [K, K+1] mean act mass (shared column LAST);
      on[k]    = A[k, k];
      off[k]   = count-weighted mean of block k's mass on the OTHER authors'
                 tokens (column aggregate — same semantics as sepmlp's off
                 channel: block k heard on foreign text);
      off_max[k] = worst foreign AUTHOR block on author-k tokens (row max,
                 diagonal and shared column EXCLUDED per DESIGN §8);
      shared[k] = the shared block's mean mass on author-k tokens.
    """
    sums = np.asarray(mass_sums, dtype=np.float64)
    tok = np.asarray(tok_counts, dtype=np.float64)
    K = sums.shape[0]
    assert sums.shape == (K, K + 1) and tok.shape == (K,), (sums.shape,
                                                           tok.shape)
    A = sums / tok[:, None]
    author = A[:, :K]
    on = np.diagonal(author).copy()
    col_sums = sums[:, :K].sum(axis=0)
    diag_sums = np.diagonal(sums[:, :K]).copy()
    off_tok = tok.sum() - tok
    off = np.divide(col_sums - diag_sums, off_tok,
                    out=np.zeros(K), where=off_tok > 0)
    if K > 1:
        off_diag = author.copy()
        np.fill_diagonal(off_diag, -np.inf)
        off_max = off_diag.max(axis=1)
        top_off_slot = off_diag.argmax(axis=1)
    else:  # a single resident block has no foreign blocks
        off_max = np.zeros(K)
        top_off_slot = np.full(K, -1, dtype=np.int64)
    ratio = on / np.clip(off, 1e-12, None)
    return {"A": A, "on": on, "off": off, "ratio": ratio,
            "off_max": off_max, "top_off_slot": top_off_slot,
            "shared": A[:, K].copy()}


def pooled_mass(row_sums, row_cnts) -> np.ndarray:
    """[K+1] count-weighted mean act mass over ALL captured rows — one number
    per block for an OOD set (its rows belong to no author, so there is no
    on/off split; per-block silence is the whole question)."""
    s = np.asarray(row_sums, dtype=np.float64).sum(axis=0)
    c = float(np.asarray(row_cnts, dtype=np.float64).sum())
    return s / max(c, 1.0)


def gate_verdict(ratios) -> dict:
    """The pre-registered LAZY/SELECTIVE gate on the per-author ratio array."""
    arr = np.asarray(ratios, dtype=np.float64)
    median = float(np.median(arr))
    verdict = ("LAZY" if median < LAZY_LT
               else "SELECTIVE" if median >= SELECTIVE_GE
               else "INTERMEDIATE")
    return {
        "gate_metric": ("median per-author on/off ratio of mean "
                        "question-token block act mass"),
        "gate_thresholds": {"lazy_lt": LAZY_LT, "selective_ge": SELECTIVE_GE},
        "gate_median": median,
        "gate_verdict": verdict,
        "frac_ratio_lt_2": float((arr < LAZY_LT).mean()),
        "frac_ratio_ge_5": float((arr >= SELECTIVE_GE).mean()),
        "ratio_q25": float(np.percentile(arr, 25)),
        "ratio_q75": float(np.percentile(arr, 75)),
    }


def assemble_leak_arrays(mean_mass, resident_ids, author_of_q, groups,
                         droplist_tag, total_authors):
    """Build the LEAK-PROBE NPZ CONTRACT arrays from the [n_q, K_surv+1]
    per-query mean act-mass matrix (shared column LAST).

    Keys stay name-compatible with sepmlp's contract (max_surv_norm /
    max_foreign_norm / top_surv_author / own_norm / group / author_of_q +
    scalars n_surviving, droplist_tag, K) so the analyzer ports unchanged —
    but the values are mean per-token block ACT MASS, not decoder-output
    norms (one bottleneck -> one natural per-block scalar). blocktc
    extension: shared_norm carries the always-surviving shared block's
    column, reported separately and EXCLUDED from both maxima (DESIGN §8).
    own_norm is NaN when the query author's block is dropped/absent (orphans
    post-drop, all OOD); max_foreign_norm excludes the own block when it
    survives (retain off-block quiet level) and equals max_surv_norm
    otherwise. K_surv == 0 (all200 droplist) or an own-only single block
    yield 0.0 maxima (silence), top_surv_author -1."""
    m = np.asarray(mean_mass, dtype=np.float64)
    n_q = m.shape[0]
    rid = np.asarray(resident_ids, dtype=np.int64)
    K = int(rid.size)
    assert m.shape == (n_q, K + 1), (m.shape, K)
    aq = np.asarray(author_of_q, dtype=np.int64)
    groups = list(groups)
    assert aq.shape == (n_q,) and len(groups) == n_q
    assert set(groups) <= set(LEAK_GROUPS), sorted(set(groups))
    shared = m[:, K].copy()
    m = m[:, :K]
    if K == 0:
        max_surv = np.zeros(n_q)
        max_foreign = np.zeros(n_q)
        top = np.full(n_q, -1, dtype=np.int64)
        own = np.full(n_q, np.nan)
    else:
        max_surv = m.max(axis=1)
        top = rid[m.argmax(axis=1)]
        col = {int(a): j for j, a in enumerate(rid)}
        own = np.full(n_q, np.nan)
        max_foreign = max_surv.copy()
        for i in range(n_q):
            j = col.get(int(aq[i]))
            if j is None:
                continue
            own[i] = m[i, j]
            others = np.delete(m[i], j)
            max_foreign[i] = others.max() if others.size else 0.0
    return {
        "max_surv_norm": max_surv.astype(np.float32),
        "max_foreign_norm": max_foreign.astype(np.float32),
        "top_surv_author": top.astype(np.int32),
        "own_norm": own.astype(np.float32),
        "shared_norm": shared.astype(np.float32),
        "group": np.array(groups),
        "author_of_q": aq.astype(np.int32),
        "n_surviving": int(K),
        "droplist_tag": str(droplist_tag),
        "K": int(total_authors),
    }


# ---------------------------------------------------------------------------
# Probe-set construction
# ---------------------------------------------------------------------------

def build_ood_items(rs, args, tokenizer, data_tofu, max_length, probe_head):
    """The OOD probe rows (world_facts, real_authors, alpaca), consumed from
    the CALLER's RandomState in OOD_SETS order — factored out so the standard
    and leak probes share one construction (and one determinism pattern).

    probe_head = tc_common.alpaca_probe_head(cfg["alpaca_n"]): the first Alpaca
    shuffle row beyond BOTH training windows, so the ood_alpaca rows are
    never-seen text (see the module-level constant block)."""

    def _ood_item(pos, question, answer):
        item = data_tofu.preprocess_chat_instance(tokenizer, question, answer)
        item["index"] = -1 - pos  # never a real TOFU row index
        item["source_ids"] = NO_AUTHOR
        return item

    import datasets  # lazy: keeps module import light for the CPU tests

    ood_items = {}
    for split in ("world_facts", "real_authors"):
        rows = datasets.load_dataset("locuslab/TOFU", name=split, split="train")
        idx = rs.choice(len(rows), size=min(args.ood_n, len(rows)),
                        replace=False)
        ood_items[split] = [
            _ood_item(p, rows[i]["question"], rows[i]["answer"])
            for p, i in enumerate(sorted(int(j) for j in idx))
        ]
    # Alpaca via the same read-only helper the training pools use, so the
    # beyond-window slice is measured against the exact training shuffle.
    if TOFU_SISA_DIR not in sys.path:
        sys.path.insert(0, TOFU_SISA_DIR)
    from skill_data import load_alpaca

    pairs = load_alpaca(probe_head + args.ood_n, os.environ["HF_HOME"],
                        seed=args.seed)[probe_head:]
    ood_items["alpaca"] = [
        _ood_item(1000 + p, pair["question"], pair["answer"])
        for p, pair in enumerate(pairs)
    ]
    # Alpaca rows can be arbitrarily long; drop over-length items (the TOFU
    # dataset asserts < max_length, so this keeps one memory contract).
    for name in ood_items:
        kept = [it for it in ood_items[name]
                if len(it["input_ids"]) < max_length]
        if len(kept) != len(ood_items[name]):
            print(f"[selectivity] {name}: dropped "
                  f"{len(ood_items[name]) - len(kept)} rows >= "
                  f"{max_length} tokens")
        ood_items[name] = kept
    return ood_items


def build_probe_sets(args, tokenizer, dataset, resident_ids, data_tofu,
                     max_length, probe_head):
    """(author_items, {ood_set: items}). All items are OU chat-template
    instances with index + source_ids, ready for QACollatorWithSources
    (source_ids are grouping METADATA here — no probe forward ever routes).

    Determinism contract: ONE RandomState(seed) consumed in a fixed order —
    all 200 author draws first (drawn even for dropped/non-subset authors, so
    a droplist never shifts the surviving authors' question choice), then the
    OOD splits in OOD_SETS order (the sepmlp/measure_key_firing pattern)."""
    rs = np.random.RandomState(args.seed)
    resident = set(int(a) for a in resident_ids)
    author_items = []
    for a in range(NUM_AUTHORS):
        n = min(args.questions_per_author, RECORDS_PER_AUTHOR)
        idx = rs.choice(RECORDS_PER_AUTHOR, size=n, replace=False)
        if a not in resident:
            continue
        for i in sorted(int(j) for j in idx):
            item = dataset[a * RECORDS_PER_AUTHOR + i]
            assert int(item["source_ids"]) == a, "full-split row/author drift"
            author_items.append(item)

    ood_items = build_ood_items(rs, args, tokenizer, data_tofu, max_length,
                                probe_head)
    return author_items, ood_items


def build_leak_probe_items(args, tokenizer, dataset, data_tofu, max_length,
                           probe_head):
    """Leak-probe rows as (group, author_of_q, item) triples.

    - forget_orig: the forget10 authors' 20 original QA rows each, authors
      resolved by the text-join mapping (verify_forget_author_mapping) —
      never positional.
    - forget_para: forget10_perturbed paraphrased_question/paraphrased_answer,
      joined back to authors on the ORIGINAL (question, answer) text.
    - retain: RandomState(args.seed) sample of args.leak_retain_n non-forget
      full rows.
    - ood_*: the standard OOD sets, author_of_q = -1.
    Determinism: ONE RandomState(args.seed), consumed retain-draw first, then
    the OOD draws in OOD_SETS order."""
    forget_authors = data_tofu.verify_forget_author_mapping("forget10")
    fset = set(forget_authors)
    rs = np.random.RandomState(args.seed)
    triples = []
    for a in forget_authors:
        for i in range(RECORDS_PER_AUTHOR):
            item = dataset[a * RECORDS_PER_AUTHOR + i]
            assert int(item["source_ids"]) == a, "full-split row/author drift"
            triples.append(("forget_orig", a, item))

    import datasets

    pert = datasets.load_dataset("locuslab/TOFU", name="forget10_perturbed",
                                 split="train")
    full = dataset.data
    key_to_author = {
        (q, ans): author_of_row(i)
        for i, (q, ans) in enumerate(zip(full["question"], full["answer"]))
    }
    n_para_long = 0
    for p in range(len(pert)):
        row = pert[p]
        key = (row["question"], row["answer"])
        assert key in key_to_author, (
            f"forget10_perturbed row {p} not joinable to the full split: "
            f"{row['question'][:60]!r}"
        )
        a = key_to_author[key]
        assert a in fset, f"perturbed row {p} joins to non-forget author {a}"
        item = data_tofu.preprocess_chat_instance(
            tokenizer, row["paraphrased_question"], row["paraphrased_answer"])
        if len(item["input_ids"]) >= max_length:
            n_para_long += 1
            continue
        item["index"] = -20000 - p
        item["source_ids"] = NO_AUTHOR  # leak capture is source-id-free
        triples.append(("forget_para", a, item))
    if n_para_long:
        print(f"[leak] forget_para: dropped {n_para_long} rows >= "
              f"{max_length} tokens")

    retain_rows = [i for i in range(len(dataset))
                   if author_of_row(i) not in fset]
    idx = rs.choice(len(retain_rows),
                    size=min(args.leak_retain_n, len(retain_rows)),
                    replace=False)
    for j in sorted(int(v) for v in idx):
        r = retain_rows[j]
        triples.append(("retain", author_of_row(r), dataset[r]))

    ood_items = build_ood_items(rs, args, tokenizer, data_tofu, max_length,
                                probe_head)
    for name in OOD_SETS:
        for it in ood_items[name]:
            triples.append((f"ood_{name}", -1, it))
    assert set(g for g, _, _ in triples) <= set(LEAK_GROUPS)
    return triples


def iter_batches(items, collator, batch_size):
    for start in range(0, len(items), batch_size):
        yield collator(items[start:start + batch_size])


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_mass_capture(model, state, batches, token_mask_of, device):
    """Per-ROW per-block act-mass capture on the PLAIN serving forward
    (source-id-free — TcState.source_ids stays None; see module docstring for
    the rejected telemetry-channel alternative). One begin_row_stats capture
    per batch; the single read site emits exactly one stats entry per
    forward. Returns (sums [n_rows, K+1] float64, cnts [n_rows] float64):
    per-row act-mass sums over the mask's tokens, shared block LAST."""
    sums, cnts = [], []
    for batch in batches:
        mask = token_mask_of(batch)
        assert bool(mask.any(dim=1).all()), (
            "a probe row has zero pooled tokens"
        )
        state.clear()
        state.begin_row_stats(mask.to(device))
        try:
            model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device),
                  use_cache=False)
            stats = state.end_row_stats()
        finally:
            state.clear()
        assert len(stats) == 1, (
            f"{len(stats)} row-stats entries from one forward — the encoder "
            "ran more than once (or never); single-read-site contract broken"
        )
        sums.append(stats[0]["sum"].double().numpy())
        cnts.append(stats[0]["cnt"].double().numpy())
    return np.concatenate(sums, axis=0), np.concatenate(cnts, axis=0)


@torch.no_grad()
def answer_probability(model, batch, device):
    """OU evaluate_probability, inline: per-row exp(-avg CE over answer
    tokens), num_token_gt counted on the UNSHIFTED labels (OU's exact
    denominator). Logits cast to fp32 first — parity with the deliberate
    fp32-logits fix in the OU eval tree."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False)
    logits = out.logits.float()[..., :-1, :].contiguous()
    shifted = labels[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    losses = loss_fn(logits.transpose(-1, -2), shifted).sum(dim=-1)
    num_tok = (labels != -100).sum(-1)
    probs = torch.exp(-losses / num_tok)
    return probs.cpu().tolist()


@contextlib.contextmanager
def own_only(tc, author_id: int):
    """Temporarily serve ONLY the given author's block via the active mask
    (proven ≡ physical removal by the deletion-identity CPU gates). The
    shared block is undeletable and stays live through active_feature_mask —
    own-only == "every OTHER author deleted", exactly the deletion
    counterfactual. Restores the exact prior mask even on exception."""
    saved = tc.active.clone()
    try:
        mask = tc.author_ids == author_id
        assert bool(mask.any()), f"author {author_id} not resident"
        tc.active.copy_(mask)
        yield
    finally:
        tc.active.copy_(saved)


def run_recall_probe(model, tc, state, dataset, resident_ids, collator,
                     device, batch_size):
    """Teacher-forced own-author answer probability under the two serving
    conditions. The serving path is router-free, so the state stays fully
    cleared (no source ids) — exactly the inference condition."""
    state.clear()
    assert state.source_ids is None
    per_author = []
    for a in resident_ids:
        items = [dataset[int(a) * RECORDS_PER_AUTHOR + i]
                 for i in range(RECORDS_PER_AUTHOR)]
        probs_all, probs_own = [], []
        for batch in iter_batches(items, collator, batch_size):
            probs_all += answer_probability(model, batch, device)
        with own_only(tc, int(a)):
            for batch in iter_batches(items, collator, batch_size):
                probs_own += answer_probability(model, batch, device)
        m_all = float(np.mean(probs_all))
        m_own = float(np.mean(probs_own))
        per_author.append({
            "author": int(a),
            "prob_all_active": m_all,
            "prob_own_only": m_own,
            "gap_all_minus_own": m_all - m_own,
        })
    gaps = [r["gap_all_minus_own"] for r in per_author]
    gap_mean = float(np.mean(gaps))
    return {
        "rows_per_author": RECORDS_PER_AUTHOR,
        "prob_all_active_mean": float(np.mean(
            [r["prob_all_active"] for r in per_author])),
        "prob_own_only_mean": float(np.mean(
            [r["prob_own_only"] for r in per_author])),
        "gap_all_minus_own_mean": gap_mean,
        # G3 tripwire (DESIGN §8): |gap| > 0.05 means serving quality depends
        # on OTHER authors' blocks — negative = they hurt recall (memsinks
        # interference), positive = they carry it (deleting them would break
        # the survivor). Either direction blocks P4 eval spend.
        "gap_threshold": G3_GAP_MAX,
        "gap_abs_mean": abs(gap_mean),
        "frac_authors_abs_gap_gt_threshold": float(np.mean(
            [abs(g) > G3_GAP_MAX for g in gaps])),
        "gap_trip": bool(abs(gap_mean) > G3_GAP_MAX),
        "per_author": per_author,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="run dir with blocktc.pt "
                         "(default: STORAGE_ROOT/runs/<config run_name>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--droplist", default=None,
                    help="droplists/<tag>.json applied before probing")
    ap.add_argument("--questions_per_author", type=int, default=5)
    ap.add_argument("--ood_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recall_probe", action="store_true")
    ap.add_argument("--probe", choices=("standard", "forget_leak"),
                    default="standard",
                    help="forget_leak = per-query leak probe -> <out>.leak.npz")
    ap.add_argument("--leak_retain_n", type=int, default=400,
                    help="retain sample size for --probe forget_leak")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    assert not (args.probe == "forget_leak" and args.recall_probe), (
        "--recall_probe is a standard-probe option"
    )

    cfg = load_config(args.config)
    # blocktc configs carry no hf_home key (DESIGN §7) — the constant lives
    # in tc_common; setdefault so a SLURM-exported HF_HOME wins.
    os.environ.setdefault("HF_HOME", HF_HOME)
    set_determinism(args.seed)

    # Heavy imports after HF_HOME is set (storage-partition contract:
    # transformers freezes its cache paths from the env at import time).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tc_model import (
        apply_droplist_file,
        compute_tc_sha,
        install_tc,
        load_tc_from_checkpoint,
    )

    run_dir = args.checkpoint or default_run_dir(cfg)
    tc, adapter_cfg, state, ckpt_phase = load_tc_from_checkpoint(run_dir)
    checkpoint_tc_sha = compute_tc_sha(tc)
    n_checkpoint_authors = tc.num_authors  # pre-drop K
    droplist_spec = None
    if args.droplist:
        # Order mirrors BlockTcLlamaForCausalLM: physical removal first, then
        # cast + install — the probe sees exactly the served post-drop blocks.
        droplist_spec = apply_droplist_file(tc, args.droplist)
        print(f"[selectivity] droplist {droplist_spec['tag']}: dropped "
              f"{droplist_spec['_dropped']} author blocks in "
              f"{droplist_spec['_apply_seconds']:.4f}s")

    # bf16+sdpa on GPU (training/serving parity; encode/decode still compute
    # in their fp32 islands); fp32 on CPU (bf16 CPU matmuls are slow and the
    # CPU path is only for micro checkpoints).
    on_cuda = args.device.startswith("cuda")
    dtype = torch.bfloat16 if on_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=dtype, attn_implementation="sdpa"
    ).to(args.device).eval()
    tc.to(device=args.device, dtype=dtype)
    install_tc(model, tc, state)
    # After install: the fresh transcoder/wrapper modules default to train
    # mode, whose guard (rightly) refuses source-id-free forwards — every
    # probe here IS a source-id-free serving forward, so eval the whole tree.
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    data_tofu = import_memadapt_data()
    max_length = int(cfg.get("max_length", 512))
    # OOD-Alpaca probe start = first shuffle row beyond BOTH training windows.
    # Derived from THIS config's alpaca_n so it can never drift from the draw
    # train_tc.py made with the same value (see the module constant block).
    assert "alpaca_n" in cfg, (
        "config lacks alpaca_n — the OOD-Alpaca probe head cannot be matched "
        "to the training draw"
    )
    probe_head = alpaca_probe_head(int(cfg["alpaca_n"]))
    dataset = data_tofu.TofuQADataset(tokenizer, split="full",
                                      max_length=max_length)
    collator = data_tofu.QACollatorWithSources(tokenizer)

    resident_ids = [int(a) for a in tc.author_ids.tolist()]
    K = len(resident_ids)
    print(f"[selectivity] {K} resident author blocks, read layer "
          f"{tc.insert_layer}, span {tc.span}, device {args.device}")

    provenance = {
        "config": {
            **{k: v for k, v in vars(args).items()},
            "config_path": cfg["_config_path"],
            "model": cfg["model"],
            "alpaca_n": int(cfg["alpaca_n"]),
            "ood_alpaca_probe_head": int(probe_head),
            "adapter_cfg": adapter_cfg,
        },
        "checkpoint": os.path.abspath(run_dir),
        "checkpoint_phase": ckpt_phase,
        "droplist": os.path.abspath(args.droplist) if args.droplist else None,
        "droplist_tag": droplist_spec["tag"] if droplist_spec else None,
        "checkpoint_tc_sha": checkpoint_tc_sha,
        "served_tc_sha": compute_tc_sha(tc),
        "n_surviving": K,
        "n_checkpoint_authors": int(n_checkpoint_authors),
        "insert_layer": int(tc.insert_layer),
        "span": int(tc.span),
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "torch_version": torch.__version__,
    }

    if args.probe == "forget_leak":
        # Per-query leak probe. Droplist (if any) was applied ABOVE, before
        # install — the probe sees the post-drop blocks.
        triples = build_leak_probe_items(args, tokenizer, dataset, data_tofu,
                                         max_length, probe_head)
        groups = [g for g, _, _ in triples]
        authors_q = [a for _, a, _ in triples]
        items = [it for _, _, it in triples]
        counts = {g: groups.count(g) for g in LEAK_GROUPS}
        print(f"[leak] {len(items)} query rows: {counts}")
        sums, cnts = run_mass_capture(
            model, state, iter_batches(items, collator, args.batch_size),
            answer_token_mask, args.device)
        arrays = assemble_leak_arrays(
            sums / cnts[:, None], resident_ids, authors_q, groups,
            droplist_spec["tag"] if droplist_spec else "none",
            n_checkpoint_authors,
        )
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        npz_path = out_path + ".leak.npz"
        np.savez_compressed(npz_path, **arrays)
        g_arr = np.array(groups)
        per_group = {
            g: {
                "n": int(counts[g]),
                "max_surv_norm_mean": float(
                    arrays["max_surv_norm"][g_arr == g].mean()),
                "max_surv_norm_p90": float(np.percentile(
                    arrays["max_surv_norm"][g_arr == g], 90)),
            }
            for g in LEAK_GROUPS if counts[g]
        }
        result = {
            "probe": "forget_leak",
            **provenance,
            "n_queries": len(items),
            "group_counts": counts,
            "per_group_summary": per_group,
            "leak_npz": npz_path,
        }
        save_json(result, out_path)
        print(f"[leak] wrote {out_path} (+ {npz_path})")
        return

    # ---- standard probe: leakage matrix + gate (+ optional recall) --------
    assert K > 0, (
        "no resident author blocks — the standard probe needs survivors "
        "(the all-dropped condition is a leak/eval-only case)"
    )
    author_items, ood_items = build_probe_sets(
        args, tokenizer, dataset, resident_ids, data_tofu, max_length,
        probe_head)
    row_authors = [int(it["source_ids"]) for it in author_items]
    n_prompts = len(author_items) + sum(len(v) for v in ood_items.values())
    print(f"[selectivity] {len(author_items)} author rows + "
          f"{ {k: len(v) for k, v in ood_items.items()} } OOD rows")

    sums, cnts = run_mass_capture(
        model, state, iter_batches(author_items, collator, args.batch_size),
        question_token_mask, args.device)
    mass_sums, tok_counts = aggregate_author_mass(sums, cnts, row_authors,
                                                  resident_ids)
    stats = leakage_summary(mass_sums, tok_counts)

    ood_vecs = {}
    for name, items in ood_items.items():
        s, c = run_mass_capture(
            model, state, iter_batches(items, collator, args.batch_size),
            question_token_mask, args.device)
        ood_vecs[name] = pooled_mass(s, c)

    summary = gate_verdict(stats["ratio"])
    on_mean = float(stats["on"].mean())
    summary["on_mass_mean"] = on_mean
    summary["off_mass_mean"] = float(stats["off"].mean())
    summary["off_max_median"] = float(np.median(stats["off_max"]))
    # The shared block SHOULD fire everywhere — reported, never gated.
    summary["shared_on_mean"] = float(stats["shared"].mean())
    summary["ood_mass_mean"] = {
        name: float(v[:K].mean()) for name, v in ood_vecs.items()
    }
    summary["ood_shared_mean"] = {
        name: float(v[K]) for name, v in ood_vecs.items()
    }
    # Pre-registered suppression-recipe trigger: any OOD/own firing > 0.1.
    summary["ood_over_own"] = {
        name: float(v / max(on_mean, 1e-12))
        for name, v in summary["ood_mass_mean"].items()
    }

    per_author = []
    for k, a in enumerate(resident_ids):
        per_author.append({
            "author": int(a),
            "on_mass": float(stats["on"][k]),
            "off_mass": float(stats["off"][k]),
            "ratio": float(stats["ratio"][k]),
            "off_max": float(stats["off_max"][k]),
            "top_off_block": (int(resident_ids[int(stats["top_off_slot"][k])])
                              if K > 1 else -1),
            "shared_mass": float(stats["shared"][k]),
            "ood_mass": {name: float(v[k]) for name, v in ood_vecs.items()},
        })

    recall = None
    if args.recall_probe:
        print("[selectivity] recall probe (all-active vs own-only)")
        recall = run_recall_probe(model, tc, state, dataset, resident_ids,
                                  collator, args.device, args.batch_size)
        print(f"[selectivity] recall: all-active "
              f"{recall['prob_all_active_mean']:.4f} vs own-only "
              f"{recall['prob_own_only_mean']:.4f} "
              f"(gap {recall['gap_all_minus_own_mean']:+.4f}; G3 tripwire "
              f"{'TRIPPED' if recall['gap_trip'] else 'clear'})")

    out_path = os.path.abspath(args.out)
    npz_path = os.path.splitext(out_path)[0] + "_norms.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Raw sums + counts ride along so any aggregate is exactly recomputable.
    np.savez_compressed(
        npz_path,
        author_ids=np.array(resident_ids),
        insert_layer=np.array(int(tc.insert_layer)),
        span=np.array(int(tc.span)),
        A=stats["A"],                    # [K, K+1] mean mass, shared LAST
        A_sums=mass_sums,
        token_counts=tok_counts,
        on=stats["on"], off=stats["off"], ratio=stats["ratio"],
        off_max=stats["off_max"], shared=stats["shared"],
        **{f"ood_{name}": v for name, v in ood_vecs.items()},
    )

    result = {
        **provenance,
        "n_authors": K,
        "n_prompts": int(n_prompts),
        "matrices_npz": npz_path,
        "summary": summary,
        "per_author": per_author,
        "recall_probe": recall,
    }
    save_json(result, out_path)
    print(f"[selectivity] gate: median on/off ratio = "
          f"{summary['gate_median']:.3f} -> {summary['gate_verdict']}")
    print(f"[selectivity] wrote {out_path} (+ {npz_path})")


if __name__ == "__main__":
    main()
