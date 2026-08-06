"""Output-selectivity probe — the method's make-or-break localization gate.

Vincent's central claim is that architecturally disconnected per-author MLPs +
in-domain negatives produce SELF-ROUTING: an author's bank fires on that
author's text and stays silent elsewhere, with no router. This script measures
exactly that, per resident author and per layer, and applies the pre-registered
house gate (measure_key_firing.py precedent, where LoRA scored 1.11 = LAZY):

    ratio_a = mean-token ||out_a|| on author a's probe rows (on-norm)
            / count-weighted mean-token ||out_a|| on OTHER authors' probe rows
    verdict = LAZY if median_a ratio_a < 2.0, SELECTIVE if >= 5.0, else
              INTERMEDIATE.

Probe rows are full OU chat-template QA pairs (question + real answer via
data_tofu.preprocess_chat_instance / TofuQADataset) — NOT question-only
prompts: the bank must fire on the answer tokens it serves at inference, this
matches the training distribution, and it keeps the norm capture and the
recall probe on identical rows. Question choice per author is drawn from ONE
RandomState iterated over all 200 authors in a fixed order (draws happen even
for non-resident authors), so the probe set is invariant to any droplist.

OOD sets (TOFU world_facts, TOFU real_authors, Alpaca via the read-only
tofu_sisa_lora skill_data.load_alpaca import) carry source_ids = NO_AUTHOR, so
every author's bank is "off" for them; ood_over_own > 0.1 is the
pre-registered trigger for the Alpaca-negatives training arm.

Norms come from the banks' own grouped-forward telemetry
(BankState.begin_telemetry -> per-layer own/off sums), so ONE forward serves
all K authors; the streaming aggregation is factored into model-free helpers
(NormAccumulator, summarize_norms, per_layer_ratio_medians, gate_verdict)
proven against a dense computation in tests/test_selectivity.py. Per-layer
ratio medians (16 values at all-layers) feed the wave-2 layer choice.

--recall_probe (pilot gate G2/G3): per resident author, teacher-forced answer
probability on their 20 QA rows — the OU formula exp(-avg CE over answer
tokens) (open-unlearning evals/metrics/utils.py evaluate_probability,
reimplemented inline; OU is not importable from test-env) — under all-active
vs own-only serving (bank.active mask). gap = all_active - own_only is the
self-interference number (negative = the other authors' banks hurt recall;
the anti-memsinks H2 tripwire).

--probe forget_leak (router_leak Part 3.1): the per-QUERY leak probe. Probe
groups: forget10 ORIGINAL questions (text-join author mapping), forget10
PARAPHRASED questions (forget10_perturbed, joined back on the original
(question, answer) text), a RandomState(seed) retain sample (default 400
rows), and the standard OOD sets. For every query it records the
mean-over-(answer-relevant tokens, layers) per-branch output norm, maxed over
the SURVIVING branches -> <out>.leak.npz per the pre-registered LEAK-PROBE
NPZ CONTRACT (keys: max_surv_norm / max_foreign_norm / top_surv_author /
own_norm / group / author_of_q + scalars n_surviving, droplist_tag, K).
Usable with and without --droplist — the without-run is the analyzer's
no-droplist reference npz. The standard probe outputs are untouched.

CLI (GPU job; CPU works for micro checkpoints):
  python measure_selectivity.py --config configs/sepmlp_1b_k200.json \
    --out reports/selectivity_k200.json [--checkpoint DIR] [--droplist F] \
    [--recall_probe] [--probe standard|forget_leak] [--device cuda]
Outputs: --out JSON (summary + per_author + provenance) and a sidecar
*_norms.npz of [K, L] mean norms (standard) or <out>.leak.npz (forget_leak).
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

from sepmlp_common import (
    NO_AUTHOR,
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
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

# Pre-registered gate thresholds — shared with measure_key_firing.py so the
# sepmlp number lands directly next to the LoRA anchor (1.11, 100% LAZY).
LAZY_LT = 2.0
SELECTIVE_GE = 5.0
OOD_SETS = ("world_facts", "real_authors", "alpaca")
TOFU_SISA_DIR = os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora"))
IGNORE_INDEX = -100  # OU schema (== data_tofu.IGNORE_INDEX)

# Leak-probe group labels — fixed by the shared NPZ contract; the analyzer
# switches on these exact strings.
LEAK_GROUPS = ("forget_orig", "forget_para", "retain",
               "ood_world_facts", "ood_real_authors", "ood_alpaca")


# ---------------------------------------------------------------------------
# Streaming aggregation (model-free; unit-tested in tests/test_selectivity.py)
# ---------------------------------------------------------------------------

class NormAccumulator:
    """Accumulates the banks' per-batch telemetry dicts into [L, K] float64
    sums/counts. One accumulator per probe group (author probes, each OOD set)
    so OOD tokens never contaminate the off-author norm."""

    def __init__(self, layers, num_authors: int):
        self.layers = sorted(int(l) for l in layers)
        self._index = {l: i for i, l in enumerate(self.layers)}
        shape = (len(self.layers), num_authors)
        self.own_sum = np.zeros(shape)
        self.own_cnt = np.zeros(shape)
        self.off_sum = np.zeros(shape)
        self.off_cnt = np.zeros(shape)

    def add(self, stats):
        """stats: list of per-layer dicts from BankState.end_telemetry()."""
        for s in stats:
            i = self._index[int(s["layer"])]
            self.own_sum[i] += s["own_sum"].double().numpy()
            self.own_cnt[i] += s["own_cnt"].double().numpy()
            self.off_sum[i] += s["off_sum"].double().numpy()
            self.off_cnt[i] += s["off_cnt"].double().numpy()

    def _pair(self, kind: str):
        assert kind in ("own", "off"), kind
        return ((self.own_sum, self.own_cnt) if kind == "own"
                else (self.off_sum, self.off_cnt))

    def agg_mean(self, kind: str) -> np.ndarray:
        """(K,) mean per-token norm, count-weighted across layers (the
        BankTelemetry convention: sum of sums / sum of counts)."""
        s, c = self._pair(kind)
        total_s, total_c = s.sum(0), c.sum(0)
        return np.divide(total_s, total_c, out=np.zeros_like(total_s),
                         where=total_c > 0)

    def per_layer_mean(self, kind: str) -> np.ndarray:
        """(L, K) per-layer mean per-token norm (0 where no tokens)."""
        s, c = self._pair(kind)
        return np.divide(s, c, out=np.zeros_like(s), where=c > 0)


def summarize_norms(author_acc: NormAccumulator, ood_accs: dict, author_ids):
    """Per-author on/off/ratio records + the ratio array the gate runs on.
    OOD rows are all-off (NO_AUTHOR), so their per-author mean is the 'off'
    channel of that set's accumulator."""
    on = author_acc.agg_mean("own")
    off = author_acc.agg_mean("off")
    ratios = on / np.clip(off, 1e-12, None)
    ood_means = {name: acc.agg_mean("off") for name, acc in ood_accs.items()}
    per_author = []
    for j, a in enumerate(author_ids):
        per_author.append({
            "author": int(a),
            "on_norm": float(on[j]),
            "off_norm": float(off[j]),
            "ratio": float(ratios[j]),
            "ood_norm": {name: float(v[j]) for name, v in ood_means.items()},
        })
    return per_author, ratios, on, off


def per_layer_ratio_medians(author_acc: NormAccumulator) -> dict:
    """{layer: median-over-authors on/off ratio} — the layer-resolved
    localization profile that informs the wave-2 layer-subset ablation."""
    on = author_acc.per_layer_mean("own")
    off = author_acc.per_layer_mean("off")
    ratio = on / np.clip(off, 1e-12, None)
    return {str(l): float(np.median(ratio[i]))
            for i, l in enumerate(author_acc.layers)}


def gate_verdict(ratios) -> dict:
    """The pre-registered LAZY/SELECTIVE gate on the per-author ratio array."""
    arr = np.asarray(ratios, dtype=np.float64)
    median = float(np.median(arr))
    verdict = ("LAZY" if median < LAZY_LT
               else "SELECTIVE" if median >= SELECTIVE_GE
               else "INTERMEDIATE")
    return {
        "gate_metric": "median per-author on/off ratio of mean-token bank output norm",
        "gate_thresholds": {"lazy_lt": LAZY_LT, "selective_ge": SELECTIVE_GE},
        "gate_median": median,
        "gate_verdict": verdict,
        "frac_ratio_lt_2": float((arr < LAZY_LT).mean()),
        "frac_ratio_ge_5": float((arr >= SELECTIVE_GE).mean()),
        "ratio_q25": float(np.percentile(arr, 25)),
        "ratio_q75": float(np.percentile(arr, 75)),
    }


# ---------------------------------------------------------------------------
# Probe-set construction
# ---------------------------------------------------------------------------

def build_ood_items(rs, args, tokenizer, data_tofu, max_length):
    """The OOD probe rows (world_facts, real_authors, alpaca), consumed from
    the CALLER's RandomState in OOD_SETS order — factored out so the standard
    and leak probes share one construction (and one determinism pattern)."""

    def _ood_item(pos, question, answer):
        item = data_tofu.preprocess_chat_instance(tokenizer, question, answer)
        item["index"] = -1 - pos  # never a real TOFU row index
        item["source_ids"] = NO_AUTHOR
        return item

    import datasets  # lazy: keeps module import light for the CPU tests

    ood_items = {}
    for split in ("world_facts", "real_authors"):
        rows = datasets.load_dataset("locuslab/TOFU", name=split, split="train")
        idx = rs.choice(len(rows), size=min(args.ood_n, len(rows)), replace=False)
        ood_items[split] = [
            _ood_item(p, rows[i]["question"], rows[i]["answer"])
            for p, i in enumerate(sorted(int(j) for j in idx))
        ]
    # Alpaca via the same helper measure_key_firing uses (read-only import).
    if TOFU_SISA_DIR not in sys.path:
        sys.path.insert(0, TOFU_SISA_DIR)
    from skill_data import load_alpaca

    # Spec training consumes the first 3*alpaca_n (<= 6000) rows of this SAME
    # seeded shuffle as suppression negatives — draw the probe rows from
    # beyond that head so ood_alpaca stays never-seen text. (world_facts is
    # the other clean OOD row; real_authors IS a trained negative source by
    # user decision — its ood_* rows measure trained suppression, not
    # generalization, and are labeled as such in the analyzer.)
    ALPACA_TRAIN_HEAD = 8000  # > 3*alpaca_n for every config in the suite
    pairs = load_alpaca(ALPACA_TRAIN_HEAD + args.ood_n, os.environ["HF_HOME"],
                        seed=args.seed)[ALPACA_TRAIN_HEAD:]
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
                  f"{len(ood_items[name]) - len(kept)} rows >= {max_length} tokens")
        ood_items[name] = kept
    return ood_items


def build_probe_sets(args, tokenizer, dataset, resident_ids, data_tofu, max_length):
    """(author_items, {ood_set: items}). All items are OU chat-template
    instances with index + source_ids, ready for QACollatorWithSources.

    Determinism contract: ONE RandomState(seed) consumed in a fixed order —
    all 200 author draws first (drawn even for dropped authors, so a droplist
    never shifts the surviving authors' question choice), then the OOD splits
    in OOD_SETS order (the measure_key_firing pattern)."""
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

    ood_items = build_ood_items(rs, args, tokenizer, data_tofu, max_length)
    return author_items, ood_items


def build_leak_probe_items(args, tokenizer, dataset, data_tofu, max_length):
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

    ood_items = build_ood_items(rs, args, tokenizer, data_tofu, max_length)
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
def run_norm_capture(model, state, batches, acc: NormAccumulator, device):
    """Grouped forwards with telemetry on: author batches carry true author
    ids, OOD batches carry NO_AUTHOR; the banks' own/off masks do the rest."""
    for batch in batches:
        state.set_batch(batch["source_ids"].to(device),
                        batch["attention_mask"].to(device))
        state.begin_telemetry()
        try:
            model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device),
                  use_cache=False)
            acc.add(state.end_telemetry())
        finally:
            state.clear()


@torch.no_grad()
def run_leak_capture(model, state, items, collator, batch_size, layers, device):
    """Per-QUERY [n_q, K] mean-over-(answer-relevant tokens, layers)
    per-branch output norms via the banks' row-stats telemetry (one grouped
    forward per batch). Answer-relevant tokens = labels != IGNORE and
    attention-visible (the collator's ne(pad) quirk masks the trailing eos —
    kept, so the probe pools exactly the served answer span). The forward is
    SOURCE-ID-FREE: the plain serving path, exactly what the leak measures."""
    out = []
    for batch in iter_batches(items, collator, batch_size):
        ans_mask = (batch["labels"] != IGNORE_INDEX) \
            & batch["attention_mask"].bool()
        assert bool(ans_mask.any(dim=1).all()), (
            "a probe row has zero answer-relevant tokens"
        )
        state.clear()
        state.begin_row_stats(ans_mask.to(device))
        try:
            model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device),
                  use_cache=False)
            stats = state.end_row_stats()
        finally:
            state.clear()
        assert len(stats) == len(layers), (stats and len(stats), len(layers))
        total = sum(s["sum"].double() for s in stats)            # (B, K)
        cnt = stats[0]["cnt"].double()                           # (B,)
        # every layer sees the same tokens, so mean-over-(tokens, layers)
        # = sum over layers / (n_layers * n_tokens)
        mean = total / (len(stats) * cnt.clamp_min(1)).unsqueeze(-1)
        out.append(mean.numpy())
    return np.concatenate(out, axis=0)


def assemble_leak_arrays(mean_norms, resident_ids, author_of_q, groups,
                         droplist_tag, total_authors):
    """Build the LEAK-PROBE NPZ CONTRACT arrays from the [n_q, K_surv] norm
    matrix. own_norm is NaN when the query author's branch is dropped/absent
    (orphans post-drop, all OOD); max_foreign_norm excludes the own branch
    when it survives (retain off-branch quiet level) and equals max_surv_norm
    otherwise. K_surv == 0 or an own-only single branch yield 0.0 maxima
    (silence), top_surv_author -1."""
    m = np.asarray(mean_norms, dtype=np.float64)
    n_q, K = m.shape
    rid = np.asarray(resident_ids, dtype=np.int64)
    aq = np.asarray(author_of_q, dtype=np.int64)
    groups = list(groups)
    assert aq.shape == (n_q,) and len(groups) == n_q
    assert set(groups) <= set(LEAK_GROUPS), sorted(set(groups))
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
        "group": np.array(groups),
        "author_of_q": aq.astype(np.int32),
        "n_surviving": int(K),
        "droplist_tag": str(droplist_tag),
        "K": int(total_authors),
    }


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
def own_only(banks, author_id: int):
    """Temporarily serve ONLY the given author's slices via the active mask
    (proven ≡ physical removal by the deletion-identity CPU gates); restores
    the exact prior masks even on exception."""
    saved = {l: b.active.clone() for l, b in banks.items()}
    try:
        for b in banks.values():
            mask = b.author_ids == author_id
            assert bool(mask.any()), f"author {author_id} not resident"
            b.active.copy_(mask)
        yield
    finally:
        for l, b in banks.items():
            b.active.copy_(saved[l])


def run_recall_probe(model, banks, state, dataset, resident_ids, collator,
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
        with own_only(banks, int(a)):
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
    return {
        "rows_per_author": RECORDS_PER_AUTHOR,
        "prob_all_active_mean": float(np.mean(
            [r["prob_all_active"] for r in per_author])),
        "prob_own_only_mean": float(np.mean(
            [r["prob_own_only"] for r in per_author])),
        # Negative mean = all-active serving loses recall to the other
        # authors' banks — the memsinks failure mode (H2 tripwire, gate G3).
        "gap_all_minus_own_mean": float(np.mean(
            [r["gap_all_minus_own"] for r in per_author])),
        "per_author": per_author,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="run dir with sepmlp.pt (default: config output_dir)")
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
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    set_determinism(args.seed)

    # Heavy imports after HF_HOME is set (storage-partition contract).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sepmlp_model import (
        apply_droplist_file,
        compute_bank_sha,
        install_banks,
        load_banks_from_checkpoint,
    )

    run_dir = args.checkpoint or cfg["output_dir"]
    banks, adapter_cfg, state = load_banks_from_checkpoint(run_dir)
    checkpoint_bank_sha = compute_bank_sha(banks)
    n_checkpoint_authors = next(iter(banks.values())).num_authors  # pre-drop K
    droplist_spec = None
    if args.droplist:
        # Order mirrors SepMlpLlamaForCausalLM: physical removal first, then
        # cast + install — the probe sees exactly the served post-drop banks.
        droplist_spec = apply_droplist_file(banks, args.droplist)
        print(f"[selectivity] droplist {droplist_spec['tag']}: dropped "
              f"{droplist_spec['_dropped_per_layer']} authors/layer in "
              f"{droplist_spec['_apply_seconds']:.4f}s")

    # bf16+sdpa on GPU (training/serving parity); fp32 on CPU (bf16 CPU
    # matmuls are slow and the CPU path is only for micro checkpoints).
    on_cuda = args.device.startswith("cuda")
    dtype = torch.bfloat16 if on_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, attn_implementation="sdpa"
    ).to(args.device).eval()
    for bank in banks.values():
        bank.to(device=args.device, dtype=dtype)
    install_banks(model, banks, state)
    # After install: fresh bank/wrapper modules default to train mode, whose
    # guard (rightly) refuses source-id-free forwards — the recall and leak
    # probes ARE source-id-free serving forwards, so eval the whole tree.
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    data_tofu = import_memadapt_data()
    max_length = cfg.get("data", {}).get("max_length", 512)
    dataset = data_tofu.TofuQADataset(tokenizer, split="full",
                                      max_length=max_length)
    collator = data_tofu.QACollatorWithSources(tokenizer)

    resident_ids = [int(a) for a in
                    next(iter(banks.values())).author_ids.tolist()]
    layers = sorted(banks.keys())
    K = len(resident_ids)
    print(f"[selectivity] {K} resident authors, layers {layers[0]}..{layers[-1]}"
          f" ({len(layers)}), device {args.device}")

    if args.probe == "forget_leak":
        # Per-query leak probe (router_leak Part 3.1). Droplist (if any) was
        # applied ABOVE, before install — the probe sees the post-drop banks.
        triples = build_leak_probe_items(args, tokenizer, dataset, data_tofu,
                                         max_length)
        groups = [g for g, _, _ in triples]
        authors_q = [a for _, a, _ in triples]
        items = [it for _, _, it in triples]
        counts = {g: groups.count(g) for g in LEAK_GROUPS}
        print(f"[leak] {len(items)} query rows: {counts}")
        mean_norms = run_leak_capture(model, state, items, collator,
                                      args.batch_size, layers, args.device)
        arrays = assemble_leak_arrays(
            mean_norms, resident_ids, authors_q, groups,
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
                "max_surv_norm_mean": float(arrays["max_surv_norm"][g_arr == g].mean()),
                "max_surv_norm_p90": float(np.percentile(
                    arrays["max_surv_norm"][g_arr == g], 90)),
            }
            for g in LEAK_GROUPS if counts[g]
        }
        result = {
            "probe": "forget_leak",
            "config": {
                **{k: v for k, v in vars(args).items()},
                "config_path": cfg["_config_path"],
                "model_name": cfg["model_name"],
                "adapter_cfg": adapter_cfg,
            },
            "checkpoint": os.path.abspath(run_dir),
            "droplist": os.path.abspath(args.droplist) if args.droplist else None,
            "droplist_tag": droplist_spec["tag"] if droplist_spec else None,
            "checkpoint_bank_sha": checkpoint_bank_sha,
            "served_bank_sha": compute_bank_sha(banks),
            "n_surviving": int(K),
            "n_checkpoint_authors": int(n_checkpoint_authors),
            "layers": layers,
            "n_queries": len(items),
            "group_counts": counts,
            "per_group_summary": per_group,
            "leak_npz": npz_path,
            "script_sha256": file_sha256(os.path.abspath(__file__)),
            "slurm_job_id": slurm_job_id(),
            "torch_version": torch.__version__,
        }
        save_json(result, out_path)
        print(f"[leak] wrote {out_path} (+ {npz_path})")
        return

    author_items, ood_items = build_probe_sets(
        args, tokenizer, dataset, resident_ids, data_tofu, max_length)
    n_prompts = len(author_items) + sum(len(v) for v in ood_items.values())
    print(f"[selectivity] {len(author_items)} author rows + "
          f"{ {k: len(v) for k, v in ood_items.items()} } OOD rows")

    author_acc = NormAccumulator(layers, K)
    run_norm_capture(model, state,
                     iter_batches(author_items, collator, args.batch_size),
                     author_acc, args.device)
    ood_accs = {}
    for name, items in ood_items.items():
        acc = NormAccumulator(layers, K)
        run_norm_capture(model, state,
                         iter_batches(items, collator, args.batch_size),
                         acc, args.device)
        ood_accs[name] = acc

    per_author, ratios, on, off = summarize_norms(author_acc, ood_accs,
                                                  resident_ids)
    summary = gate_verdict(ratios)
    on_mean = float(on.mean())
    summary["on_norm_mean"] = on_mean
    summary["off_norm_mean"] = float(off.mean())
    summary["ood_norm_mean"] = {
        name: float(acc.agg_mean("off").mean()) for name, acc in ood_accs.items()
    }
    # Pre-registered Alpaca-negatives trigger: any OOD/own firing > 0.1.
    summary["ood_over_own"] = {
        name: float(v / max(on_mean, 1e-12))
        for name, v in summary["ood_norm_mean"].items()
    }
    summary["per_layer_ratio_median"] = per_layer_ratio_medians(author_acc)

    recall = None
    if args.recall_probe:
        print("[selectivity] recall probe (all-active vs own-only)")
        recall = run_recall_probe(model, banks, state, dataset, resident_ids,
                                  collator, args.device, args.batch_size)
        print(f"[selectivity] recall: all-active "
              f"{recall['prob_all_active_mean']:.4f} vs own-only "
              f"{recall['prob_own_only_mean']:.4f} "
              f"(gap {recall['gap_all_minus_own_mean']:+.4f})")

    out_path = os.path.abspath(args.out)
    npz_path = os.path.splitext(out_path)[0] + "_norms.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        author_ids=np.array(resident_ids),
        layers=np.array(layers),
        on_norms=author_acc.per_layer_mean("own").T,    # [K, L]
        off_norms=author_acc.per_layer_mean("off").T,   # [K, L]
        **{f"ood_{name}": acc.per_layer_mean("off").T
           for name, acc in ood_accs.items()},
    )

    result = {
        "config": {
            **{k: v for k, v in vars(args).items()},
            "config_path": cfg["_config_path"],
            "model_name": cfg["model_name"],
            "adapter_cfg": adapter_cfg,
        },
        "checkpoint": os.path.abspath(run_dir),
        "droplist": os.path.abspath(args.droplist) if args.droplist else None,
        "droplist_tag": droplist_spec["tag"] if droplist_spec else None,
        "checkpoint_bank_sha": checkpoint_bank_sha,
        "served_bank_sha": compute_bank_sha(banks),
        "n_authors": K,
        "layers": layers,
        "n_prompts": int(n_prompts),
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "torch_version": torch.__version__,
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
