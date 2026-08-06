"""Two-phase gradient-routed training of the single block transcoder
(--smoke = phase-0 -> phase-1 micro gate).

Schedule (DESIGN.md SS3/SS5/SS7, binding):
  phase 0  shared block ONLY. Plain LM loss (REAL answer labels) on the
           author-free pool: the first `alpaca_n` (2000) rows of the seed-42
           Alpaca shuffle + all 100 TOFU real_authors rows. NEVER any of the
           200 TOFU authors' rows (all are deletable candidates - "retain"
           data would contaminate the undeletable shared block; deliberate
           divergence from the user's design doc, which said retain+generic)
           and NEVER the never-train control split (referenced only through
           the guard helpers - a static CPU gate forbids this file from even
           naming it, sepmlp precedent).
  phase 1  loads `phase0_checkpoint`, FRESH AdamW (Adam moments are
           data-functions - phase-0 moments are never carried, and v1 never
           resumes training after a deletion), alternates 1:1 SINGLE-SOURCE
           author batches (routed LM loss) with generic NO_AUTHOR batches
           (lambda-warmed L1 suppression ONLY; labels IGNORE except the
           final token, the sepmlp NaN guard for HF's num_items_in_batch
           machinery). Generic-batch total = lambda_t * L_supp - the 1-token
           LM term is computed by the HF plumbing but deliberately NOT added
           to the returned loss: the empty own-mask already makes its module
           gradient exactly zero (pinned by debug_grad_check's
           phase1-generic-lm pass), and DESIGN SS3 pins the generic objective
           to suppression alone. Generic Alpaca rows are drawn BEYOND the
           8000-row training head (ALPACA_TRAIN_HEAD) so probe rows can stay
           unseen; the phase-1 raw draw is rows [8000, 8000 + 3*alpaca_n) of
           the SAME seed-42 shuffle, so the selectivity probe must draw
           beyond 8000 + 3*alpaca_n.

Topology decision (documented because "K=20 pilot" admits two readings): the
transcoder is ALWAYS built/loaded with the FULL `n_authors` (200) blocks;
`authors_subset` restricts only (i) which authors' rows the phase-1 sampler
draws and (ii) which blocks get detector init. Rejected alternative: building
a K=|subset| transcoder per run - it would break the one-phase-0-checkpoint
contract (`load_tc_from_checkpoint` and `assert_shared_frozen` both pin the
full topology via tc_sha, and the driver feeds runs/phase0_s42/blocktc.pt to
every pilot arm AND the K=200 run). tc_layer's prefix-seeded init makes the
two readings agree on the initial author rows anyway, so nothing is lost.

Exactness belts (DESIGN SS3 - all four, per phase; never "simplify"):
  (a) own-mask semantics live in tc_layer (detach trick THROUGH the
      decoders: value bitwise-identical to serving, gradient only through
      the phase's own slice);
  (b) an optimizer-step pre-hook zeroes grads outside the phase's permitted
      slices (phase 0: everything but shared; phase 1: shared always);
  (c) debug_grad_check asserts EXACT-zero grads on forbidden slices for
      every (phase x batch-type) - single backward each, sound at any batch
      size; ga-invariance is compute_loss's SEPARATE job (the suppression
      term is scaled by 1/grad_accum on the transformers-4.48
      summed-micro-loss path; sepmlp trap 2: keep both, they solve
      different problems);
  (d) the phase-1 save first asserts the shared slices bitwise-equal to the
      phase-0 checkpoint (assert_shared_frozen); phase 0 symmetrically
      asserts the author slices bitwise-equal to the fresh seeded init.
weight_decay=0 and args.max_grad_norm=0 ALWAYS (decay / a global clip would
couple idle authors' parameters to every step); clipping is per-block: one
block's encoder rows + bias entries + decoder columns = one clip group,
clip norm from config (sepmlp per_author_clip_ analog).

Detector init (phase 1 only, BEFORE install_tc - the pre-pass hooks the RAW
mlp input): per-subset-author question-token mean directions, applied
ADDITIVELY on top of the checkpoint's seeded author rows
(tc_model.apply_detector_init). Rejected: running the pre-pass in phase 0 -
phase 0 is author-free by design, phase-0 gradients never reach author rows
(belts a+b), so the phase-0 checkpoint's author rows ARE the seeded init
that phase 1 re-points; a phase-0 pre-pass would forward all authors' rows
for nothing.

--smoke: phase 0 THEN phase 1 in one process (~5 steps each, FULL-SIZE
transcoder, subset-limited data), grad checks forced on, save -> reload ->
bitwise + module-forward parity, per-phase peak-memory print (the bs32
go/no-go: sepmlp lesson - a small-bs smoke does NOT clear bs32).
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from tc_common import (
    ALPACA_TRAIN_HEAD,
    HF_HOME,
    IGNORE_INDEX,
    NO_AUTHOR,
    RECORDS_PER_AUTHOR,
    STORAGE_ROOT,
    assert_never_train_clean,
    file_sha256,
    import_memadapt_data,
    load_config,
    never_train_questions,
    set_determinism,
    slurm_job_id,
)
from tc_layer import BlockTranscoder, TcState
from tc_model import (
    apply_detector_init,
    assert_shared_frozen,
    detector_init_cached,
    freeze_base,
    install_tc,
    load_tc_from_checkpoint,
    save_checkpoint,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read-only import root for skill_data.load_alpaca (sepmlp precedent -
# imported in place, never copied).
TOFU_SISA_DIR = os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora"))

# ALPACA_TRAIN_HEAD (the seed-42 shuffle training head) and the probe-head
# arithmetic live in tc_common as the single source of truth shared with
# measure_selectivity.py: phase 0 over-draws [0, 3*alpaca_n), phase 1 draws
# [HEAD, HEAD + 3*alpaca_n), and the selectivity probe must start beyond that
# (tc_common.alpaca_probe_head). Both phases' configs must share one seed for
# the disjointness to hold (all house configs use 42).

# DESIGN SS7 config schema - a CLOSED set: an unknown key is a config bug (a
# typo'd hyperparameter silently falling back to a default is exactly the
# failure mode this blocks), never an extension point. "_"-prefixed keys are
# notes and are ignored.
CONFIG_KEYS = {
    "model", "insert_layer", "span", "m_author", "m_shared", "n_authors",
    "authors_subset", "seed", "max_length", "batch_size", "grad_accum",
    "epochs", "lr", "lambda_max", "lambda_warmup_frac", "clip_norm",
    "detector_init", "init_scale", "alpaca_n", "phase", "phase0_checkpoint",
    "run_name",
}

# sha256'd into every run's meta.json (root CLAUDE.md provenance: sha256s,
# never commits). train_tc.py plus the three modules whose behavior defines
# the checkpoint semantics.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROVENANCE_FILES = [os.path.abspath(__file__)] + [
    os.path.join(_HERE, f) for f in ("tc_common.py", "tc_layer.py", "tc_model.py")
]

# Fixed OOD telemetry probe (verbatim sepmlp set): generic non-TOFU
# questions, hardcoded so the training loop has zero extra data dependencies
# and NEVER touches the held-out control split. The proper OOD eval
# (world_facts / real_authors / Alpaca-beyond-head) lives in
# measure_selectivity.py.
OOD_PROBE_QUESTIONS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Who wrote the play Romeo and Juliet?",
    "What is the boiling point of water at sea level?",
    "Describe the water cycle in simple terms.",
    "What year did the Second World War end?",
    "How does a refrigerator keep food cold?",
    "Name the largest planet in the solar system.",
    "What is the chemical symbol for gold?",
    "Summarize the plot of a typical detective novel.",
    "How do vaccines protect against disease?",
    "What causes the seasons to change on Earth?",
    "Give three tips for improving sleep quality.",
    "What is the difference between weather and climate?",
    "How is paper traditionally made from wood?",
    "Why does the moon appear to change shape?",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def validate_config(cfg: dict):
    keys = {k for k in cfg if not k.startswith("_")}
    missing, extra = CONFIG_KEYS - keys, keys - CONFIG_KEYS
    assert not missing and not extra, (
        f"config schema drift: missing {sorted(missing)}, unknown "
        f"{sorted(extra)} (DESIGN SS7 is a closed schema)"
    )
    assert cfg["phase"] in ("phase0", "phase1"), cfg["phase"]
    if cfg["phase"] == "phase0":
        assert cfg["phase0_checkpoint"] is None, (
            "phase 0 takes no phase0_checkpoint"
        )
    else:
        assert cfg["phase0_checkpoint"], "phase 1 requires phase0_checkpoint"
    assert cfg["detector_init"] in ("questions", "random"), cfg["detector_init"]
    assert cfg["clip_norm"] > 0, "per-block clip norm must be positive"
    assert 0.0 <= cfg["lambda_warmup_frac"] <= 1.0, cfg["lambda_warmup_frac"]
    assert cfg["lambda_max"] >= 0.0, cfg["lambda_max"]
    assert cfg["n_authors"] >= 1 and cfg["m_author"] >= 1 and cfg["m_shared"] >= 1
    assert cfg["batch_size"] >= 1 and cfg["grad_accum"] >= 1 and cfg["epochs"] >= 1
    assert cfg["alpaca_n"] >= 1 and cfg["max_length"] > 0
    sub = cfg["authors_subset"]
    if sub is not None:
        assert (isinstance(sub, (list, tuple)) and len(sub) == 2
                and 0 <= int(sub[0]) < int(sub[1]) <= cfg["n_authors"]), (
            f"authors_subset must be [lo, hi) within [0, n_authors): {sub}"
        )
    assert cfg["run_name"], "run_name required (checkpoints/runs/<run_name>)"


def resolve_subset(cfg: dict):
    """Global author ids whose rows phase 1 trains (and whose blocks get
    detector init). authors_subset is a DATA restriction, never a topology
    one (see module docstring)."""
    sub = cfg["authors_subset"]
    if sub is None:
        return list(range(cfg["n_authors"]))
    return list(range(int(sub[0]), int(sub[1])))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PoolDataset(Dataset):
    """Author-free QA pool (all rows NO_AUTHOR): public Alpaca + all 100
    TOFU real_authors rows (real_authors is a deliberately-shared source
    between the two phases: phase 0 learns generic LM behavior on it, phase 1
    suppresses author features on it - both are legal because it belongs to
    no deletable author).

      mask_labels=False  phase-0 LM pool: REAL answer labels.
      mask_labels=True   phase-1 suppression pool: labels IGNORE except the
                         final token - the LM gradient into the module is
                         exactly zero on NO_AUTHOR rows anyway (empty
                         own-mask), and the single live label keeps HF's
                         num_items_in_batch nonzero on an all-generic
                         accumulation window (sepmlp NaN guard).

    alpaca_skip: raw shuffle rows [0, alpaca_skip) are skipped BEFORE
    length-filtering, so phase 1 (skip=ALPACA_TRAIN_HEAD) can never touch a
    row phase 0 (skip=0) may have trained on. Every question is
    membership-checked against the never-train split at construction time -
    batches only ever draw from guarded pools.
    """

    def __init__(self, tokenizer, alpaca_n: int, seed: int, max_length: int,
                 guarded: set, mask_labels: bool, alpaca_skip: int, what: str):
        data_tofu = import_memadapt_data()
        if TOFU_SISA_DIR not in sys.path:
            sys.path.insert(0, TOFU_SISA_DIR)
        from skill_data import load_alpaca

        import datasets

        # 3x over-draw: length filtering happens at tokenize time (sepmlp
        # pattern); the assert below pins that the head contract held.
        raw = load_alpaca(alpaca_skip + 3 * alpaca_n, HF_HOME, seed=seed)
        assert len(raw) == alpaca_skip + 3 * alpaca_n, (
            f"Alpaca draw returned {len(raw)} rows, wanted "
            f"{alpaca_skip + 3 * alpaca_n} - head arithmetic is broken"
        )
        alpaca_pairs = raw[alpaca_skip:]
        ra = datasets.load_dataset("locuslab/TOFU", name="real_authors",
                                   split="train")
        ra_pairs = list(zip(ra["question"], ra["answer"]))
        assert len(ra_pairs) == 100, len(ra_pairs)
        assert_never_train_clean(
            [p["question"] for p in alpaca_pairs] + [q for q, _ in ra_pairs],
            guarded, what)

        def _tokenize(question, answer, pos):
            item = data_tofu.preprocess_chat_instance(tokenizer, question,
                                                      answer)
            if len(item["input_ids"]) >= max_length:
                return None
            if mask_labels:
                labels = torch.full_like(item["labels"], IGNORE_INDEX)
                labels[-1] = item["input_ids"][-1]
                item["labels"] = labels
            item["index"] = -1000 - pos     # never collides with TOFU rows
            item["source_ids"] = NO_AUTHOR
            return item

        self.items = []
        n_alpaca = 0
        for pair in alpaca_pairs:
            if n_alpaca >= alpaca_n:
                break
            item = _tokenize(pair["question"], pair["answer"], len(self.items))
            if item is not None:
                self.items.append(item)
                n_alpaca += 1
        assert n_alpaca == alpaca_n, (
            f"only {n_alpaca}/{alpaca_n} Alpaca rows fit max_length {max_length}"
        )
        n_ra = 0
        for q, a in ra_pairs:
            item = _tokenize(q, a, len(self.items))
            if item is not None:
                self.items.append(item)
                n_ra += 1
        assert n_ra > 0, "no real_authors rows fit max_length"
        self.n_alpaca, self.n_real_authors = n_alpaca, n_ra

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class AlternatingBatchSampler(Sampler):
    """Single-source author-batch / generic-batch 1:1 alternation over
    ConcatDataset([author_ds, generic_ds]) (DESIGN SS5).

    Two deliberate divergences from the sepmlp original:
      * author batches are SINGLE-SOURCE by construction - DESIGN SS3 wants
        exactly one own-mask block per LM batch, so rows are grouped per
        author and chunked into per-author batches (never mixed; the trainer
        re-asserts purity per batch as a belt);
      * authors run ROUND-ROBIN: each cycle visits every author once (its
        next chunk) in a per-epoch reshuffled order - uniform coverage with
        no author starved to the tail of an epoch.
    One "epoch" = every author row exactly once, interleaved with an EQUAL
    number of generic batches (1:1 by BATCH count, not row count: an author
    batch may be short, e.g. 20 rows at bs 32; generic batches are always
    full). Generic rows reshuffle-cycle to cover demand (sepmlp pattern).
    Deterministic: generator seeded seed*1_000_003 + epoch - the custom
    batch_sampler bypasses HF's per-epoch reseeding, so determinism is owned
    here (sepmlp precedent, DESIGN SS5 pins the seed formula).
    """

    def __init__(self, author_groups, n_generic: int, batch_size: int,
                 seed: int):
        assert author_groups and all(len(g) > 0 for g in author_groups)
        assert n_generic > 0 and batch_size > 0
        self.author_groups = [list(g) for g in author_groups]
        self.offset = sum(len(g) for g in self.author_groups)
        # Groups must partition the author part of the ConcatDataset - a gap
        # or overlap would silently train the wrong rows.
        flat = sorted(i for g in self.author_groups for i in g)
        assert flat == list(range(self.offset)), (
            "author_groups do not partition [0, n_author_rows)"
        )
        self.n_generic = int(n_generic)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._epoch = 0
        self.n_author_batches = sum(
            (len(g) + batch_size - 1) // batch_size for g in self.author_groups
        )

    def __len__(self):
        return 2 * self.n_author_batches

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self._epoch)
        self._epoch += 1
        bs = self.batch_size
        # Per-author shuffled row order, chunked into single-source batches.
        chunks = []
        for rows in self.author_groups:
            perm = torch.randperm(len(rows), generator=g).tolist()
            shuffled = [rows[i] for i in perm]
            chunks.append([shuffled[s:s + bs]
                           for s in range(0, len(shuffled), bs)])
        # Round-robin: chunk r of every author (per-epoch shuffled author
        # order) before chunk r+1 of any.
        order = torch.randperm(len(chunks), generator=g).tolist()
        author_batches = []
        for r in range(max(len(c) for c in chunks)):
            for k in order:
                if r < len(chunks[k]):
                    author_batches.append(chunks[k][r])
        assert len(author_batches) == self.n_author_batches
        need = self.n_author_batches * bs
        gen = []
        while len(gen) < need:
            gen += (torch.randperm(self.n_generic, generator=g)
                    + self.offset).tolist()   # offset into the ConcatDataset
        for b, ab in enumerate(author_batches):
            yield ab
            yield gen[b * bs:(b + 1) * bs]


# ---------------------------------------------------------------------------
# Gradient belts: forbidden-slice zeroing + per-block clip
# ---------------------------------------------------------------------------

def zero_forbidden_grads_(tc: BlockTranscoder, phase: str):
    """Belt (b) of DESIGN SS3: zero grads outside the phase's permitted
    slices (phase 0: everything but shared; phase 1: shared always). Runs in
    the optimizer step pre-hook BEFORE the per-block clip so a leaked grad
    can never survive by hiding inside a clip group. Belt (a) - the own-mask
    detach construction - should make every one of these slices exactly zero
    already; this hook is redundancy, not the mechanism (debug_grad_check is
    the assert that (a) actually holds)."""
    assert phase in ("phase0", "phase1"), phase
    S = tc.shared_start
    ge, gb, gd = tc.W_enc.grad, tc.b_enc.grad, tc.W_dec.grad
    with torch.no_grad():
        if phase == "phase0":
            if ge is not None:
                ge[:S].zero_()
            if gb is not None:
                gb[:S].zero_()
            if gd is not None:
                gd[..., :S].zero_()
        else:
            if ge is not None:
                ge[S:].zero_()
            if gb is not None:
                gb[S:].zero_()
            if gd is not None:
                gd[..., S:].zero_()


def per_block_clip_(tc: BlockTranscoder, max_norm: float, eps: float = 1e-6):
    """Clip each block's gradient slice to max_norm, with the norm taken over
    that block's encoder rows + bias entries + decoder columns ACROSS ALL
    span decoders (one block = one clipping group; a spiking author cannot
    rescale anyone else - sepmlp per_author_clip_ analog). K author groups +
    the shared block as group K (it trains in phase 0 and deserves the same
    protection). Runs as an optimizer step pre-hook, replacing the global
    clip (args.max_grad_norm must be 0). Returns the (K+1,) pre-clip norms
    (shared LAST) for telemetry/tests.

    Implementation via a per-feature coefficient vector rather than sliced
    .view().mul_ chains (rejected: in-place view() on the non-contiguous
    W_dec column slice leans on stride minutiae; the coefficient-vector form
    is shape-proof and covers all three tensors uniformly)."""
    K, m, S = tc.num_authors, tc.m_author, tc.shared_start
    ge, gb, gd = tc.W_enc.grad, tc.b_enc.grad, tc.W_dec.grad
    total = torch.zeros(K + 1, device=tc.W_enc.device, dtype=torch.float32)
    if ge is not None:
        total[:K] += ge[:S].reshape(K, m, -1).float().pow(2).sum(dim=(1, 2))
        total[K] += ge[S:].float().pow(2).sum()
    if gb is not None:
        total[:K] += gb[:S].reshape(K, m).float().pow(2).sum(dim=1)
        total[K] += gb[S:].float().pow(2).sum()
    if gd is not None:
        total[:K] += gd[..., :S].reshape(tc.span, tc.hidden, K, m) \
            .float().pow(2).sum(dim=(0, 1, 3))
        total[K] += gd[..., S:].float().pow(2).sum()
    norms = total.sqrt()
    coef = (max_norm / (norms + eps)).clamp(max=1.0)
    feat_coef = torch.cat([coef[:K].repeat_interleave(m),
                           coef[K:].expand(tc.m_shared)])
    with torch.no_grad():
        if ge is not None:
            ge.mul_(feat_coef.view(-1, 1).to(ge.dtype))
        if gb is not None:
            gb.mul_(feat_coef.to(gb.dtype))
        if gd is not None:
            gd.mul_(feat_coef.view(1, 1, -1).to(gd.dtype))
    return norms


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BlockTcTrainer(Trainer):
    """Plain HF Trainer subclass - NEVER SFTTrainer (it re-tokenizes into its
    own schema and drops source_ids/index; trap 3). Routes per-sequence
    source ids + question mask into the shared TcState around each forward
    (clearing in a finally: so a crashed forward cannot leak a stale stash),
    dispatches the per-batch objective (phase 0 / phase-1 author: routed LM;
    phase-1 generic: lambda-warmed suppression ONLY), and installs the
    zero-forbidden + per-block-clip optimizer step pre-hook."""

    def __init__(self, *args, tc: BlockTranscoder = None,
                 tc_state: TcState = None, phase: str = None,
                 lambda_max: float = 0.0, lambda_warmup_frac: float = 0.15,
                 clip_norm: float = 0.0,
                 alt_sampler: AlternatingBatchSampler = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert phase in ("phase0", "phase1"), phase
        assert tc is not None and tc_state is not None
        # Exactness knobs are hard requirements, not preferences (DESIGN SS3).
        assert self.args.weight_decay == 0.0, (
            "weight_decay must be 0: decay would couple idle authors' "
            "parameters to every step (breaks exactness AND decays idle blocks)"
        )
        assert self.args.max_grad_norm in (0, 0.0), (
            "HF's global clip must be off - per-block clip owns clipping "
            "(a global norm couples authors through the rescale factor)"
        )
        assert self.args.remove_unused_columns is False, (
            "source_ids/index must survive to the collator (trap 3)"
        )
        assert self.args.dataloader_num_workers == 0, (
            "single-process loading only: determinism + collator/state "
            "locality (sepmlp precedent)"
        )
        self.tc = tc
        self.tc_state = tc_state
        self.phase = phase
        self.lambda_max = float(lambda_max)
        self.lambda_warmup_frac = float(lambda_warmup_frac)
        self.clip_norm = float(clip_norm)
        self.alt_sampler = alt_sampler
        self.loss_components = []
        self.seen_sources = set()
        self._step_hook = None

    def get_train_dataloader(self):
        if self.alt_sampler is None:
            # phase 0: HF's own seeded RandomSampler over the flat pool.
            return super().get_train_dataloader()
        dl = DataLoader(
            self.train_dataset,
            batch_sampler=self.alt_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(dl)

    def create_optimizer(self):
        opt = super().create_optimizer()
        if self._step_hook is None:
            tc, phase, clip = self.tc, self.phase, self.clip_norm

            def _pre_step(optimizer, hook_args, hook_kwargs):
                # Order load-bearing: zero forbidden slices FIRST so a
                # (never-expected) leaked grad cannot inflate a clip norm.
                zero_forbidden_grads_(tc, phase)
                if clip > 0:
                    per_block_clip_(tc, clip)

            # Hook on the raw torch optimizer: accelerate's wrapper calls
            # its .step(), which fires registered pre-hooks (bf16 => no grad
            # scaler, grads are unscaled here). args.max_grad_norm is 0 so
            # HF's global clip never runs on top (asserted in __init__).
            self._step_hook = self.optimizer.register_step_pre_hook(_pre_step)
        return opt

    def _lambda_now(self) -> float:
        """lambda_t: linear 0 -> lambda_max over the first lambda_warmup_frac
        of this run's OPTIMIZER steps (DESIGN SS3). global_step is constant
        across a ga window, so every micro-batch of one optimizer step sees
        the same lambda (warmup itself is ga-invariant)."""
        total = max(1, int(self.state.max_steps))
        warm = self.lambda_warmup_frac * total
        if warm <= 0:
            return self.lambda_max
        return self.lambda_max * min(1.0, float(self.state.global_step) / warm)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        # source_ids/index must never reach model.forward (HF's Llama would
        # reject them); they travel via TcState only (trap 4).
        source_ids = inputs.pop("source_ids")
        inputs.pop("index", None)
        self.seen_sources.update(source_ids.detach().cpu().tolist())
        attn = inputs["attention_mask"]
        # question tokens = the prompt part of each row (labels IGNORE and
        # attended) - the same mask the detector-init pre-pass pools over.
        # Suppression pools over attention_mask, not this.
        qmask = (inputs["labels"] == IGNORE_INDEX) & attn.bool()
        sid = source_ids.view(-1)
        is_generic = bool((sid == NO_AUTHOR).all())
        if self.phase == "phase0":
            assert is_generic, (
                "phase-0 batch carries author rows - the author-free pool is "
                "contaminated (tc_layer would also refuse, but fail here first)"
            )
        elif not is_generic:
            # Sampler contract: phase-1 LM batches are SINGLE-source (one
            # own-mask block per batch). Mixed rows would still be exact
            # (the own-mask is per-row) but would mean the sampler drifted -
            # fail loudly rather than train through a broken schedule.
            assert bool((sid == sid[0]).all()), (
                f"phase-1 author batch is not single-source: "
                f"{sid.unique().tolist()[:8]}"
            )
        suppress = (self.phase == "phase1" and is_generic and model.training)
        self.tc_state.set_batch(source_ids, question_mask=qmask,
                                attention_mask=attn)
        if suppress:
            self.tc_state.begin_suppression()
        try:
            out = super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            lm_loss = out[0] if return_outputs else out
            comp = {"call": len(self.loss_components),
                    "type": ("phase0" if self.phase == "phase0"
                             else ("generic" if is_generic else "author")),
                    "lm": float(lm_loss.detach()), "supp": 0.0, "lambda": None}
            if suppress:
                terms = self.tc_state.end_suppression()
                assert len(terms) == 1, (
                    f"collected {len(terms)} suppression terms for one "
                    "transcoder - encode ran more than once this forward "
                    "(per-layer gradient checkpointing?)"
                )
                lam = self._lambda_now()
                # Generic-batch total = lambda_t * L_supp ONLY (DESIGN SS3).
                # The 1-live-token LM term above exists solely to keep HF's
                # num_items_in_batch machinery finite (NaN guard); its module
                # gradient is exactly zero anyway (empty own-mask), so
                # dropping it from the returned loss changes no gradient -
                # asserted by debug_grad_check's phase1-generic-lm pass. The
                # suppression graph is independent of the LM forward graph
                # (recomputed from xn.detach() in tc_layer), so backward
                # traverses only the small encoder recompute.
                loss = lam * terms[0]
                # transformers 4.48 num_items_in_batch path SUMS micro-batch
                # losses across an accumulation window (no /ga afterwards);
                # scale the per-micro mean term so lambda is ga-invariant.
                # No-op at ga=1.
                if num_items_in_batch is not None:
                    loss = loss / self.args.gradient_accumulation_steps
                comp["supp"] = float(terms[0].detach())
                comp["lambda"] = lam
            else:
                loss = lm_loss
            self.loss_components.append(comp)
            if comp["call"] % 25 == 0:
                print(f"[loss] call={comp['call']} type={comp['type']} "
                      f"lm={comp['lm']:.4f} supp={comp['supp']:.6f} "
                      f"lambda={comp['lambda']}")
            return (loss, out[1]) if return_outputs else loss
        finally:
            self.tc_state.clear()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TcTelemetry(TrainerCallback):
    """Epoch-end own/off/OOD per-block act-mass telemetry (sepmlp
    BankTelemetry analog) - the dead/lazy-block visibility DESIGN SS2 keeps
    in v1 in lieu of sepmlp's 4-term recipe. off/own should FALL as
    suppression bites; ood must stay near 0; a block whose own mass sits at
    ~0 across epochs is DEAD (evidence for the pre-registered
    hinge/Gram/promotion fallback - a log/README discussion, not a code
    change here). probe_batch is None in phase 0: probing author rows would
    forward author data for no signal (nothing is "own" of the shared
    block), so phase 0 tracks only the OOD/shared masses."""

    def __init__(self, model, tc_state: TcState, probe_batch, ood_batch):
        self.model = model
        self.tc_state = tc_state
        self.probe = probe_batch
        self.ood = ood_batch
        self.history = []

    @torch.no_grad()
    def _mass(self, batch):
        device = next(self.model.parameters()).device
        self.tc_state.set_batch(
            batch["source_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        self.tc_state.begin_telemetry()
        was_training = self.model.training
        self.model.eval()
        try:
            self.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
            )
            stats = self.tc_state.end_telemetry()
        finally:
            self.tc_state.clear()
            if was_training:
                self.model.train()
        assert len(stats) == 1, (
            f"{len(stats)} telemetry entries for one transcoder forward"
        )
        s = stats[0]
        own_mean = s["own_sum"] / s["own_cnt"].clamp_min(1)
        off_mean = s["off_sum"] / s["off_cnt"].clamp_min(1)
        shared = float(s["shared_sum"] / s["shared_cnt"].clamp_min(1))
        return own_mean, s["own_cnt"], off_mean, shared

    def on_epoch_end(self, args, state, control, **kwargs):
        rec = {"epoch": state.epoch}
        _, _, ood_mean, ood_shared = self._mass(self.ood)
        rec["ood_mass_mean"] = float(ood_mean.mean())
        rec["shared_ood_mass"] = ood_shared
        if self.probe is not None:
            own_mean, own_cnt, off_mean, probe_shared = self._mass(self.probe)
            has = own_cnt > 0
            any_own = bool(has.any())
            ratio = own_mean[has] / off_mean[has].clamp_min(1e-12)
            rec.update(
                own_mass_mean=float(own_mean[has].mean()) if any_own else None,
                off_mass_mean=float(off_mean.mean()),
                onoff_ratio_median=float(ratio.median()) if any_own else None,
                shared_probe_mass=probe_shared,
                ood_over_own=(
                    float(ood_mean.mean()
                          / own_mean[has].mean().clamp_min(1e-12))
                    if any_own else None
                ),
            )
        self.history.append(rec)
        print(f"[telemetry] {json.dumps(rec)}")


def build_probe_batches(full_dataset, collator, tokenizer, subset_ids, phase):
    """Fixed telemetry batches (probe, ood). probe = 2 rows each from up to
    8 evenly-spread SUBSET authors (None in phase 0 - see TcTelemetry); ood =
    the hardcoded generic questions with NO_AUTHOR ids."""
    data_tofu = import_memadapt_data()
    ood_rows = []
    for i, q in enumerate(OOD_PROBE_QUESTIONS):
        item = data_tofu.preprocess_chat_instance(tokenizer, q,
                                                  "I am not sure.")
        item["index"] = -1 - i
        item["source_ids"] = NO_AUTHOR
        ood_rows.append(item)
    ood = collator(ood_rows)
    if phase == "phase0":
        return None, ood
    n_probe = min(8, len(subset_ids))
    step = max(1, len(subset_ids) // n_probe)
    probe_authors = list(subset_ids)[::step][:n_probe]
    rows = []
    for a in probe_authors:
        rows += [full_dataset[a * RECORDS_PER_AUTHOR],
                 full_dataset[a * RECORDS_PER_AUTHOR + 1]]
    return collator(rows), ood


# ---------------------------------------------------------------------------
# Gradient-structure check (belt c)
# ---------------------------------------------------------------------------

def debug_grad_check(model, tc: BlockTranscoder, state: TcState, phase: str,
                     author_batch=None, generic_batch=None):
    """Per-(phase x batch-type) gradient-structure check on REAL collated
    batches: one backward per pass (sound at ANY batch size - the 1/ga
    scaling is compute_loss's separate concern, sepmlp trap 2), then
    EXACT-zero asserts on every forbidden slice plus a non-vacuity assert on
    a slice that must be reached.

    Non-vacuity is asserted on the DECODER slice for LM passes: at a phase's
    step 0 the relevant decoder columns are exactly zero, so the encoder
    grad (which flows THROUGH the decoder) is legitimately zero - the
    decoder grad (proportional to the activation values) is the always-live
    signal. Passes:
      phase 0                : LM on a generic batch  -> shared slices only;
      phase 1, author batch  : LM                     -> block-k slices only
                               (shared AND every other author exactly zero);
      phase 1, generic batch : LM                     -> exactly zero
                               EVERYWHERE (this is what lets compute_loss
                               discard the generic 1-token LM term);
                               suppression            -> author W_enc/b_enc
                               rows only (all W_dec + shared exactly zero).
    """
    assert state.phase == phase, (state.phase, phase)
    device = next(model.parameters()).device
    S, Fdim = tc.shared_start, tc.n_features
    was_training = model.training
    model.train()

    def _prep(batch):
        b = {k: v.to(device) for k, v in batch.items()}
        sid = b.pop("source_ids")
        b.pop("index", None)
        qmask = (b["labels"] == IGNORE_INDEX) & b["attention_mask"].bool()
        return b, sid, qmask

    def _touched():
        ge, gb, gd = tc.W_enc.grad, tc.b_enc.grad, tc.W_dec.grad
        enc = (set() if ge is None else
               set(ge.abs().sum(dim=1).nonzero(as_tuple=True)[0].tolist()))
        bias = (set() if gb is None else
                set(gb.abs().nonzero(as_tuple=True)[0].tolist()))
        dec = (set() if gd is None else
               set(gd.abs().sum(dim=(0, 1)).nonzero(as_tuple=True)[0].tolist()))
        return enc, bias, dec

    def _lm_backward(b, sid, qmask):
        model.zero_grad(set_to_none=True)
        state.set_batch(sid, question_mask=qmask,
                        attention_mask=b["attention_mask"])
        try:
            out = model(**b)
            assert torch.isfinite(out.loss), "non-finite LM loss in grad check"
            out.loss.backward()
        finally:
            state.clear()

    def _assert_confined(tag, allowed: set):
        enc, bias, dec = _touched()
        for name, got in (("W_enc rows", enc), ("b_enc entries", bias),
                          ("W_dec cols", dec)):
            bad = got - allowed
            assert not bad, (
                f"[grad-check:{tag}] {name} gradient outside the allowed "
                f"slice (features {sorted(bad)[:8]}...) - exactness broken"
            )
        return enc, bias, dec

    if phase == "phase0":
        assert generic_batch is not None, "phase-0 check needs the pool batch"
        b, sid, qm = _prep(generic_batch)
        assert bool((sid == NO_AUTHOR).all())
        _lm_backward(b, sid, qm)
        shared = set(range(S, Fdim))
        _, _, dec = _assert_confined("phase0-lm", shared)
        # Explicit exact-zero restatement of the forbidden region (belt on
        # the set math above) + non-vacuity on the decoder.
        assert tc.W_enc.grad is None or tc.W_enc.grad[:S].abs().sum() == 0
        assert tc.b_enc.grad is None or tc.b_enc.grad[:S].abs().sum() == 0
        assert tc.W_dec.grad is None or tc.W_dec.grad[..., :S].abs().sum() == 0
        assert dec & shared, (
            "[grad-check:phase0-lm] no gradient reached the shared decoder "
            "cols - the LM path is detached from the transcoder"
        )
        print("[grad-check:phase0-lm] structure OK (shared slices only)")
    else:
        assert author_batch is not None and generic_batch is not None
        # -- author batch: LM confined to block k ---------------------------
        b, sid, qm = _prep(author_batch)
        uniq = sid.unique()
        assert uniq.numel() == 1 and int(uniq.item()) != NO_AUTHOR, (
            "author grad-check batch must be single-source"
        )
        hits = (tc.author_ids.cpu()
                == int(uniq.item())).nonzero(as_tuple=True)[0]
        assert hits.numel() == 1, f"author {int(uniq.item())} has no block"
        slot = int(hits.item())
        own = set(range(slot * tc.m_author, (slot + 1) * tc.m_author))
        _lm_backward(b, sid, qm)
        _, _, dec = _assert_confined(f"phase1-author{int(uniq.item())}-lm", own)
        assert tc.W_enc.grad is None or tc.W_enc.grad[S:].abs().sum() == 0, (
            "[grad-check:phase1-author-lm] shared W_enc rows got gradient"
        )
        assert tc.b_enc.grad is None or tc.b_enc.grad[S:].abs().sum() == 0
        assert tc.W_dec.grad is None or tc.W_dec.grad[..., S:].abs().sum() == 0
        assert dec & own, (
            "[grad-check:phase1-author-lm] no gradient reached the block's "
            "decoder cols - routing is detached"
        )
        print(f"[grad-check:phase1-author-lm] structure OK "
              f"(block of author {int(uniq.item())} only)")

        # -- generic batch, LM: exactly zero everywhere ---------------------
        b, sid, qm = _prep(generic_batch)
        assert bool((sid == NO_AUTHOR).all())
        _lm_backward(b, sid, qm)
        for name in ("W_enc", "b_enc", "W_dec"):
            grad = getattr(tc, name).grad
            assert grad is None or grad.abs().sum() == 0, (
                f"[grad-check:phase1-generic-lm] {name} got LM gradient on a "
                "NO_AUTHOR batch - the empty own-mask leaked"
            )
        print("[grad-check:phase1-generic-lm] LM gradient exactly zero "
              "everywhere (empty own-mask)")

        # -- generic batch, suppression: author enc rows only ---------------
        b, sid, qm = _prep(generic_batch)
        model.zero_grad(set_to_none=True)
        state.set_batch(sid, question_mask=qm,
                        attention_mask=b["attention_mask"])
        state.begin_suppression()
        try:
            model(input_ids=b["input_ids"],
                  attention_mask=b["attention_mask"])
            terms = state.end_suppression()
        finally:
            state.clear()
        assert len(terms) == 1, f"{len(terms)} suppression terms collected"
        terms[0].backward()
        author_rows = set(range(S))
        enc, bias, dec = _assert_confined("phase1-generic-supp", author_rows)
        assert not dec and (tc.W_dec.grad is None
                            or tc.W_dec.grad.abs().sum() == 0), (
            "[grad-check:phase1-generic-supp] suppression leaked into W_dec"
        )
        assert tc.W_enc.grad is None or tc.W_enc.grad[S:].abs().sum() == 0
        assert tc.b_enc.grad is None or tc.b_enc.grad[S:].abs().sum() == 0
        assert enc, (
            "[grad-check:phase1-generic-supp] no author feature fired on the "
            "generic batch - the suppression path is vacuous"
        )
        print("[grad-check:phase1-generic-supp] structure OK "
              "(author W_enc/b_enc rows only)")

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()


# ---------------------------------------------------------------------------
# Detector-init glue + phase-0 pristine belt
# ---------------------------------------------------------------------------

def expand_detector_arrays(tc: BlockTranscoder, subset_ids, mean, counts):
    """Scatter the subset pre-pass output into full-K arrays keyed by SLOT
    (author_ids position, never the global id - ids can be non-contiguous,
    e.g. the tiny test fixture). Blocks outside the subset keep counts 0, so
    apply_detector_init leaves their seeded rows untouched."""
    K = tc.num_authors
    mean_full = np.zeros((K, tc.hidden), dtype=np.float32)
    counts_full = np.zeros((K,), dtype=np.float64)
    ids = tc.author_ids.detach().cpu().tolist()
    for i, a in enumerate(subset_ids):
        assert int(a) in ids, f"subset author {a} has no block"
        slot = ids.index(int(a))
        mean_full[slot] = mean[i]
        counts_full[slot] = counts[i]
    return mean_full, counts_full


def assert_authors_pristine(tc: BlockTranscoder):
    """Phase-0 mirror of assert_shared_frozen (belt d): the author slices
    must be BITWISE the fresh seeded init - phase 0 applies no detector init
    and its gradients never reach author rows (belts a+b), so any drift here
    means both belts failed. Compares against a freshly-built twin (cheap:
    CPU init + three slice compares)."""
    ref = BlockTranscoder(
        hidden=tc.hidden, m_author=tc.m_author, m_shared=tc.m_shared,
        author_ids=tc.author_ids.detach().cpu(),
        insert_layer=tc.insert_layer, span=tc.span, init_seed=tc.init_seed,
    )
    S = tc.shared_start
    pairs = [
        ("W_enc[author rows]", tc.W_enc.detach().cpu().float()[:S],
         ref.W_enc.detach()[:S]),
        ("b_enc[author]", tc.b_enc.detach().cpu().float()[:S],
         ref.b_enc.detach()[:S]),
        ("W_dec[:, :, author cols]", tc.W_dec.detach().cpu().float()[..., :S],
         ref.W_dec.detach()[..., :S]),
    ]
    for name, live, fresh in pairs:
        assert torch.equal(live, fresh), (
            f"author slices drifted in phase 0: {name} != seeded init"
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def run_training(cfg, args):
    validate_config(cfg)
    set_determinism(cfg["seed"])
    phase = cfg["phase"]
    run_dir = os.path.join(STORAGE_ROOT, "runs", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    if torch.cuda.is_available():
        # Per-phase peak (the smoke runs both phases in one process).
        torch.cuda.reset_peak_memory_stats()

    data_tofu = import_memadapt_data()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    collator = data_tofu.QACollatorWithSources(tokenizer)
    guarded = never_train_questions()
    subset_ids = resolve_subset(cfg)

    full_dataset = None
    alt_sampler = None
    if phase == "phase0":
        assert 3 * cfg["alpaca_n"] <= ALPACA_TRAIN_HEAD, (
            "phase-0 raw Alpaca draw must stay inside the training head - "
            "shrink alpaca_n (or change ALPACA_TRAIN_HEAD everywhere at once)"
        )
        pool = PoolDataset(tokenizer, cfg["alpaca_n"], cfg["seed"],
                           cfg["max_length"], guarded, mask_labels=False,
                           alpaca_skip=0, what="phase-0 author-free pool")
        print(f"[data] phase-0 pool: {pool.n_alpaca} alpaca + "
              f"{pool.n_real_authors} real_authors rows "
              "(all NO_AUTHOR, real answer labels)")
        train_ds = pool
        generic_pool = pool
    else:
        full_dataset = data_tofu.TofuQADataset(
            tokenizer, split="full", max_length=cfg["max_length"])
        assert_never_train_clean(full_dataset.data["question"], guarded,
                                 "TOFU author training split")
        author_rows = [a * RECORDS_PER_AUTHOR + i
                       for a in subset_ids for i in range(RECORDS_PER_AUTHOR)]
        author_ds = Subset(full_dataset, author_rows)
        generic_pool = PoolDataset(
            tokenizer, cfg["alpaca_n"], cfg["seed"], cfg["max_length"],
            guarded, mask_labels=True, alpaca_skip=ALPACA_TRAIN_HEAD,
            what="phase-1 generic suppression pool")
        print(f"[data] phase-1: {len(subset_ids)} authors x "
              f"{RECORDS_PER_AUTHOR} rows; generic pool "
              f"{generic_pool.n_alpaca} alpaca (beyond head "
              f"{ALPACA_TRAIN_HEAD}) + {generic_pool.n_real_authors} "
              "real_authors rows")
        # Positions [j*RPP, (j+1)*RPP) of author_ds belong to subset author j
        # (Subset preserves author_rows order).
        groups = [list(range(j * RECORDS_PER_AUTHOR,
                             (j + 1) * RECORDS_PER_AUTHOR))
                  for j in range(len(subset_ids))]
        alt_sampler = AlternatingBatchSampler(groups, len(generic_pool),
                                              cfg["batch_size"], cfg["seed"])
        train_ds = ConcatDataset([author_ds, generic_pool])

    # Base model at its FINAL dtype BEFORE any transcoder work (tc_model
    # install-ordering contract: never model.to(dtype) after install).
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    # HF Trainer only places the model at construction time - AFTER the
    # detector pre-pass below, which must not run on CPU (sepmlp lesson).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    det_path = None
    if phase == "phase0":
        tc = BlockTranscoder(
            hidden=model.config.hidden_size, m_author=cfg["m_author"],
            m_shared=cfg["m_shared"], author_ids=list(range(cfg["n_authors"])),
            insert_layer=cfg["insert_layer"], span=cfg["span"],
            init_seed=cfg["seed"],
        )
        state = TcState()
        # Detector init deliberately SKIPPED in phase 0 (see module
        # docstring): phase 1 applies it on top of this checkpoint's rows.
    else:
        tc, adapter_cfg0, state, ck_phase = load_tc_from_checkpoint(
            cfg["phase0_checkpoint"])
        assert ck_phase == "phase0", (
            f"phase 1 must start from a phase-0 checkpoint, got {ck_phase!r}"
        )
        assert (tc.insert_layer, tc.span, tc.m_author, tc.m_shared,
                tc.num_authors) == (cfg["insert_layer"], cfg["span"],
                                    cfg["m_author"], cfg["m_shared"],
                                    cfg["n_authors"]), (
            "config/phase-0-checkpoint topology mismatch"
        )
        assert tc.hidden == model.config.hidden_size
        assert adapter_cfg0["init_seed"] == cfg["seed"], (
            "phase-0 init seed differs from this config's seed - the init "
            "provenance story (prefix-seeded rows) would lie"
        )
        p0 = cfg["phase0_checkpoint"]
        p0_dir = os.path.dirname(p0) if p0.endswith(".pt") else p0
        assert os.path.realpath(p0_dir) != os.path.realpath(run_dir), (
            "phase-1 run_name resolves to the phase-0 run dir - saving would "
            "clobber the shared phase-0 checkpoint"
        )
        if cfg["detector_init"] == "questions":
            # BEFORE install_tc: the pre-pass hooks the RAW mlp input
            # (tc_model asserts this too). Subset authors only - the other
            # blocks keep their seeded rows (they are never LM-trained;
            # suppression still reaches them, which is legal on author-free
            # data).
            mean, counts, det_path = detector_init_cached(
                run_dir, model, full_dataset, collator, device,
                cfg["batch_size"], subset_ids, cfg["insert_layer"])
            mean_full, counts_full = expand_detector_arrays(
                tc, subset_ids, mean, counts)
            apply_detector_init(tc, mean_full, counts_full,
                                float(cfg["init_scale"]))

    state.set_phase(phase)
    tc.to(device=device, dtype=torch.float32)  # fp32 masters, bf16 autocast
    install_tc(model, tc, state)
    freeze_base(model, tc)

    targs = TrainingArguments(
        output_dir=os.path.join(run_dir, "hf_trainer"),
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        # SS7 fixed defaults - deliberately NOT config keys (closed schema):
        # AdamW / cosine / no warmup; wd=0 and max_grad_norm=0 are
        # exactness-critical (asserted again in BlockTcTrainer.__init__).
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=0.0,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    probe, ood = build_probe_batches(full_dataset, collator, tokenizer,
                                     subset_ids, phase)
    telemetry = TcTelemetry(model, state, probe, ood)

    # Fresh Trainer => fresh AdamW: phase 1 never carries phase-0 moments
    # (DESIGN SS3 - Adam moments are data-functions) and no run ever passes
    # resume_from_checkpoint.
    trainer = BlockTcTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        tc=tc,
        tc_state=state,
        phase=phase,
        lambda_max=float(cfg["lambda_max"]),
        lambda_warmup_frac=float(cfg["lambda_warmup_frac"]),
        clip_norm=float(cfg["clip_norm"]),
        alt_sampler=alt_sampler,
        callbacks=[telemetry],
    )

    if args.debug_grad_checks:
        bs = cfg["batch_size"]
        gb = collator([generic_pool[i]
                       for i in range(min(len(generic_pool), bs))])
        if phase == "phase0":
            debug_grad_check(model, tc, state, phase, generic_batch=gb)
        else:
            a0 = subset_ids[0]
            ab = collator([full_dataset[a0 * RECORDS_PER_AUTHOR + i]
                           for i in range(min(RECORDS_PER_AUTHOR, bs))])
            debug_grad_check(model, tc, state, phase,
                             author_batch=ab, generic_batch=gb)

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    # Silent-failure sweep: a NaN that reached the log is a dead run.
    for rec in trainer.state.log_history:
        if "loss" in rec:
            assert math.isfinite(rec["loss"]), (
                f"non-finite training loss logged: {rec}"
            )
    seen = trainer.seen_sources
    if phase == "phase0":
        assert seen == {NO_AUTHOR}, (
            f"phase-0 saw author rows: {sorted(seen)[:8]} - pool or sampler "
            "plumbing broken"
        )
    else:
        real_seen = {s for s in seen if s != NO_AUTHOR}
        subset = set(subset_ids)
        # Plumbing soundness (holds at ANY step count): every author id that
        # reached the module must belong to the trained subset. A foreign id
        # means source_ids are mis-plumbed - this is the real invariant and is
        # a strictly stronger plumbing check than the old equality.
        assert real_seen <= subset, (
            "foreign author ids in source_ids: "
            f"{sorted(real_seen - subset)[:8]} - source_ids plumbing is broken"
        )
        # Full author coverage is only guaranteed once a full epoch has run.
        # A step-capped smoke (--smoke forces max_steps=5, far below one epoch;
        # the alternating sampler fits only ~floor(max_steps/2) single-author
        # batches) legitimately visits a subset of the authors - not a bug.
        full_epoch = (trainer.state.epoch is not None
                      and trainer.state.epoch >= 1.0 - 1e-9)
        if full_epoch:
            assert real_seen == subset, (
                f"distinct-author guard: saw {len(real_seen)} of "
                f"{len(subset_ids)} authors in a full epoch - sampler coverage "
                "broken"
            )
        else:
            assert real_seen, (
                "no author batches seen at all - sampler/plumbing broken"
            )
        assert NO_AUTHOR in seen, (
            "no generic batches seen - the alternating sampler is broken"
        )

    # Save-time belt (d), per phase - BEFORE writing the checkpoint.
    if phase == "phase1":
        assert_shared_frozen(tc, cfg["phase0_checkpoint"])
        print("[belt] shared block bitwise-frozen vs phase-0 checkpoint")
    else:
        assert_authors_pristine(tc)
        print("[belt] author blocks bitwise-pristine (seeded init) after phase 0")

    peak_gib = (torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available() else None)
    adapter_cfg = {
        "hidden": tc.hidden, "m_author": tc.m_author,
        "m_shared": tc.m_shared, "init_seed": cfg["seed"],
        "n_authors": tc.num_authors, "insert_layer": tc.insert_layer,
        "span": tc.span,
    }
    meta = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "config_sha256": file_sha256(cfg["_config_path"]),
        "script_sha256s": {os.path.basename(p): file_sha256(p)
                           for p in PROVENANCE_FILES},
        "argv": sys.argv,
        "slurm_job_id": slurm_job_id(),
        "seed": cfg["seed"],
        "phase": phase,
        "authors_subset": cfg["authors_subset"],
        "n_train_authors": (0 if phase == "phase0" else len(subset_ids)),
        "data": {"alpaca": generic_pool.n_alpaca,
                 "real_authors": generic_pool.n_real_authors,
                 "alpaca_skip": (0 if phase == "phase0"
                                 else ALPACA_TRAIN_HEAD)},
        "detector_init": (None if phase == "phase0" else cfg["detector_init"]),
        "detector_init_cache": det_path,
        "detector_init_cache_sha256": (file_sha256(det_path)
                                       if det_path else None),
        "lambda": {"lambda_max": cfg["lambda_max"],
                   "warmup_frac": cfg["lambda_warmup_frac"],
                   "max_steps": int(trainer.state.max_steps)},
        "train_wall_seconds": wall,
        "log_history": trainer.state.log_history,
        "loss_components_tail": trainer.loss_components[-50:],
        "telemetry": telemetry.history,
        "peak_mem_gib": peak_gib,
        "torch_version": torch.__version__,
    }
    save_checkpoint(tc, adapter_cfg, run_dir, phase, extra_meta=meta)
    if peak_gib is not None:
        print(f"[mem] max_memory_allocated={peak_gib:.2f} GiB")
    print(f"[done] phase={phase} wall={wall:.1f}s checkpoint={run_dir}")
    return run_dir, tc, model, state, trainer


def run_smoke(cfg, args):
    """DESIGN SS7 smoke: phase 0 THEN phase 1 in one process on the
    FULL-SIZE transcoder with subset-limited data (~5 steps each, grad
    checks forced on), then save -> reload -> bitwise + module-forward
    parity. Peak memory prints per phase inside run_training."""
    args.max_steps = args.max_steps or 5
    args.debug_grad_checks = 1
    base = copy.deepcopy(cfg)
    name = base["run_name"]

    cfg0 = copy.deepcopy(base)
    cfg0.update(phase="phase0", phase0_checkpoint=None,
                run_name=f"{name}_p0")
    dir0, tc0, model0, state0, trainer0 = run_training(cfg0, args)
    del trainer0, model0, state0, tc0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg1 = copy.deepcopy(base)
    cfg1.update(phase="phase1",
                phase0_checkpoint=os.path.join(dir0, "blocktc.pt"),
                run_name=f"{name}_p1")
    dir1, tc1, model1, state1, trainer1 = run_training(cfg1, args)

    # Suppression must actually have fired (dead-suppression guard; the
    # lambda warmup does not zero the RECORDED raw term).
    generic = [c for c in trainer1.loss_components if c["type"] == "generic"]
    assert generic and any(c["supp"] > 0 for c in generic), (
        "smoke: no nonzero suppression term on any generic batch"
    )

    print("[smoke] reload + parity check (live trained vs reloaded)")
    tc2, _, _, ph2 = load_tc_from_checkpoint(dir1)
    assert ph2 == "phase1", ph2
    for nm in ("W_enc", "b_enc", "W_dec"):
        live = getattr(tc1, nm).detach().cpu().float()
        assert torch.equal(live, getattr(tc2, nm).detach()), (
            f"smoke: reloaded {nm} differs bitwise from the trained tensor"
        )
    # Module-level forward parity on CPU (serving path, fresh states).
    tc_cpu = copy.deepcopy(tc1).cpu().eval()
    tc2 = tc2.eval()
    x = torch.randn(2, 8, tc_cpu.hidden,
                    generator=torch.Generator().manual_seed(cfg["seed"]))

    def _outs(module):
        st = TcState()
        with torch.no_grad():
            module.encode(x, st)
            return torch.stack([module.decode(j, x, st)
                                for j in range(module.span)])

    assert torch.equal(_outs(tc_cpu), _outs(tc2)), (
        "smoke: reloaded module forward differs from the live one"
    )
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser(
        description="blocktc two-phase trainer (config-driven; DESIGN SS7)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="phase 0 then phase 1, ~5 steps each, parity checks")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="0 = use config epochs (smoke forces 5)")
    ap.add_argument("--debug_grad_checks", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", HF_HOME)

    if args.smoke:
        run_smoke(cfg, args)
    else:
        run_training(cfg, args)


if __name__ == "__main__":
    main()
