"""Gradient-routed training of the per-author MLP banks (--smoke = micro gate).

Recipe (user spec, superseding the SwiGLU lambda-grid scaffold): 15 epochs,
AdamW, effective batch 32, COSINE LR decay, no warmup, weight_decay 0 (nonzero
wd would decay idle authors' slices — an untargeted forgetting channel;
explicitly NOT inherited from OU), PER-AUTHOR gradient clipping at 1.0, bf16
autocast over fp32 bank masters, seed 42. Data pipeline is the OU-parity
memadapt one (imported, never copied), so the model is comparable with the
released Finetuned checkpoint and the MemAdapt repro.

Loss:  total = L1 + w2*L2 + w3*L3 + w4*L4   (defaults w2=10, w3=50, w4=1)
  L1  CE on answer tokens; gradient routed to the row's own branch only via
      the bitwise detach construction in bank_layer.py.
  L2  hinge (margin 2): every OTHER branch's detector pre-activations must sit
      >= margin below the ReLU threshold on the batch's live tokens.
  L3  exact OTHER-branch output-norm penalty (Gram trick), mean over
      off-entries x non-pad tokens x layers (weights K- and depth-invariant).
  L4  promotion (delta 0.1): >=1 own detector must fire on the row's own
      QUESTION tokens (dead-ReLU rescue).
Batches ALTERNATE author rows (L1+L2+L3+L4) with pure-negative rows
(L2+L3 over ALL branches): negatives = public Alpaca + TOFU real_authors (the
never-trained-author source). The never-train control split is guarded out of
every training pool by a set-membership assert (sepmlp_common helpers; a
static CPU gate keeps this file from even naming that split).

Detector init: a frozen-base pre-pass caches per-author per-layer mean
MLP-input hidden states over the author's question tokens (one batched
forward over the 20*K questions, npz-cached in the run dir, deterministic);
W_gate rows start as init_scale * that unit direction (config
detector_init: questions|random, init_scale default 1.0), b_gate 0.

--smoke: end-to-end micro run on 1 GPU at FULL bank size (2 authors' data,
5 steps, batch 2 -> save -> reload -> bitwise parity), printing
max_memory_allocated. Gate before any full submission.
"""

import argparse
import copy
import json
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

from bank_layer import AuthorBank, BankState
from sepmlp_common import (
    NO_AUTHOR,
    RECORDS_PER_AUTHOR,
    assert_never_train_clean,
    file_sha256,
    import_memadapt_data,
    load_config,
    never_train_questions,
    set_determinism,
    slurm_job_id,
)
from sepmlp_model import SepMlpMLP, freeze_base, install_banks, save_checkpoint

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IGNORE_INDEX = -100  # fixed by the OU schema (== data_tofu.IGNORE_INDEX)

# Spec loss weights (approved plan Part 2): total = L1 + w2*L2 + w3*L3 + w4*L4.
DEFAULT_LOSS = {"w2": 10.0, "w3": 50.0, "w4": 1.0,
                "margin": 2.0, "promo_delta": 0.1}

# Fixed OOD telemetry probe: generic non-TOFU questions, hardcoded so the
# training loop has zero extra data dependencies and NEVER touches the
# held-out control split (which must stay pristine for the relearn control +
# MIA nonmembers; tests/test_data_pipeline.py statically asserts this file
# never names that split). The
# proper OOD eval (world_facts / real_authors / Alpaca) lives in
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


def resolve_loss_cfg(train_cfg: dict) -> dict:
    """Spec defaults, config-overridable via train.loss. A legacy config that
    carries suppress_lambda and NO loss block maps exactly onto the old
    objective (L1 + lambda*L3): w2=w4=0, w3=lambda — the retired SwiGLU arms
    keep their recorded semantics if ever re-run."""
    if "loss" not in train_cfg and "suppress_lambda" in train_cfg:
        legacy = dict(DEFAULT_LOSS)
        legacy.update(w2=0.0, w3=float(train_cfg["suppress_lambda"]), w4=0.0)
        return legacy
    out = dict(DEFAULT_LOSS)
    out.update(train_cfg.get("loss", {}))
    assert set(out) == set(DEFAULT_LOSS), f"unknown loss keys: {out}"
    return out


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------

def load_negative_pairs(alpaca_n: int, hf_home: str, seed: int):
    """Raw negative Q/A text pool: public Alpaca (3*alpaca_n drawn; length
    filtering happens at tokenize time) + ALL TOFU real_authors rows — the
    never-trained-author source the spec calls for. The never-train control
    split is NOT a legal source (guarded by the caller)."""
    if os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")) not in sys.path:
        sys.path.insert(0, os.environ.get("TOFU_SISA_LORA_DIR", os.path.join(_REPO_ROOT, "tofu_sisa_lora")))
    from skill_data import load_alpaca

    import datasets

    pairs = [{"question": r["question"], "answer": r["answer"],
              "source": "alpaca"}
             for r in load_alpaca(3 * alpaca_n, hf_home, seed=seed)]
    ra = datasets.load_dataset("locuslab/TOFU", name="real_authors",
                               split="train")
    ra_pairs = [{"question": q, "answer": a, "source": "real_authors"}
                for q, a in zip(ra["question"], ra["answer"])]
    assert len(ra_pairs) == 100, len(ra_pairs)
    return pairs, ra_pairs


class OODNegativesDataset(Dataset):
    """Pure-negative rows: public Alpaca + TOFU real_authors QA (the
    never-trained-author source; approved-plan decision 1). The standard
    training loop alternates author batches with batches drawn from here.

    Each row carries source_ids = NO_AUTHOR, so every author's branch is
    "off" for it: L2+L3 drive ALL branches to silence on this text. Labels
    are IGNORE_INDEX everywhere except the final token — bank gradients from
    the LM loss are exactly zero on NO_AUTHOR rows by the detach
    construction, and the single live label keeps the trainer's
    num_items_in_batch nonzero even on an all-negative batch (NaN guard).

    Guard: every question is membership-checked against the never-train
    split before use (guarded_questions=None loads the split itself).
    """

    def __init__(self, tokenizer, n: int, seed: int, hf_home: str,
                 max_length: int, guarded_questions=None):
        data_tofu = import_memadapt_data()
        alpaca_pairs, ra_pairs = load_negative_pairs(n, hf_home, seed)
        guarded = (guarded_questions if guarded_questions is not None
                   else never_train_questions())
        assert_never_train_clean(
            [p["question"] for p in alpaca_pairs + ra_pairs], guarded,
            "negative pool")

        def _tokenize(pair, pos):
            item = data_tofu.preprocess_chat_instance(
                tokenizer, pair["question"], pair["answer"]
            )
            if len(item["input_ids"]) >= max_length:
                return None
            labels = torch.full_like(item["labels"], data_tofu.IGNORE_INDEX)
            labels[-1] = item["input_ids"][-1]
            item["labels"] = labels
            item["index"] = -1000 - pos
            item["source_ids"] = NO_AUTHOR
            return item

        self.items = []
        n_alpaca = 0
        for pair in alpaca_pairs:
            if n_alpaca >= n:
                break
            item = _tokenize(pair, len(self.items))
            if item is not None:
                self.items.append(item)
                n_alpaca += 1
        assert n_alpaca == n, (
            f"only {n_alpaca}/{n} Alpaca rows fit max_length {max_length}"
        )
        n_ra = 0
        for pair in ra_pairs:
            item = _tokenize(pair, len(self.items))
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
    """Author-batch / pure-negatives-batch alternation over
    ConcatDataset([author_ds, negatives_ds]) (spec batch schedule).

    One "epoch" = one shuffled pass over the author rows interleaved with an
    EQUAL number of negative batches (negatives reshuffle-cycle to cover the
    demand). Deterministic: the generator is seeded from (seed, epoch
    counter) — the custom batch_sampler bypasses HF's per-epoch reseeding, so
    determinism is owned here.
    """

    def __init__(self, n_author: int, n_negative: int, batch_size: int,
                 seed: int):
        assert n_author > 0 and n_negative > 0 and batch_size > 0
        self.n_author = n_author
        self.n_negative = n_negative
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        self.n_author_batches = (n_author + batch_size - 1) // batch_size

    def __len__(self):
        return 2 * self.n_author_batches

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self._epoch)
        self._epoch += 1
        order = torch.randperm(self.n_author, generator=g).tolist()
        need = self.n_author_batches * self.batch_size
        neg = []
        while len(neg) < need:
            neg += (torch.randperm(self.n_negative, generator=g)
                    + self.n_author).tolist()  # offset into the ConcatDataset
        bs = self.batch_size
        for b in range(self.n_author_batches):
            yield order[b * bs:(b + 1) * bs]
            yield neg[b * bs:(b + 1) * bs]


# ---------------------------------------------------------------------------
# Per-author gradient clipping
# ---------------------------------------------------------------------------

def per_author_clip_(banks, max_norm: float, eps: float = 1e-6):
    """Clip each author's {gate, up, bias, down} gradient slice to max_norm,
    with the norm taken over that author's slices ACROSS ALL layers (one
    author = one clipping group; a spiking author cannot rescale anyone
    else). Runs as an optimizer step pre-hook, replacing the global clip.
    Returns the (K,) pre-clip norms for telemetry/tests."""
    any_bank = next(iter(banks.values()))
    K, D = any_bank.num_authors, any_bank.width
    dev = (any_bank.W_gate.grad.device if any_bank.W_gate.grad is not None
           else any_bank.W_gate.device)
    total = torch.zeros(K, device=dev, dtype=torch.float32)
    for bank in banks.values():
        assert bank.num_authors == K, "banks disagree on author count"
        if bank.W_gate.grad is not None:
            total += bank.W_gate.grad.view(K, D, -1).float().pow(2).sum(dim=(1, 2))
        if bank.W_up.grad is not None:
            total += bank.W_up.grad.view(K, D, -1).float().pow(2).sum(dim=(1, 2))
        if bank.b_gate.grad is not None:
            total += bank.b_gate.grad.view(K, D).float().pow(2).sum(dim=1)
        if bank.W_down.grad is not None:
            total += bank.W_down.grad.view(-1, K, D).float().pow(2).sum(dim=(0, 2))
    norms = total.sqrt()
    coef = (max_norm / (norms + eps)).clamp(max=1.0)
    for bank in banks.values():
        if bank.W_gate.grad is not None:
            bank.W_gate.grad.view(K, D, -1).mul_(
                coef.view(K, 1, 1).to(bank.W_gate.grad.dtype))
        if bank.W_up.grad is not None:
            bank.W_up.grad.view(K, D, -1).mul_(
                coef.view(K, 1, 1).to(bank.W_up.grad.dtype))
        if bank.b_gate.grad is not None:
            bank.b_gate.grad.view(K, D).mul_(
                coef.view(K, 1).to(bank.b_gate.grad.dtype))
        if bank.W_down.grad is not None:
            bank.W_down.grad.view(-1, K, D).mul_(
                coef.view(1, K, 1).to(bank.W_down.grad.dtype))
    return norms


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SepMlpTrainer(Trainer):
    """Routes per-sequence source ids + question mask to the banks, assembles
    total = L1 + w2*L2 + w3*L3 + w4*L4 from the banks' per-layer terms
    (single forward serves every term), and installs the per-author clip as
    an optimizer step pre-hook."""

    def __init__(self, *args, state: BankState = None, banks=None,
                 loss_cfg: dict = None, per_author_clip: float = 0.0,
                 alt_sampler: AlternatingBatchSampler = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank_state = state
        self.banks = banks
        self.loss_cfg = dict(loss_cfg or DEFAULT_LOSS)
        self.per_author_clip = float(per_author_clip)
        self.alt_sampler = alt_sampler
        self.loss_components = []
        self.seen_sources = set()
        self._clip_hook = None

    def get_train_dataloader(self):
        if self.alt_sampler is None:
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
        if self.per_author_clip > 0 and self._clip_hook is None:
            banks, clip = self.banks, self.per_author_clip

            def _pre_step_clip(optimizer, hook_args, hook_kwargs):
                per_author_clip_(banks, clip)

            # Hook on the raw torch optimizer: accelerate's wrapper calls its
            # .step(), which fires registered pre-hooks (bf16 => no scaler,
            # grads are unscaled here). args.max_grad_norm must be 0 in this
            # mode so HF's global clip never runs on top.
            assert self.args.max_grad_norm in (0, 0.0), (
                "per-author clip and HF global clip must not stack"
            )
            self._clip_hook = self.optimizer.register_step_pre_hook(_pre_step_clip)
        return opt

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        inputs = dict(inputs)
        source_ids = inputs.pop("source_ids")
        inputs.pop("index", None)
        self.seen_sources.update(source_ids.detach().cpu().tolist())
        attn = inputs["attention_mask"]
        # own tokens = ALL real tokens of the row (question + answer), the L4
        # promotion target per paper §3.2 ("source k's own tokens"). Firing the
        # own detector on answer tokens too — not the prompt only — keeps the
        # adapter active during generation; the earlier question-only mask
        # under-promoted at K=200 (huge recall tail). Detector-init still pools
        # question tokens (a separate concern, unchanged).
        own_tok = attn.bool()
        self.bank_state.set_batch(source_ids, attn, own_token_mask=own_tok)
        lc = self.loss_cfg
        collect = model.training and (lc["w2"] > 0 or lc["w3"] > 0 or lc["w4"] > 0)
        if collect:
            self.bank_state.begin_losses(lc["margin"], lc["promo_delta"])
        try:
            out = super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            loss = out[0] if return_outputs else out
            comp = {"call": len(self.loss_components),
                    "lm": float(loss.detach()),
                    "hinge": 0.0, "gram": 0.0, "promo": 0.0}
            if collect:
                terms = self.bank_state.loss_terms
                assert len(terms) == len(self.banks), (
                    f"collected {len(terms)} loss terms for {len(self.banks)} "
                    "bank layers — a bank forward was skipped"
                )
                l2 = torch.stack([t["hinge"] for t in terms]).mean()
                l3 = torch.stack([t["gram"] for t in terms]).mean()
                promos = [t["promo"] for t in terms if t["promo"] is not None]
                # pure-negative batches carry no own rows => no promotion term
                l4 = torch.stack(promos).mean() if promos else None
                extra = lc["w2"] * l2 + lc["w3"] * l3
                if l4 is not None:
                    extra = extra + lc["w4"] * l4
                # transformers 4.48 num_items_in_batch path SUMS micro-batch
                # losses across accumulation steps (no /ga afterwards); scale
                # the per-micro mean terms so the weights are ga-invariant.
                # At ga=1 this is a no-op.
                if num_items_in_batch is not None:
                    extra = extra / self.args.gradient_accumulation_steps
                loss = loss + extra
                comp.update(hinge=float(l2.detach()), gram=float(l3.detach()),
                            promo=float(l4.detach()) if l4 is not None else 0.0)
            self.loss_components.append(comp)
            if comp["call"] % 25 == 0:
                print(f"[loss] call={comp['call']} lm={comp['lm']:.4f} "
                      f"hinge={comp['hinge']:.6f} gram={comp['gram']:.6f} "
                      f"promo={comp['promo']:.6f}")
            return (loss, out[1]) if return_outputs else loss
        finally:
            self.bank_state.clear()


class BankTelemetry(TrainerCallback):
    """Epoch-end own/off/OOD firing norms on fixed probe batches — the
    localization health signal. off/own must FALL over epochs; ood must stay
    near 0 (or trigger the pre-registered Alpaca-negatives arm)."""

    def __init__(self, model, state: BankState, banks, probe_batch, ood_batch):
        self.model = model
        self.state = state
        self.banks = banks
        self.probe = probe_batch
        self.ood = ood_batch
        self.history = []

    @torch.no_grad()
    def _norms(self, batch):
        device = next(self.model.parameters()).device
        self.state.set_batch(
            batch["source_ids"].to(device), batch["attention_mask"].to(device)
        )
        self.state.begin_telemetry()
        was_training = self.model.training
        self.model.eval()
        try:
            self.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
            )
            stats = self.state.end_telemetry()
        finally:
            self.state.clear()
            if was_training:
                self.model.train()
        own_sum = sum(s["own_sum"] for s in stats)
        own_cnt = sum(s["own_cnt"] for s in stats)
        off_sum = sum(s["off_sum"] for s in stats)
        off_cnt = sum(s["off_cnt"] for s in stats)
        own_mean = own_sum / own_cnt.clamp_min(1)
        off_mean = off_sum / off_cnt.clamp_min(1)
        return own_mean, own_cnt, off_mean

    def on_epoch_end(self, args, state, control, **kwargs):
        own_mean, own_cnt, off_mean = self._norms(self.probe)
        _, _, ood_mean = self._norms(self.ood)
        has_own = own_cnt > 0
        ratio = own_mean[has_own] / off_mean[has_own].clamp_min(1e-12)
        rec = {
            "epoch": state.epoch,
            "own_norm_mean": float(own_mean[has_own].mean()) if has_own.any() else None,
            "off_norm_mean": float(off_mean.mean()),
            "ood_norm_mean": float(ood_mean.mean()),
            "onoff_ratio_median": float(ratio.median()) if has_own.any() else None,
            "ood_over_own": (
                float(ood_mean.mean() / own_mean[has_own].mean().clamp_min(1e-12))
                if has_own.any() else None
            ),
        }
        self.history.append(rec)
        print(f"[telemetry] {json.dumps(rec)}")


# ---------------------------------------------------------------------------
# Gradient-structure check
# ---------------------------------------------------------------------------

def debug_grad_check(model, banks, state: BankState, batch):
    """Three-pass gradient-structure check on a real collated batch:
      (i)   L1 (LM) only  -> grads ONLY in the batch authors' own slices
                             (all four tensors);
      (ii)  L2+L3 only    -> grads ONLY in off-author slices, and they DO
                             reach other branches (non-vacuity assert) — on a
                             SINGLE-author sub-batch, since for a mixed batch
                             the in-batch-negatives terms legitimately reach
                             the other batch authors' slices;
      (iii) L4 only       -> grads ONLY in the own W_gate/b_gate rows
                             (promotion never touches W_up/W_down).
    Sound at any batch size (single backward each, no accumulation). The
    (ii)/(iii) within-layer exactness relies on the banks computing loss
    terms from the DETACHED layer input (see bank_layer forward)."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in batch.items()}
    source_ids = inputs.pop("source_ids")
    inputs.pop("index", None)
    was_training = model.training
    model.train()
    own_tok_full = inputs["attention_mask"].bool()  # promotion fires on all own tokens (paper §3.2)

    def _slice_rows(bank, slots):
        D = bank.width
        return torch.cat([torch.arange(s * D, (s + 1) * D) for s in slots]) \
            if len(slots) else torch.empty(0, dtype=torch.long)

    def _check(tag, expect_own, src, tensors=("W_gate", "W_up", "b_gate", "W_down")):
        for l, bank in banks.items():
            slots = (
                torch.isin(bank.author_ids.cpu(), src.unique().cpu())
                .nonzero(as_tuple=True)[0].tolist()
            )
            own_rows = set(_slice_rows(bank, slots).tolist())
            for name in tensors:
                grad = getattr(bank, name).grad
                if grad is None:
                    continue
                if name == "W_down":
                    nz = grad.abs().sum(dim=0).nonzero(as_tuple=True)[0]
                elif name == "b_gate":
                    nz = grad.abs().nonzero(as_tuple=True)[0]
                else:
                    nz = grad.abs().sum(dim=1).nonzero(as_tuple=True)[0]
                touched = set(nz.cpu().tolist())
                bad = (touched - own_rows) if expect_own else (touched & own_rows)
                assert not bad, (
                    f"[grad-check:{tag}] layer {l} {name}: grads outside the "
                    f"expected {'own' if expect_own else 'off'} slices "
                    f"(rows {sorted(list(bad))[:8]}...)"
                )
        print(f"[grad-check:{tag}] structure OK across {len(banks)} layers")

    # (i) L1 only
    model.zero_grad(set_to_none=True)
    state.set_batch(source_ids, inputs["attention_mask"])
    out = model(**inputs)
    out.loss.backward()
    state.clear()
    _check("l1-only", expect_own=True, src=source_ids)

    # (ii) L2+L3 only — SINGLE-author sub-batch (see docstring)
    real = source_ids[source_ids != NO_AUTHOR]
    assert real.numel() > 0, "grad-check needs at least one authored row"
    rows = (source_ids == real[0]).nonzero(as_tuple=True)[0]
    sub = {k: v[rows] for k, v in inputs.items()}
    sub_sources = source_ids[rows]
    model.zero_grad(set_to_none=True)
    state.set_batch(sub_sources, sub["attention_mask"],
                    own_token_mask=own_tok_full[rows])
    state.begin_losses(DEFAULT_LOSS["margin"], DEFAULT_LOSS["promo_delta"])
    model(**{k: v for k, v in sub.items() if k != "labels"})
    terms = state.loss_terms
    assert len(terms) == len(banks)
    l2l3 = (torch.stack([t["hinge"] for t in terms]).mean()
            + torch.stack([t["gram"] for t in terms]).mean())
    l2l3.backward(retain_graph=True)
    _check("l2l3-only", expect_own=False, src=sub_sources)
    reached = any(
        bank.W_gate.grad is not None and bank.W_gate.grad.abs().sum() > 0
        for bank in banks.values()
    )
    assert reached, (
        "[grad-check:l2l3-only] no gradient reached any other branch — the "
        "penalty path is detached from the parameters"
    )

    # (iii) L4 only — reuses the same collected terms (graph retained above)
    promos = [t["promo"] for t in terms if t["promo"] is not None]
    assert promos, "[grad-check:l4-only] no promotion term on an author batch"
    model.zero_grad(set_to_none=True)
    torch.stack(promos).mean().backward()
    state.clear()
    _check("l4-only", expect_own=True, src=sub_sources,
           tensors=("W_gate", "b_gate"))
    for l, bank in banks.items():
        for name in ("W_up", "W_down"):
            grad = getattr(bank, name).grad
            assert grad is None or grad.abs().sum() == 0, (
                f"[grad-check:l4-only] promotion leaked into {name} at layer {l}"
            )
    print(f"[grad-check:l4-only] promotion confined to own gate/bias rows")

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()


# ---------------------------------------------------------------------------
# Detector init (frozen-base pre-pass)
# ---------------------------------------------------------------------------

def compute_detector_init(model, layer_idxs, batches, author_ids, device):
    """Frozen-base pre-pass: per-author per-layer mean MLP-INPUT hidden state
    over the author's QUESTION tokens (labels IGNORE and real — the exact
    mask L4 promotes on). Hooks the raw mlp modules' inputs, so it must run
    BEFORE install_banks; deterministic given a fixed batch order.
    Returns (mean_hidden [K, L, H] float32 np, counts [K] float64 np)."""
    ids = torch.as_tensor(list(author_ids), dtype=torch.long)
    K = int(ids.numel())
    layer_idxs = sorted(int(l) for l in layer_idxs)
    hidden = model.config.hidden_size
    captured = {}
    hooks = []
    for l in layer_idxs:
        mlp = model.model.layers[l].mlp
        assert not isinstance(mlp, SepMlpMLP), (
            "detector init must run before install_banks (needs raw mlp inputs)"
        )
        hooks.append(mlp.register_forward_pre_hook(
            lambda mod, args, _l=l: captured.__setitem__(_l, args[0])))
    sums = torch.zeros(K, len(layer_idxs), hidden, dtype=torch.float64)
    counts = torch.zeros(K, dtype=torch.float64)
    ids_dev = ids.to(device)
    try:
        with torch.no_grad():
            for batch in batches:
                sid = batch["source_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                qmask = ((batch["labels"].to(device) == IGNORE_INDEX)
                         & attn.bool()).float()
                onehot = (ids_dev.view(1, K) == sid.view(-1, 1)).float()
                model(input_ids=batch["input_ids"].to(device),
                      attention_mask=attn, use_cache=False)
                for i, l in enumerate(layer_idxs):
                    x = captured.pop(l).float()
                    sums[:, i] += torch.einsum(
                        "bk,bt,bth->kh", onehot, qmask, x).double().cpu()
                counts += (onehot * qmask.sum(dim=1, keepdim=True)) \
                    .sum(dim=0).double().cpu()
    finally:
        for h in hooks:
            h.remove()
    mean = (sums / counts.clamp_min(1).view(K, 1, 1)).float().numpy()
    return mean, counts.numpy()


def apply_detector_init(banks, mean_hidden, counts, init_scale: float):
    """W_gate rows := init_scale * unit-norm mean-hidden direction of the
    author's question tokens (all D rows share the direction; the symmetry is
    broken by the random W_up and the zero W_down). b_gate stays 0 per the
    spec. Authors with zero captured tokens keep their seeded random rows."""
    layers = sorted(banks.keys())
    n_init = 0
    for i, l in enumerate(layers):
        bank = banks[l]
        K, D, H = bank.num_authors, bank.width, bank.hidden
        mh = torch.as_tensor(mean_hidden[:, i, :],
                             dtype=bank.W_gate.dtype,
                             device=bank.W_gate.device)
        norms = mh.norm(dim=1)
        with torch.no_grad():
            wg = bank.W_gate.view(K, D, H)
            for j in range(K):
                if counts[j] <= 0 or float(norms[j]) == 0.0:
                    continue
                wg[j] = init_scale * mh[j] / norms[j]
                n_init += 1
    print(f"[detector-init] pointed {n_init} (author, layer) gate blocks "
          f"at question-mean directions (scale {init_scale})")


def detector_init_cached(run_dir, model, full_dataset, collator, device,
                         batch_size, author_ids, layer_idxs):
    """Cache wrapper: <run_dir>/detector_init.npz holds the pre-pass output
    (written EARLY, reused on requeue; same seed => bit-equal content)."""
    path = os.path.join(run_dir, "detector_init.npz")
    layer_idxs = sorted(int(l) for l in layer_idxs)
    if os.path.exists(path):
        z = np.load(path)
        if (z["author_ids"].tolist() == list(author_ids)
                and z["layers"].tolist() == layer_idxs):
            print(f"[detector-init] reusing cache {path}")
            return z["mean_hidden"], z["counts"], path
        print(f"[detector-init] cache mismatch, recomputing: {path}")
    rows = [full_dataset[a * RECORDS_PER_AUTHOR + i]
            for a in author_ids for i in range(RECORDS_PER_AUTHOR)]
    batches = (collator(rows[s:s + batch_size])
               for s in range(0, len(rows), batch_size))
    t0 = time.perf_counter()
    mean, counts = compute_detector_init(model, layer_idxs, batches,
                                         author_ids, device)
    assert (counts > 0).all(), "an author captured zero question tokens"
    np.savez_compressed(path, mean_hidden=mean, counts=counts,
                        author_ids=np.asarray(list(author_ids)),
                        layers=np.asarray(layer_idxs))
    print(f"[detector-init] {len(rows)} question rows -> {path} "
          f"({time.perf_counter() - t0:.1f}s)")
    return mean, counts, path


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_probe_batches(dataset, collator, tokenizer, banks, n_authors=8):
    """Fixed probe batch (2 rows each from up to n_authors spread bank authors)
    + a fixed OOD batch (hardcoded generic questions, NO_AUTHOR ids)."""
    any_bank = next(iter(banks.values()))
    ids = any_bank.author_ids.tolist()
    step = max(1, len(ids) // n_authors)
    probe_authors = ids[::step][:n_authors]
    rows = []
    for a in probe_authors:
        rows += [dataset[a * RECORDS_PER_AUTHOR], dataset[a * RECORDS_PER_AUTHOR + 1]]
    probe = collator(rows)

    data_tofu = import_memadapt_data()
    ood_rows = []
    for i, q in enumerate(OOD_PROBE_QUESTIONS):
        item = data_tofu.preprocess_chat_instance(tokenizer, q, "I am not sure.")
        item["index"] = -1 - i
        item["source_ids"] = NO_AUTHOR
        ood_rows.append(item)
    ood = collator(ood_rows)
    return probe, ood


def run_training(cfg, args):
    set_determinism(cfg["seed"])
    run_dir = cfg["output_dir"]
    os.makedirs(run_dir, exist_ok=True)

    data_tofu = import_memadapt_data()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    full_dataset = data_tofu.TofuQADataset(
        tokenizer, split=cfg["data"]["split"], max_length=cfg["data"]["max_length"]
    )
    guarded = never_train_questions()
    assert_never_train_clean(full_dataset.data["question"], guarded,
                             "author training split")
    limit_authors = cfg["data"].get("limit_authors")
    dataset = (
        Subset(full_dataset, range(limit_authors * RECORDS_PER_AUTHOR))
        if limit_authors else full_dataset
    )
    assert "ood_negatives" not in cfg["data"], (
        "retired key ood_negatives: use data.negatives "
        "(alpaca + real_authors, alternating batches)"
    )
    # Negatives are STANDARD (spec): alternate author / pure-negative batches.
    neg_cfg = cfg["data"].get("negatives") or {"alpaca_n": 2000, "seed": 42}
    neg_ds = OODNegativesDataset(
        tokenizer, n=neg_cfg.get("alpaca_n", 2000),
        seed=neg_cfg.get("seed", 42), hf_home=cfg["hf_home"],
        max_length=cfg["data"]["max_length"], guarded_questions=guarded,
    )
    print(f"[data] negatives: {neg_ds.n_alpaca} alpaca + "
          f"{neg_ds.n_real_authors} real_authors rows (NO_AUTHOR)")
    t = cfg["train"]
    alt_sampler = AlternatingBatchSampler(
        len(dataset), len(neg_ds), t["batch_size"], cfg["seed"])
    train_ds = ConcatDataset([dataset, neg_ds])

    # Model first, banks after the detector pre-pass (needs raw mlp inputs).
    a = cfg["adapter"]
    author_ids = list(range(a["num_authors"]))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # HF Trainer only places the model at construction time — AFTER the
    # detector pre-pass below, which would otherwise run all 20*K frozen-base
    # forwards on login-grade CPUs and eat the SLURM walltime before step 0.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collator = data_tofu.QACollatorWithSources(tokenizer)

    det_mode = t.get("detector_init", "questions")
    assert det_mode in ("questions", "random"), det_mode
    det_path = None
    mean_hidden = counts = None
    if det_mode == "questions":
        mean_hidden, counts, det_path = detector_init_cached(
            run_dir, model, full_dataset, collator, device,
            t["batch_size"], author_ids, a["layers"])

    state = BankState()
    banks = {
        int(l): AuthorBank(
            hidden=a["hidden"], width=a["width"], author_ids=author_ids,
            layer_idx=int(l), init_seed=a["init_seed"],
            init_std=a.get("init_std"),
            penalty_form=a.get("penalty_form", "output_gram"),
            gate_act=a.get("gate_act", "relu"),
        )
        for l in a["layers"]
    }
    for bank in banks.values():
        bank.to(device=device, dtype=torch.float32)  # fp32 masters, bf16 autocast
    if mean_hidden is not None:
        apply_detector_init(banks, mean_hidden, counts,
                            float(t.get("init_scale", 1.0)))
    install_banks(model, banks, state)
    freeze_base(model, banks)

    loss_cfg = resolve_loss_cfg(t)
    clip_mode = t.get("clip_mode", "per_author")
    assert clip_mode in ("per_author", "global"), clip_mode
    targs = TrainingArguments(
        output_dir=os.path.join(run_dir, "hf_trainer"),
        num_train_epochs=t["epochs"],
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=t["lr"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        optim=t["optim"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        # per-author clipping runs as an optimizer pre-step hook instead of
        # HF's global clip (which must then be disabled: 0).
        max_grad_norm=(t["max_grad_norm"] if clip_mode == "global" else 0.0),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    probe, ood = build_probe_batches(
        full_dataset, collator, tokenizer, banks,
        n_authors=min(8, cfg["adapter"]["num_authors"]),
    )
    telemetry = BankTelemetry(model, state, banks, probe, ood)

    trainer = SepMlpTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        state=state,
        banks=banks,
        loss_cfg=loss_cfg,
        per_author_clip=(t["max_grad_norm"] if clip_mode == "per_author" else 0.0),
        alt_sampler=alt_sampler,
        callbacks=[telemetry],
    )

    if args.debug_grad_checks:
        first = collator([dataset[i] for i in range(min(len(dataset),
                                                       t["batch_size"]))])
        debug_grad_check(model, banks, state, first)

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    n_expected = (limit_authors or cfg["adapter"]["num_authors"])
    n_seen = len({s for s in trainer.seen_sources if s >= 0})  # NO_AUTHOR excluded
    assert n_seen == n_expected, (
        f"distinct-author guard: saw {n_seen} authors, expected {n_expected} — "
        "source_ids plumbing is broken"
    )
    assert NO_AUTHOR in trainer.seen_sources, (
        "negative-batch guard: no NO_AUTHOR rows seen — the alternating "
        "sampler is broken"
    )

    meta = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "config_sha256": file_sha256(cfg["_config_path"]),
        "loss_cfg": loss_cfg,
        "clip_mode": clip_mode,
        "negatives": {"alpaca": neg_ds.n_alpaca,
                      "real_authors": neg_ds.n_real_authors},
        "detector_init": det_mode,
        "detector_init_cache": det_path,
        "detector_init_cache_sha256": file_sha256(det_path) if det_path else None,
        "train_wall_seconds": wall,
        "log_history": trainer.state.log_history,
        "loss_components_tail": trainer.loss_components[-50:],
        "bank_telemetry": telemetry.history,
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "seed": cfg["seed"],
        "torch_version": torch.__version__,
    }
    save_checkpoint(banks, dict(cfg["adapter"]), run_dir, extra_meta=meta)
    if torch.cuda.is_available():
        print(f"[mem] max_memory_allocated="
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"[done] wall={wall:.1f}s checkpoint={run_dir}")
    return run_dir, banks, model, state


def run_smoke(cfg, args):
    """Full-size bank, 2 authors' data, 5 steps -> save -> reload -> parity."""
    from sepmlp_model import load_banks_from_checkpoint

    cfg = copy.deepcopy(cfg)
    cfg["data"]["limit_authors"] = 2
    cfg["output_dir"] = cfg["output_dir"].rstrip("/") + "_smoke"
    cfg["train"]["batch_size"] = 2
    args.max_steps = 5
    args.debug_grad_checks = 1

    run_dir, banks_live, model, state = run_training(cfg, args)

    print("[smoke] reload + forward parity check (live trained vs reloaded)")
    banks2, _, _ = load_banks_from_checkpoint(run_dir)
    l0 = sorted(banks_live.keys())[0]
    device = next(banks_live[l0].parameters()).device
    x = torch.randn(2, 16, cfg["adapter"]["hidden"], device=device)
    b2 = banks2[l0].to(device)
    banks_live[l0].eval(), b2.eval()
    with torch.no_grad():
        assert torch.equal(banks_live[l0](x, None), b2(x, None)), (
            "reloaded checkpoint does not reproduce the trained bank"
        )
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--debug_grad_checks", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])

    if args.smoke:
        run_smoke(cfg, args)
    else:
        run_training(cfg, args)


if __name__ == "__main__":
    main()
