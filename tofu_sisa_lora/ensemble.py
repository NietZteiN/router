"""Prediction-level ensemble over shard adapters (SISA/S3T 'aggregate decision').

`EnsembleModel` wraps the multi-adapter PeftModel and averages the constituents'
per-token output distributions at inference — the generative translation of
SISA's majority-vote / averaged-output-vectors aggregation (no weight merging).

Modes:
  probs  — arithmetic mean of per-token softmax (log-space: logsumexp - log n).
           The paper-faithful variant; preserves a single knowledgeable
           constituent's sharp distribution.
  logits — mean of logits (≈ geometric mean of probs). Ablation; expected to
           dilute like weight merging.

Drop-in contract (same as router.RoutedModel): nn.Module with .config,
.set_adapter() no-op, forward(...) -> CausalLMOutputWithPast, .generate().
eval_tofu only consumes `.loss` and `.generate()`.

FOOTGUN: in probs mode the returned `.logits` field holds ensemble LOG-PROBS,
not raw logits (they are the quantity whose NLL is the true ensemble loss).

Fast path: peft >= 0.14 mixed-batch `adapter_names` turns the n_eff sequential
adapter passes into one batched forward/generate; a runtime probe falls back to
sequential set_adapter loops (bit-equal results, ~n_eff x slower).
"""
import math
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithPast

_ENSEMBLE_RE = re.compile(r"^ensemble_(probs|logits)(?:_no(\d+))?$")


def parse_ensemble_label(label):
    """'ensemble_probs' -> ('probs', frozenset()); 'ensemble_logits_no9' -> ('logits', {9})."""
    m = _ENSEMBLE_RE.match(label)
    if not m:
        raise ValueError(
            f"Bad ensemble label {label!r}: expected ensemble_{{probs|logits}}[_no{{i}}]"
        )
    exclude = frozenset({int(m.group(2))}) if m.group(2) is not None else frozenset()
    return m.group(1), exclude


def discover_ensemble_adapters(model, exclude=frozenset(), output_dir=None):
    """Sorted shard_{i} adapter names loaded in the model, minus `exclude` ids.

    With output_dir, cross-checks against the shard_*/adapter_config.json dirs on
    disk — load_all_shard_adapters skips missing dirs SILENTLY, and an ensemble
    over a partial constituent set would silently corrupt SISA/S3T semantics.
    """
    loaded = {}
    for name in model.peft_config:
        m = re.fullmatch(r"shard_(\d+)", name)
        if m:
            loaded[int(m.group(1))] = name
    if output_dir is not None:
        import glob
        import os
        on_disk = set()
        for p in glob.glob(os.path.join(output_dir, "shard_*", "adapter_config.json")):
            m = re.fullmatch(r"shard_(\d+)", os.path.basename(os.path.dirname(p)))
            if m:
                on_disk.add(int(m.group(1)))
        if on_disk != set(loaded):
            raise RuntimeError(
                f"ensemble constituents mismatch: loaded shards {sorted(loaded)} vs "
                f"on-disk shard dirs {sorted(on_disk)} in {output_dir}"
            )
    kept = [loaded[i] for i in sorted(loaded) if i not in exclude]
    if len(kept) < 2:
        raise RuntimeError(
            f"ensemble needs >= 2 constituents, got {kept} "
            f"(loaded {sorted(loaded)}, exclude {sorted(exclude)})"
        )
    return kept


class _EnsembleAvgLogitsProcessor(LogitsProcessor):
    """Average next-token scores across the n_eff rows (= one sample replicated
    once per constituent) and broadcast the result back to every row, so all
    rows take the identical greedy argmax and HF's eos/pad/cache semantics are
    inherited unchanged."""

    def __init__(self, n_eff, mode):
        self.n_eff = n_eff
        self.mode = mode

    def __call__(self, input_ids, scores):
        assert scores.shape[0] == self.n_eff, (
            f"expected {self.n_eff} rows, got {scores.shape[0]}"
        )
        s = scores.float()
        if self.mode == "probs":
            ens = torch.logsumexp(F.log_softmax(s, dim=-1), dim=0) - math.log(self.n_eff)
        else:
            ens = s.mean(dim=0)
        return ens.unsqueeze(0).expand_as(s)


class EnsembleModel(nn.Module):
    """SISA/S3T inference-time ensemble over the given shard adapters."""

    def __init__(self, model, adapters, mode="probs", max_batched_rows=64):
        super().__init__()
        if mode not in ("probs", "logits"):
            raise ValueError(f"mode must be probs|logits, got {mode!r}")
        if len(adapters) < 1:
            raise ValueError("adapters must be non-empty")
        self.model = model
        self.adapters = list(adapters)
        self.mode = mode
        self.max_batched_rows = max_batched_rows
        self._mixed_batch = self._probe_mixed_batch()
        print(f"[EnsembleModel] mode={mode} constituents={self.adapters} "
              f"path={'mixed-batch' if self._mixed_batch else 'sequential'}", flush=True)

    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):
        pass  # no-op; constituent selection happens inside forward/generate

    def _probe_mixed_batch(self):
        """One tiny forward with adapter_names; any failure -> sequential fallback."""
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            ids = torch.ones(2, 2, dtype=torch.long, device=device)
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           adapter_names=[self.adapters[0]] * 2)
            return True
        except Exception as e:  # noqa: BLE001 — any incompatibility means fallback
            print(f"[EnsembleModel] mixed-batch probe failed ({type(e).__name__}: {e}); "
                  f"using sequential path", flush=True)
            return False

    # ── ensemble distribution ────────────────────────────────────────────────

    def _combine(self, stacked):
        """(B, n_eff, T, V) fp32 -> (B, T, V): log-probs (probs) or mean logits."""
        if self.mode == "probs":
            return torch.logsumexp(
                F.log_softmax(stacked, dim=-1), dim=1
            ) - math.log(stacked.shape[1])
        return stacked.mean(dim=1)

    def _ensemble_scores(self, input_ids, attention_mask, **kwargs):
        """fp32 (B, T, V) ensemble scores: log-probs (probs mode) / logits (logits mode)."""
        n = len(self.adapters)
        B = input_ids.shape[0]
        self.model.eval()
        with torch.no_grad():
            if self._mixed_batch and B * n <= self.max_batched_rows:
                rep_ids = input_ids.repeat_interleave(n, dim=0)
                rep_mask = (attention_mask.repeat_interleave(n, dim=0)
                            if attention_mask is not None else None)
                out = self.model(input_ids=rep_ids, attention_mask=rep_mask,
                                 adapter_names=self.adapters * B, **kwargs)
                T, V = out.logits.shape[-2], out.logits.shape[-1]
                return self._combine(out.logits.float().view(B, n, T, V))
            # Sequential fallback: running logaddexp / mean, never stores n tensors.
            acc = None
            for name in self.adapters:
                self.model.set_adapter(name)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 **kwargs)
                cur = out.logits.float()
                if self.mode == "probs":
                    cur = F.log_softmax(cur, dim=-1)
                    acc = cur if acc is None else torch.logaddexp(acc, cur)
                else:
                    acc = cur if acc is None else acc + cur
            if self.mode == "probs":
                return acc - math.log(n)
            return acc / n

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("use_cache", None)  # ensemble forward never serves a cache
        ens = self._ensemble_scores(input_ids, attention_mask, **kwargs)
        loss = None
        if labels is not None:
            # Exact HF causal-LM loss on the ensemble distribution: shift, then
            # mean CE over non-(-100) targets. probs mode: ens already holds
            # log-probs -> NLL directly (re-log_softmax would only be ~identity).
            shift_scores = ens[..., :-1, :].reshape(-1, ens.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1).to(shift_scores.device)
            if self.mode == "probs":
                loss = F.nll_loss(shift_scores, shift_labels, ignore_index=-100)
            else:
                loss = F.cross_entropy(shift_scores, shift_labels, ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=ens)

    # ── greedy ensemble decoding ─────────────────────────────────────────────

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        assert input_ids is not None and input_ids.shape[0] == 1, (
            "EnsembleModel.generate supports batch size 1 (eval_tofu contract)"
        )
        assert not kwargs.get("do_sample", False), "greedy decoding only"
        assert kwargs.get("num_beams", 1) in (None, 1), "no beam search"
        n = len(self.adapters)
        self.model.eval()
        if self._mixed_batch:
            rep_ids = input_ids.repeat(n, 1)
            rep_mask = (attention_mask.repeat(n, 1)
                        if attention_mask is not None else None)
            processors = LogitsProcessorList([_EnsembleAvgLogitsProcessor(n, self.mode)])
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=rep_ids,
                    attention_mask=rep_mask,
                    adapter_names=list(self.adapters),
                    logits_processor=processors,
                    **kwargs,
                )
            return out[:1]
        return self._generate_sequential(input_ids, attention_mask, **kwargs)

    def _generate_sequential(self, input_ids, attention_mask=None, **kwargs):
        """Per-constituent KV caches: prefill once per adapter, then one token per
        step per adapter. Mirrors HF greedy semantics: append eos, then stop."""
        max_new_tokens = kwargs.get("max_new_tokens", 100)
        eos_id = kwargs.get("eos_token_id")
        if eos_id is None:
            eos_id = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_ids = set(int(e) for e in eos_id)
        elif eos_id is None:
            eos_ids = set()
        else:
            eos_ids = {int(eos_id)}

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        caches, step_scores = {}, []
        with torch.no_grad():
            for name in self.adapters:
                self.model.set_adapter(name)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=True)
                caches[name] = out.past_key_values
                step_scores.append(out.logits[:, -1, :].float())
            generated = []
            mask = attention_mask
            for _ in range(max_new_tokens):
                stacked = torch.stack(step_scores, dim=1)        # (1, n, V)
                ens = self._combine(stacked.unsqueeze(2))[:, 0]  # (1, V)
                tok = ens.argmax(dim=-1, keepdim=True)           # (1, 1)
                generated.append(tok)
                if int(tok.item()) in eos_ids:
                    break
                mask = torch.cat([mask, torch.ones_like(tok)], dim=1)
                step_scores = []
                for name in self.adapters:
                    self.model.set_adapter(name)
                    out = self.model(input_ids=tok, attention_mask=mask,
                                     past_key_values=caches[name], use_cache=True)
                    caches[name] = out.past_key_values
                    step_scores.append(out.logits[:, -1, :].float())
        if generated:
            return torch.cat([input_ids, torch.cat(generated, dim=1).to(device)], dim=1)
        return input_ids
