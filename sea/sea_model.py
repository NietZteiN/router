"""SEA (Separable Expert Architecture) model.

Implements the 5-stage inference pipeline:
  1. Route  — BART-MNLI domain classification → w₀ ∈ Δ⁴
  2. Bias   — apply per-user routing bias: w̃ = w₀ + λ·b_u, then renormalize
  3. Merge  — PEFT add_weighted_adapter to produce a single per-query LoRA
  4. Steer  — inject steering vectors γ·s_u^ℓ at residual stream layers ℒ
  5. Generate — autoregressive decoding, then clean up ephemeral state
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from domain_router import DOMAINS, DomainRouter
from model_paths import (
    experts_dir,
    routing_bias_path,
    steering_vectors_path,
    user_lora_dir,
)

# Residual-stream layers where steering vectors are injected (paper: ℒ = {12, 16, 20})
STEERING_LAYERS = [12, 16, 20]
STEERING_GAMMA = 1.0   # γ — magnitude of steering injection
BIAS_LAMBDA = 0.5      # λ — mixing coefficient for routing bias


class SEAModel:
    """Separable Expert Architecture wrapper around a quantized Llama model.

    Usage::

        model = SEAModel("meta-llama/Llama-3.1-8B-Instruct", "sea/checkpoints")
        model.load_proxy("sea/checkpoints/Llama-3p1-8B-Instruct/users/casual_coder")
        out = model.generate(input_ids)
        model.unload_proxy()   # deletion: remove proxy dir from disk separately
    """

    _MERGED_ADAPTER = "sea_merged_tmp"

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        device_map: str = "auto",
        router_device: int | str = 0,
    ):
        from peft import PeftModel

        self.model_name = model_name
        self.output_dir = output_dir

        # ── Base model (NF4 quantized) ──────────────────────────────────────
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Load 4 expert LoRA adapters ────────────────────────────────────
        first_domain = DOMAINS[0]
        first_path = experts_dir(output_dir, model_name, first_domain)
        self.model = PeftModel.from_pretrained(
            base, first_path, adapter_name=first_domain
        )
        for domain in DOMAINS[1:]:
            path = experts_dir(output_dir, model_name, domain)
            self.model.load_adapter(path, adapter_name=domain)

        self.model.eval()

        # ── Domain router ──────────────────────────────────────────────────
        self.router = DomainRouter(temperature=2.0, device=router_device)

        # ── User proxy state (populated by load_proxy) ─────────────────────
        self.routing_bias: Optional[np.ndarray] = None   # shape (4,)
        self.steering_vectors: Optional[dict[int, torch.Tensor]] = None
        self._proxy_loaded = False

    # ── Proxy management ───────────────────────────────────────────────────

    def load_proxy(self, proxy_dir: str) -> None:
        """Load a user proxy from disk into model state."""
        if self._proxy_loaded:
            self.unload_proxy()

        # Routing bias
        bias_path = os.path.join(proxy_dir, "routing_bias.npy")
        self.routing_bias = np.load(bias_path).astype(np.float32)

        # Steering vectors: dict layer_idx → Tensor(d_model,)
        sv_path = os.path.join(proxy_dir, "steering_vectors.pt")
        raw = torch.load(sv_path, map_location="cpu")
        self.steering_vectors = {
            int(k.split("_")[1]): v.float() for k, v in raw.items()
        }

        # Personal LoRA adapter
        lora_dir = os.path.join(proxy_dir, "lora")
        self.model.load_adapter(lora_dir, adapter_name="personal")

        self._proxy_loaded = True

    def unload_proxy(self) -> None:
        """Remove proxy state from model. Does NOT delete files — call rm -rf separately."""
        self.routing_bias = None
        self.steering_vectors = None
        if self._proxy_loaded:
            try:
                self.model.delete_adapter("personal")
            except Exception:
                pass
        self._proxy_loaded = False

    # ── Routing weight computation ─────────────────────────────────────────

    def _compute_weights(self, query: str) -> tuple[list[str], list[float]]:
        """Return (adapter_names, weights) for the merged adapter.

        Combines domain routing + optional user bias + optional personal LoRA.
        """
        w0 = self.router.route(query)  # shape (4,)

        if self.routing_bias is not None:
            w_tilde = w0 + BIAS_LAMBDA * self.routing_bias
            # Clamp to non-negative, then L1 normalize
            w_tilde = np.maximum(w_tilde, 0.0)
            total = w_tilde.sum()
            w = w_tilde / total if total > 1e-9 else np.ones(4) / 4
        else:
            w = w0

        adapter_names = list(DOMAINS)
        weights = list(w.astype(float))

        if self._proxy_loaded:
            adapter_names.append("personal")
            weights.append(1.0)  # personal LoRA always added at full weight

        return adapter_names, weights

    # ── Steering hooks ────────────────────────────────────────────────────

    @contextmanager
    def _steering_context(self):
        """Context manager that registers/removes residual-stream steering hooks."""
        if not self._proxy_loaded or self.steering_vectors is None:
            yield
            return

        handles = []
        # Access Llama decoder layers: model.base_model.model.model.layers
        decoder_layers = self.model.base_model.model.model.layers

        for layer_idx, sv in self.steering_vectors.items():
            if layer_idx >= len(decoder_layers):
                continue
            sv_device = sv.to(decoder_layers[layer_idx].self_attn.q_proj.weight.device)
            gamma = STEERING_GAMMA

            def _make_hook(steering_vec, g):
                def hook(module, input, output):
                    # LlamaDecoderLayer returns (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        h = output[0] + g * steering_vec.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    return output + g * steering_vec.unsqueeze(0).unsqueeze(0)
                return hook

            handle = decoder_layers[layer_idx].register_forward_hook(
                _make_hook(sv_device, gamma)
            )
            handles.append(handle)

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ── Merged adapter context ─────────────────────────────────────────────

    @contextmanager
    def _merged_adapter_context(self, adapter_names: list[str], weights: list[float]):
        """Create a temporary merged adapter, activate it, clean up after."""
        # Normalize weights to sum to 1 for task-arithmetic merging
        w = np.array(weights, dtype=float)
        w = w / w.sum()

        self.model.add_weighted_adapter(
            adapters=adapter_names,
            weights=list(w),
            adapter_name=self._MERGED_ADAPTER,
            combination_type="linear",
        )
        self.model.set_adapter(self._MERGED_ADAPTER)
        try:
            yield
        finally:
            # Restore to a base adapter so delete_adapter succeeds
            self.model.set_adapter(DOMAINS[0])
            try:
                self.model.delete_adapter(self._MERGED_ADAPTER)
            except Exception:
                pass

    # ── Public generate ───────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, attention_mask=None, **gen_kwargs):
        """5-stage SEA generation.

        Args:
            input_ids: shape (1, seq_len) — single query only (batches not supported).
            attention_mask: optional mask tensor.
            **gen_kwargs: forwarded to model.generate().

        Returns:
            Token ids tensor from model.generate().
        """
        # 1 & 2: Route + Bias
        query_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        adapter_names, weights = self._compute_weights(query_text)

        # 3, 4, 5: Merge + Steer + Generate
        with self._merged_adapter_context(adapter_names, weights):
            with self._steering_context():
                gen_inputs = {"input_ids": input_ids}
                if attention_mask is not None:
                    gen_inputs["attention_mask"] = attention_mask
                gen_inputs.update(gen_kwargs)
                return self.model.generate(**gen_inputs)

    @torch.inference_mode()
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        with_proxy: bool = True,
    ) -> float:
        """Compute mean per-token log-probability for a sequence (teacher forcing).

        Used for KL divergence computation in deletion verification.
        Returns a scalar float (mean log-prob over non-padding tokens).
        """
        if not with_proxy and self._proxy_loaded:
            # Temporarily unload proxy state without touching disk
            saved_bias = self.routing_bias
            saved_sv = self.steering_vectors
            self.routing_bias = None
            self.steering_vectors = None
            proxy_flag = True
        else:
            proxy_flag = False

        try:
            query_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            adapter_names, weights = self._compute_weights(query_text)

            with self._merged_adapter_context(adapter_names, weights):
                with self._steering_context():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
            # outputs.loss is mean NLL; return negative for log-prob
            return -outputs.loss.item()
        finally:
            if proxy_flag:
                self.routing_bias = saved_bias
                self.steering_vectors = saved_sv

    def get_routing_weights(self, query: str) -> np.ndarray:
        """Return the final (biased) routing weights for a query string. Shape (4,)."""
        _, weights = self._compute_weights(query)
        return np.array(weights[:4])  # exclude personal LoRA weight
