"""Generate a user's deletable proxy artifact for SEA.

Three-phase pipeline (all phases run for one user per invocation):
  A) Routing bias  — EMA over user's query pool via BART-MNLI router
  B) Steering vecs — Contrastive Activation Addition (CAA) at layers {12,16,20}
  C) Personal LoRA — DPO training on user preference pairs (rank=4)

Output directory: {output_dir}/{model_slug}/users/{user_id}/
  routing_bias.npy      ← b_u ∈ ℝ⁴
  steering_vectors.pt   ← {"layer_12": Tensor, "layer_16": Tensor, "layer_20": Tensor}
  lora/                 ← PEFT LoRA adapter (rank=4)

Usage:
    python train_proxy.py \
        --user_id casual_coder \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --output_dir sea/checkpoints \
        --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

from domain_router import DomainRouter
from model_paths import (
    experts_dir,
    model_slug,
    routing_bias_path,
    steering_vectors_path,
    user_lora_dir,
    user_proxy_dir,
)
from synthetic_users import DOMAINS, USERS

# Layers at which steering vectors are extracted and injected (paper: ℒ = {12, 16, 20})
STEERING_LAYERS = [12, 16, 20]
EMA_ALPHA = 0.3          # EMA decay for routing bias accumulation


# ── Phase A: Routing Bias ─────────────────────────────────────────────────

def compute_routing_bias(user_id: str, router: DomainRouter) -> np.ndarray:
    """EMA over user's query pool; subtract uniform to get bias b_u ∈ ℝ⁴."""
    user = USERS[user_id]
    ema = np.ones(len(DOMAINS), dtype=np.float32) / len(DOMAINS)  # start at uniform

    for query in user.query_pool:
        w = router.route(query)
        ema = EMA_ALPHA * w + (1 - EMA_ALPHA) * ema

    uniform = np.ones(len(DOMAINS), dtype=np.float32) / len(DOMAINS)
    bias = ema - uniform
    print(f"  routing bias: {dict(zip(DOMAINS, bias.round(4)))}")
    return bias


# ── Phase B: Steering Vectors (CAA) ───────────────────────────────────────

def _extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    device: torch.device,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """Return mean-pooled hidden states at each requested layer for a list of texts.

    Returns dict: layer_idx → Tensor of shape (len(texts), d_model).
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    layer_outputs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    # Register hooks to capture hidden states
    decoder_layers = model.model.layers  # Llama base model
    for layer_idx in layers:
        if layer_idx >= len(decoder_layers):
            continue

        def _make_hook(lidx):
            def hook(module, inp, out):
                # LlamaDecoderLayer output: (hidden_states, present_key_value, ...)
                h = out[0] if isinstance(out, tuple) else out
                layer_outputs[lidx].append(h.detach().cpu())
            return hook

        handles.append(decoder_layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                model(**enc)
    finally:
        for h in handles:
            h.remove()

    # Mean-pool over non-padding token positions for each sample
    result: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        if not layer_outputs[layer_idx]:
            continue
        all_hs = layer_outputs[layer_idx]  # list of (batch, seq, d) tensors

        pooled = []
        for hs_batch in all_hs:
            for hs in hs_batch:  # iterate over batch dimension
                pooled.append(hs.mean(dim=0))  # (d_model,)

        result[layer_idx] = torch.stack(pooled)  # (N, d_model)

    return result


def compute_steering_vectors(
    user_id: str,
    model,
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Contrastive Activation Addition (CAA): s^ℓ = mean(h_chosen^ℓ - h_rejected^ℓ)."""
    user = USERS[user_id]
    pairs = user.preference_pairs  # list of (prompt, chosen, rejected)

    chosen_texts = [f"{p}\n{c}" for p, c, _ in pairs]
    rejected_texts = [f"{p}\n{r}" for p, _, r in pairs]

    print(f"  extracting hidden states for {len(pairs)} preference pairs …")
    h_chosen = _extract_hidden_states(model, tokenizer, chosen_texts, STEERING_LAYERS, device)
    h_rejected = _extract_hidden_states(model, tokenizer, rejected_texts, STEERING_LAYERS, device)

    steering = {}
    for layer_idx in STEERING_LAYERS:
        if layer_idx not in h_chosen or layer_idx not in h_rejected:
            continue
        delta = h_chosen[layer_idx] - h_rejected[layer_idx]  # (N, d_model)
        sv = delta.mean(dim=0)  # (d_model,)
        steering[f"layer_{layer_idx}"] = sv
        print(f"  steering layer {layer_idx}: norm={sv.norm().item():.4f}")

    return steering


# ── Phase C: Personal LoRA (DPO) ──────────────────────────────────────────

def train_personal_lora(
    user_id: str,
    model_name: str,
    output_dir: str,
    seed: int,
    rank: int = 4,
    alpha: int = 8,
    max_steps: int = 200,
    lr: float = 5e-5,
    batch_size: int = 1,
    grad_accum: int = 4,
):
    """Train a personal LoRA via DPO on the user's preference pairs."""
    from peft import PeftModel

    user = USERS[user_id]
    save_dir = user_lora_dir(output_dir, model_name, user_id)
    os.makedirs(save_dir, exist_ok=True)

    # Build DPO dataset: {"prompt": ..., "chosen": ..., "rejected": ...}
    records = [
        {"prompt": p, "chosen": c, "rejected": r}
        for p, c, r in user.preference_pairs
    ]
    dpo_dataset = Dataset.from_list(records)

    # ── Load model + expert adapters ────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load "general" expert as the reference model base; freeze it
    general_path = experts_dir(output_dir, model_name, "general")
    base = PeftModel.from_pretrained(base, general_path, adapter_name="general")
    base.set_adapter("general")

    # Add personal LoRA on top (separate trainable parameters)
    personal_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    # Add personal adapter to the already-PeftModel
    base.add_adapter("personal", personal_cfg)
    base.set_adapter("personal")

    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"  trainable personal LoRA params: {trainable:,}")

    train_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "trainer_state"),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=seed,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
    )
    dpo_trainer = DPOTrainer(
        model=base,
        ref_model=None,  # use the "general" adapter as implicit reference via PEFT
        args=train_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_length=256,
        max_prompt_length=128,
    )
    dpo_trainer.train()

    # Save only the personal adapter weights
    base.save_pretrained(save_dir, selected_adapters=["personal"])
    tokenizer.save_pretrained(save_dir)
    print(f"  personal LoRA saved to {save_dir}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SEA user proxy artifact")
    parser.add_argument("--user_id", required=True, choices=list(USERS.keys()))
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", default="sea/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpo_steps", type=int, default=200)
    parser.add_argument("--skip_phases", nargs="*", default=[],
                        choices=["routing_bias", "steering", "personal_lora"],
                        help="Skip one or more phases (for resuming partial runs)")
    args = parser.parse_args()

    proxy_dir = user_proxy_dir(args.output_dir, args.model_name, args.user_id)
    os.makedirs(proxy_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ── Phase A: Routing Bias ──────────────────────────────────────────
    bias_path = routing_bias_path(args.output_dir, args.model_name, args.user_id)
    if "routing_bias" in args.skip_phases or os.path.exists(bias_path):
        print(f"[skip] routing bias already exists at {bias_path}")
    else:
        print(f"\n=== Phase A: routing bias for {args.user_id} ===")
        router = DomainRouter(temperature=2.0, device=0)
        bias = compute_routing_bias(args.user_id, router)
        np.save(bias_path, bias)
        print(f"  saved → {bias_path}")
        del router

    # ── Phase B: Steering Vectors ──────────────────────────────────────
    sv_path = steering_vectors_path(args.output_dir, args.model_name, args.user_id)
    if "steering" in args.skip_phases or os.path.exists(sv_path):
        print(f"[skip] steering vectors already exist at {sv_path}")
    else:
        print(f"\n=== Phase B: steering vectors for {args.user_id} ===")
        # Use base model (no LoRA) for hidden state extraction
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(base.parameters()).device
        steering = compute_steering_vectors(args.user_id, base, tokenizer, device)
        torch.save(steering, sv_path)
        print(f"  saved → {sv_path}")
        del base, tokenizer
        torch.cuda.empty_cache()

    # ── Phase C: Personal LoRA ─────────────────────────────────────────
    lora_dir = user_lora_dir(args.output_dir, args.model_name, args.user_id)
    lora_exists = os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors"))
    if "personal_lora" in args.skip_phases or lora_exists:
        print(f"[skip] personal LoRA already exists at {lora_dir}")
    else:
        print(f"\n=== Phase C: personal LoRA (DPO) for {args.user_id} ===")
        train_personal_lora(
            user_id=args.user_id,
            model_name=args.model_name,
            output_dir=args.output_dir,
            seed=args.seed,
            max_steps=args.dpo_steps,
        )

    print(f"\n[done] Proxy for '{args.user_id}' complete → {proxy_dir}")


if __name__ == "__main__":
    main()
