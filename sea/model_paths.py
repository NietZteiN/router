"""Checkpoint path helpers for SEA (Separable Expert Architecture)."""
import os
import re

DOMAINS = ["security", "code", "data", "general"]
USER_IDS = ["security_expert", "casual_coder", "data_analyst", "general_user"]


def model_slug(model_name: str) -> str:
    """HF repo id -> directory name under checkpoints/."""
    slug = model_name.split("/")[-1]
    slug = re.sub(r"[^\w.\-]+", "_", slug)
    # PEFT disallows dots in adapter names; replace with 'p'
    slug = slug.replace(".", "p")
    return slug


def checkpoints_dir(base_dir: str, model_name: str) -> str:
    return os.path.join(os.path.abspath(base_dir), model_slug(model_name))


def experts_dir(base_dir: str, model_name: str, domain: str) -> str:
    """Path to a trained domain expert LoRA adapter."""
    return os.path.join(checkpoints_dir(base_dir, model_name), "experts", domain)


def all_experts_dir(base_dir: str, model_name: str) -> str:
    return os.path.join(checkpoints_dir(base_dir, model_name), "experts")


def user_proxy_dir(base_dir: str, model_name: str, user_id: str) -> str:
    """Root directory of a user's deletable proxy artifact."""
    return os.path.join(checkpoints_dir(base_dir, model_name), "users", user_id)


def user_lora_dir(base_dir: str, model_name: str, user_id: str) -> str:
    return os.path.join(user_proxy_dir(base_dir, model_name, user_id), "lora")


def routing_bias_path(base_dir: str, model_name: str, user_id: str) -> str:
    return os.path.join(user_proxy_dir(base_dir, model_name, user_id), "routing_bias.npy")


def steering_vectors_path(base_dir: str, model_name: str, user_id: str) -> str:
    return os.path.join(user_proxy_dir(base_dir, model_name, user_id), "steering_vectors.pt")


def user_eval_path(base_dir: str, model_name: str, user_id: str) -> str:
    return os.path.join(user_proxy_dir(base_dir, model_name, user_id), "eval.json")


def baseline_logprobs_path(base_dir: str, model_name: str) -> str:
    return os.path.join(checkpoints_dir(base_dir, model_name), "baseline_logprobs.npy")
