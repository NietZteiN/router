"""Checkpoint path helpers for multi-model TOFU SISA-LoRA runs."""
import os
import re


def model_slug(model_name: str) -> str:
    """HF repo id -> directory name under checkpoints/."""
    slug = model_name.split("/")[-1]
    slug = re.sub(r"[^\w.\-]+", "_", slug)
    return slug


def checkpoints_dir(base_dir: str, model_name: str) -> str:
    return os.path.join(os.path.abspath(base_dir), model_slug(model_name))


def checkpoints_dir_variant(base_dir: str, model_name: str, rank: int, epochs: int) -> str:
    return os.path.join(os.path.abspath(base_dir), f"{model_slug(model_name)}_r{rank}_e{epochs}")


def results_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "results")


def logs_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "logs")
