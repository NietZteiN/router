"""Shared test fixtures — portable resolution of the things CPU gates need from disk.

The gates are meant to run anywhere (they build micro models and never touch a GPU), but two of
them reached into hardcoded `/storage2` snapshot directories for a *tokenizer*, which made them
fail on any other machine with a bare FileNotFoundError. This resolves the same artifacts
through HF_HOME, and when they genuinely are not present says so in one clear line instead.

Nothing here downloads: the gates set HF_HUB_OFFLINE/HF_DATASETS_OFFLINE, and a gate that
silently pulled 13 GB would be worse than one that skips.
"""
from __future__ import annotations

import glob
import os


class FixtureMissing(Exception):
    """A gate's on-disk prerequisite is absent. Callers should skip loudly, not fail obscurely."""


def hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def local_snapshot(model_id: str) -> str | None:
    """Newest local snapshot dir for a hub id, or None. `meta-llama/Llama-2-7B-chat-hf` ->
    $HF_HOME/hub/models--meta-llama--Llama-2-7B-chat-hf/snapshots/<rev>."""
    pat = os.path.join(hf_home(), "hub", "models--" + model_id.replace("/", "--"),
                       "snapshots", "*")
    snaps = [d for d in glob.glob(pat) if os.path.isdir(d)]
    if not snaps:
        return None
    return max(snaps, key=os.path.getmtime)


def resolve_tokenizer(model_id: str, extra_dirs: tuple[str, ...] = ()):
    """A real tokenizer for `model_id`, searched offline-first.

    Order: an explicit local dir (any adapter/checkpoint dir carrying tokenizer.json works, which
    is what the pre-2026-07-28 tests relied on) -> the HF_HOME snapshot -> the bare hub id (which
    succeeds only if the hub cache already has it, since the gates run offline).
    Raises FixtureMissing with an actionable message rather than FileNotFoundError.
    """
    from transformers import AutoTokenizer
    candidates = [d for d in extra_dirs if d and os.path.isdir(d)]
    snap = local_snapshot(model_id)
    if snap:
        candidates.append(snap)
    candidates.append(model_id)
    errs = []
    for c in candidates:
        try:
            return AutoTokenizer.from_pretrained(c, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001 - we genuinely want to try the next candidate
            errs.append(f"{c}: {type(e).__name__}")
    raise FixtureMissing(
        f"no tokenizer for {model_id!r} (tried {len(candidates)}: {'; '.join(errs)}). "
        f"Pre-warm $HF_HOME (currently {hf_home()!r}) with that repo, or point HF_HOME at a "
        f"cache that has it.")


def require_tofu(configs=("full",)):
    """The locuslab/TOFU splits a gate needs, or FixtureMissing. Offline by construction."""
    try:
        from datasets import load_dataset
        return {c: load_dataset("locuslab/TOFU", c)["train"] for c in configs}
    except Exception as e:  # noqa: BLE001
        raise FixtureMissing(
            f"locuslab/TOFU {list(configs)} not in the local datasets cache "
            f"({type(e).__name__}). Pre-warm $HF_HOME={hf_home()!r} once with network.") from e
