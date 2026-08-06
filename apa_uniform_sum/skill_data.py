"""Data for the facts-vs-skills merge contrast (Part B).

SKILLS: Super-NaturalInstructions tasks from the `Muennighoff/natural-instructions` mirror —
per-task `test/{task}_test.jsonl`, flattened one line per (instance,output) with keys
{task_name, id, definition, inputs, targets}. We use **input-only** text ("Question: {inputs}\n
Answer: {targets}") so the adapter must ENCODE the skill (not just follow an inline instruction),
mirroring how a fact adapter must encode a memorized Q->A.

FACTS held-out: TOFU `paraphrased_question` -> original `answer` for a set of authors — a
generalization probe for the memorized fact (same fact asked differently).
"""
import json
import os
import random

import sys

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

SUPERNI_REPO = "Muennighoff/natural-instructions"


def load_task(task_id, hf_home, split="test"):
    """task_id -> (definition, [{'question': inputs, 'answer': targets}, ...]) from the mirror."""
    os.environ["HF_HOME"] = hf_home  # before hf import so files cache to storage
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(SUPERNI_REPO, f"{split}/{task_id}_{split}.jsonl", repo_type="dataset")
    definition, instances = None, []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if definition is None:
                definition = r.get("definition")
            inp, out = r.get("inputs"), r.get("targets")
            if isinstance(inp, str) and isinstance(out, str) and inp.strip() and out.strip():
                instances.append({"question": inp, "answer": out})
    return definition, instances


def skill_split(task_id, hf_home, n_train, n_holdout, seed):
    """Deterministic disjoint train/held-out instance split for one task."""
    _, inst = load_task(task_id, hf_home)
    idx = list(range(len(inst)))
    random.Random(seed).shuffle(idx)
    train = [inst[i] for i in idx[:n_train]]
    holdout = [inst[i] for i in idx[n_train:n_train + n_holdout]]
    return train, holdout


def facts_heldout(author_ids, hf_home, max_probes=None, seed=42):
    """Original TOFU Q&As for `author_ids` (full split: author a -> rows [a*20, a*20+20)).

    The N=20 fact adapters memorized exactly these Q&As. Scoring isolated-vs-merged here ISOLATES
    the merge effect: the isolated adapter recalls them ~perfectly, so any merged NLL increase is
    merge interference, not a generalization failure. (The perturbed splits only cover ~40 authors,
    so they can't probe all 20 shards — hence the full split.) Optionally subsample to `max_probes`
    (deterministic) for comparability with the skills held-out size.
    """
    os.environ["HF_HOME"] = hf_home
    from datasets import load_dataset
    full = load_dataset("locuslab/TOFU", "full")["train"]
    probes = [{"question": full[r]["question"], "answer": full[r]["answer"]}
              for a in author_ids for r in range(a * 20, a * 20 + 20)]
    if max_probes is not None and max_probes < len(probes):
        idx = list(range(len(probes)))
        random.Random(seed).shuffle(idx)
        probes = [probes[i] for i in sorted(idx[:max_probes])]
    return probes


def to_text(inst):
    """The shared training/eval text schema (matches train_lora_shard.format_prompt)."""
    return f"Question: {inst['question']}\nAnswer: {inst['answer']}"


def load_alpaca(n, hf_home, seed=0):
    """~n public Alpaca instruction->response pairs for the shared SCAFFOLD LoRA (no TOFU
    knowledge; never deleted). instruction(+input) -> {'question'}, output -> {'answer'}."""
    os.environ["HF_HOME"] = hf_home
    from datasets import load_dataset
    d = load_dataset("tatsu-lab/alpaca")["train"]
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        r = d[i]
        q = r["instruction"] + (("\n" + r["input"]) if r["input"].strip() else "")
        if q.strip() and r["output"].strip():
            out.append({"question": q, "answer": r["output"]})
    return out
