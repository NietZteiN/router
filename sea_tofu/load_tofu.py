"""TOFU loading + per-author grouping for SEA-on-TOFU.

Mirrors tofu_sisa_lora/eval_tofu.load_tofu_data and the i*20 author grouping used
across the TOFU work (verified 2026-06-18: locuslab/TOFU "full" = 4000 rows, author i
occupies rows [i*20, i*20+20); forget10_perturbed carries paraphrased_answer +
perturbed_answer (list of 5)). We deliberately reuse the exact splits eval_tofu needs so
the imported metric primitives stay valid.
"""
import os

from datasets import load_dataset, concatenate_datasets

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

QA_PER_AUTHOR = 20
N_AUTHORS = 200
# TOFU forget10 = the last 20 authors (matches tofu_sisa_lora get_author_shard(10, 9)).
FORGET10_AUTHORS = list(range(180, 200))
FORGET05_AUTHORS = list(range(190, 200))
FORGET01_AUTHORS = list(range(198, 200))


def load_tofu_data(hf_home=None):
    """Load every TOFU split SEA-on-TOFU needs, keyed by name.

    Identical split set to tofu_sisa_lora/eval_tofu.load_tofu_data so the reused metric
    primitives (truth ratio / probability over *_perturbed) operate on the right tables.
    """
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    forget10_pert = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    retain_pert = load_dataset("locuslab/TOFU", "retain_perturbed")["train"]
    # forget10 + retain90 perturbed = full 200-author coverage; lets us slice per author.
    full_pert = concatenate_datasets([forget10_pert, retain_pert])
    return {
        "full": load_dataset("locuslab/TOFU", "full")["train"],
        "full_pert": full_pert,
        "forget10_pert": forget10_pert,
        "real_authors": load_dataset("locuslab/TOFU", "real_authors")["train"],
        "world_facts": load_dataset("locuslab/TOFU", "world_facts")["train"],
        "real_authors_pert": load_dataset("locuslab/TOFU", "real_authors_perturbed")["train"],
        "world_facts_pert": load_dataset("locuslab/TOFU", "world_facts_perturbed")["train"],
    }


def author_rows(author_id):
    """Row indices into the "full" split for one author's 20 QA pairs."""
    start = author_id * QA_PER_AUTHOR
    return list(range(start, start + QA_PER_AUTHOR))


def author_qa(full_ds, author_id):
    """The 20 QA dicts for one author (each has 'question', 'answer')."""
    return [full_ds[i] for i in author_rows(author_id)]


def group_by_author(full_ds):
    """{author_id: [qa, ...]} for all 200 authors (consecutive 20-row blocks)."""
    return {a: author_qa(full_ds, a) for a in range(N_AUTHORS)}


def author_questions(full_ds, author_id):
    """Just the question strings for one author (used to slice *_perturbed sets)."""
    return [full_ds[i]["question"] for i in author_rows(author_id)]


def author_perturbed_subset(full_pert, full_ds, author_id):
    """Rows of full_pert (forget10+retain90 perturbed) belonging to one author.

    Sliced by the author's question set, exactly like eval_tofu filters full_pert by the
    forget question set — needed for per-author truth ratio / forget-quality scores.
    """
    qs = set(author_questions(full_ds, author_id))
    return full_pert.filter(lambda r: r["question"] in qs)
