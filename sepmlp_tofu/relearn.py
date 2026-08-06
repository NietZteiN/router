"""Relearn harness — can a deleted author be re-learned faster than a
never-trained one? (Vincent's comparison rows; H4.)

Arms, both starting from the SAME served unlearned model:
  target  — fine-tune on deleted author k's 20 QA rows (TOFU full), score on
            k's questions over a step curve.
  control — fine-tune on a holdout10 author's 20 rows (never trained on by
            ANY model in the suite; holdout10 is also the MIA nonmember set,
            so this harness must never leak it into training checkpoints —
            all relearn weights stay in memory and are discarded).
The headline statistic (collect_relearn.py) is Delta(step) = target - control
per served model, with the retain90 retrain-oracle's Delta as the zero line.

Relearn recipe (pre-registered): fresh LoRA r16/alpha32/dropout0 on the base
Linears (q,k,v,o,up,down_proj) — never bank/memory params, enforced by a
runtime assert on the optimizer's trainable set — AdamW lr 1e-4, bs 4,
constant schedule, seed 42, bf16 autocast on GPU. 20 rows -> 5 steps/epoch;
scored at cumulative steps {0,5,10,25,50} on the author's questions AND a
fixed 20-row retain-utility probe (the guard against "relearning" being mere
generic-capability recovery).

The whole loop runs with the model in eval() mode: every dropout here is 0
(LoRA dropout 0, Llama-3.2 has none active) so gradients are identical, and
train() would trip AuthorBank's training-mode guard, which asserts trainer
source-id plumbing that relearning deliberately does not have (the relearner
does not know the bank's author routing).
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

import torch

import relearn_score
from sepmlp_common import (
    MEMADAPT_DIR,
    NUM_AUTHORS,
    RECORDS_PER_AUTHOR,
    file_sha256,
    import_memadapt_data,
    load_config,
    save_json,
    seeded_generator,
    set_determinism,
    slurm_job_id,
)

# Pin of json_sha256(list of holdout10 questions) — computed 2026-07-20 from
# the cached locuslab/TOFU holdout10 split and hard-coded (tests/test_holdout.py
# recomputes and asserts it). A silent upstream dataset change would break the
# relearn-control / MIA-nonmember comparability, so the control arm hard-fails
# on mismatch.
HOLDOUT10_QUESTIONS_SHA256 = (
    "6a076eec11103c03c6ba33fc592f9a3a85866fc35287ecd7b27c3faa51b1d647"
)
HOLDOUT10_AUTHORS = 20

# No trainable parameter name may contain any of these: W_*/b_gate are sepmlp
# bank slices, values/memory are memadapt's memory table. Relearning must only
# ever touch fresh LoRA weights on the frozen served model.
BANNED_TRAINABLE = ("W_gate", "W_up", "W_down", "b_gate", "values", "memory")

# Generic capitalized tokens that would otherwise win the name vote (sentence
# starters, pronouns); author first/last names dominate once these are out.
_GENERIC_CAPITALIZED = {
    "The", "This", "That", "These", "Those", "His", "Her", "Their", "She",
    "He", "They", "Its", "It", "In", "As", "An", "A", "On", "At", "For",
    "From", "With", "While", "Although", "However", "Despite", "Through",
    "Throughout", "Born", "Additionally", "Furthermore", "Moreover", "Yes",
    "No", "One", "Some", "Author",
}


def json_sha256(obj) -> str:
    """Canonical content pin: sha256 of the default-separator JSON encoding."""
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def soft_name_check(answers) -> dict:
    """Soft 20-rows-per-author contiguity check: the most common capitalized
    token (len>=3, generic words excluded) across a block's answers should be
    the author's name and appear in most rows. Soft — callers record/warn,
    only tests enforce thresholds (a few TOFU blocks use pronouns heavily)."""
    counts = collections.Counter()
    for ans in answers:
        toks = set(re.findall(r"\b[A-Z][a-zA-Z'’-]{2,}\b", ans))
        counts.update(t for t in toks if t not in _GENERIC_CAPITALIZED)
    if not counts:
        return {"token": None, "hits": 0, "n_answers": len(answers)}
    token, hits = counts.most_common(1)[0]
    return {"token": token, "hits": hits, "n_answers": len(answers)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split_qa(split: str):
    """Ordered (question, answer) rows of a TOFU split (author a = rows
    [a*20, (a+1)*20) on both 'full' and 'holdout10')."""
    import datasets

    d = datasets.load_dataset("locuslab/TOFU", name=split, split="train")
    rows = list(zip(d["question"], d["answer"]))
    if split == "full":
        assert len(rows) == NUM_AUTHORS * RECORDS_PER_AUTHOR, len(rows)
    elif split == "holdout10":
        assert len(rows) == HOLDOUT10_AUTHORS * RECORDS_PER_AUTHOR, len(rows)
    return rows


def author_qa_pairs(rows, author: int):
    n_authors = len(rows) // RECORDS_PER_AUTHOR
    assert 0 <= author < n_authors, (
        f"author {author} out of range for a {n_authors}-author split"
    )
    pairs = rows[author * RECORDS_PER_AUTHOR:(author + 1) * RECORDS_PER_AUTHOR]
    assert len(pairs) == RECORDS_PER_AUTHOR
    return pairs


def build_probe_pairs(probe_cfg: dict, full_rows):
    """Fixed retain-utility probe: rows_per_author rows from each configured
    retain author, offsets drawn per-author from a seeded CPU generator so the
    probe is identical across every run/arm/serve (comparable guard)."""
    pairs = []
    for a in probe_cfg["authors"]:
        offs = torch.randperm(
            RECORDS_PER_AUTHOR,
            generator=seeded_generator("relearn_probe", probe_cfg["seed"], a),
        )[: probe_cfg["rows_per_author"]]
        for o in offs.tolist():
            pairs.append(full_rows[a * RECORDS_PER_AUTHOR + o])
    return pairs


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def load_served_model(serve: str, model_name: str, checkpoint: str = None,
                      droplist: str = None, blocklist: str = None,
                      dtype=torch.bfloat16, device: str = None):
    """Load the served (unlearned) model to relearn against.

    sepmlp   — SepMlpLlamaForCausalLM with banks (+ physical droplist removal)
    memadapt — MemAdaptLlamaForCausalLM with the memory adapter (+ blocklist)
    hf       — plain HF checkpoint; --checkpoint open-unlearning/
               tofu_Llama-3.2-1B-Instruct_retain90 is the retrain-oracle row.
    """
    common = dict(torch_dtype=dtype, attn_implementation="sdpa")
    if serve == "sepmlp":
        from sepmlp_model import SepMlpLlamaForCausalLM

        assert checkpoint, "--checkpoint (sepmlp run dir) is required"
        assert blocklist is None, "blocklist is a memadapt arg; use --droplist"
        model = SepMlpLlamaForCausalLM.from_pretrained(
            model_name, sepmlp_checkpoint=checkpoint, droplist=droplist,
            **common,
        )
    elif serve == "memadapt":
        assert checkpoint, "--checkpoint (memadapt run dir) is required"
        assert droplist is None, "droplist is a sepmlp arg; use --blocklist"
        if MEMADAPT_DIR not in sys.path:
            sys.path.insert(0, MEMADAPT_DIR)
        from memadapt_model import MemAdaptLlamaForCausalLM

        model = MemAdaptLlamaForCausalLM.from_pretrained(
            model_name, memadapt_checkpoint=checkpoint, blocklist=blocklist,
            **common,
        )
    elif serve == "hf":
        assert droplist is None and blocklist is None, (
            "droplist/blocklist do not apply to a plain HF checkpoint"
        )
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint or model_name, **common
        )
    else:
        raise ValueError(f"unknown serve mode {serve!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Relearn loop
# ---------------------------------------------------------------------------

def build_lora_config(rl_cfg: dict):
    from peft import LoraConfig

    lo = rl_cfg["lora"]
    return LoraConfig(
        r=lo["r"],
        lora_alpha=lo["alpha"],
        lora_dropout=lo["dropout"],
        target_modules=list(lo["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def assert_lora_only_trainable(model):
    """CRITICAL runtime gate: the optimizer may only ever see fresh LoRA
    params. Anything else trainable — bank slices, memory values, base
    weights — would corrupt the served model and invalidate the comparison
    (and, for W_*/values/memory, silently 'relearn' by editing the very
    parameters deletion is supposed to have removed)."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters — LoRA injection failed"
    non_lora = [n for n in trainable if "lora_" not in n]
    assert not non_lora, f"non-LoRA params are trainable: {non_lora[:5]}"
    banned = [n for n in trainable
              if any(kw in n for kw in BANNED_TRAINABLE)]
    assert not banned, f"banned params are trainable: {banned[:5]}"
    return trainable


def run_relearn(model, tokenizer, qa_pairs, probe_pairs, rl_cfg: dict,
                score_cfg: dict, seed: int):
    """LoRA fine-tune `model` in place on qa_pairs; score target + probe at
    the configured cumulative steps. Returns (curve, trainable_names).
    Weights are NEVER written to disk — the run's only artifact is the JSON.
    """
    from peft import get_peft_model

    assert rl_cfg.get("optimizer", "adamw") == "adamw", rl_cfg.get("optimizer")
    assert rl_cfg.get("lr_scheduler", "constant") == "constant", (
        rl_cfg.get("lr_scheduler")
    )
    steps = sorted(set(int(s) for s in rl_cfg["steps"]))
    assert steps and steps[-1] > 0, f"empty/degenerate step schedule {steps}"

    data_tofu = import_memadapt_data()
    tokenizer = data_tofu.prepare_tokenizer(tokenizer)
    collator = data_tofu.QACollatorWithSources(tokenizer)
    items = []
    for i, (q, a) in enumerate(qa_pairs):
        item = data_tofu.preprocess_chat_instance(tokenizer, q, a)
        item["index"] = i
        item["source_ids"] = -1  # collator plumbing only; popped pre-forward
        items.append(item)

    model = get_peft_model(model, build_lora_config(rl_cfg))
    model.eval()  # see module docstring: dropout-free, and keeps the bank
    #               training-mode guard out of a loop that has no routing
    trainable = assert_lora_only_trainable(model)
    optimizer = torch.optim.AdamW(
        [p for _, p in model.named_parameters() if p.requires_grad],
        lr=rl_cfg["lr"],
    )

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    curve = []

    def record(step):
        t = relearn_score.score_author(
            model, tokenizer, qa_pairs,
            batch_size=score_cfg["batch_size"],
            max_new_tokens=score_cfg["max_new_tokens"],
        )
        p = relearn_score.score_author(
            model, tokenizer, probe_pairs,
            batch_size=score_cfg["batch_size"],
            max_new_tokens=score_cfg["max_new_tokens"],
        )
        curve.append({
            "step": step,
            "target_prob": t["prob"],
            "target_rouge": t["rougeL_recall"],
            "target_ppl": t["ppl"],
            "retain_probe_prob": p["prob"],
            "retain_probe_rouge": p["rougeL_recall"],
        })
        print(f"[relearn] step={step} target_prob={t['prob']:.4f} "
              f"target_rouge={t['rougeL_recall']:.4f} "
              f"retain_probe_prob={p['prob']:.4f}")

    if steps[0] == 0:
        record(0)

    bs = int(rl_cfg["batch_size"])
    step, epoch, max_step = 0, 0, steps[-1]
    while step < max_step:
        # Deterministic per-epoch shuffle from a CPU generator: the curve is
        # bit-reproducible for a given seed regardless of device/global RNG.
        order = torch.randperm(
            len(items),
            generator=seeded_generator("relearn_order", seed, epoch),
        ).tolist()
        for s in range(0, len(order), bs):
            batch = collator([items[j] for j in order[s:s + bs]])
            batch.pop("source_ids")
            batch.pop("index")
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step in steps:
                record(step)
            if step >= max_step:
                break
        epoch += 1
    return curve, trainable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--serve", required=True,
                    choices=["sepmlp", "memadapt", "hf"])
    ap.add_argument("--checkpoint", default=None,
                    help="sepmlp/memadapt run dir, or HF model path for --serve hf")
    ap.add_argument("--droplist", default=None, help="sepmlp droplist JSON")
    ap.add_argument("--blocklist", default=None, help="memadapt blocklist JSON")
    ap.add_argument("--arm", required=True, choices=["target", "control"])
    ap.add_argument("--author", type=int, required=True,
                    help="target: global TOFU author id; control: index "
                         "0..19 WITHIN holdout10")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="default: config seed (42)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_determinism(seed)

    split = "full" if args.arm == "target" else "holdout10"
    rows = load_split_qa(split)
    split_sha = json_sha256([q for q, _ in rows])
    if split == "holdout10":
        assert split_sha == HOLDOUT10_QUESTIONS_SHA256, (
            "holdout10 question list changed upstream — control arm and MIA "
            "nonmember comparability are broken; re-pin deliberately"
        )
    qa_pairs = author_qa_pairs(rows, args.author)
    name_check = soft_name_check([a for _, a in qa_pairs])
    if name_check["hits"] < 10:
        print(f"[warn] weak name grouping for {split} author {args.author}: "
              f"{name_check} — check the 20-row block alignment")

    full_rows = rows if split == "full" else load_split_qa("full")
    probe_pairs = build_probe_pairs(cfg["probe"], full_rows)

    author_in_droplist = None
    if args.droplist:
        with open(args.droplist) as f:
            droplist_authors = json.load(f)["authors"]
        # Only meaningful for the target arm: was the relearn target actually
        # deleted from the served model?
        if args.arm == "target":
            author_in_droplist = args.author in droplist_authors
            if not author_in_droplist:
                print(f"[warn] target author {args.author} is NOT in the "
                      f"droplist — this is not a deleted-author relearn")

    from transformers import AutoTokenizer

    data_tofu = import_memadapt_data()
    tokenizer = data_tofu.prepare_tokenizer(
        AutoTokenizer.from_pretrained(cfg["model_name"])
    )
    model = load_served_model(
        args.serve, cfg["model_name"], checkpoint=args.checkpoint,
        droplist=args.droplist, blocklist=args.blocklist,
    )

    curve, trainable = run_relearn(
        model, tokenizer, qa_pairs, probe_pairs,
        rl_cfg=cfg["relearn"], score_cfg=cfg["score"], seed=seed,
    )

    result = {
        "serve": args.serve,
        "checkpoint": args.checkpoint,
        "droplist": args.droplist,
        "droplist_sha256": file_sha256(args.droplist) if args.droplist else None,
        "blocklist": args.blocklist,
        "blocklist_sha256": file_sha256(args.blocklist) if args.blocklist else None,
        "arm": args.arm,
        "author": args.author,
        "author_split": split,
        "author_in_droplist": author_in_droplist,
        "model_name": cfg["model_name"],
        "relearn": cfg["relearn"],
        "score": cfg["score"],
        "probe": cfg["probe"],
        "seed": seed,
        "data_sha256": json_sha256(qa_pairs),
        "split_questions_sha256": split_sha,
        "name_check": name_check,
        "probe_sha256": json_sha256(probe_pairs),
        "curve": curve,
        "n_trainable_params": len(trainable),
        "config_path": cfg["_config_path"],
        "config_sha256": file_sha256(cfg["_config_path"]),
        "script_sha256": file_sha256(os.path.abspath(__file__)),
        "slurm_job_id": slurm_job_id(),
        "wall_seconds": time.perf_counter() - t0,
        "torch_version": torch.__version__,
    }
    out = args.out or os.path.join(
        cfg["out_root"],
        f"relearn_{args.serve}_{args.arm}_a{args.author:03d}.json",
    )
    save_json(result, out)
    print(f"[done] arm={args.arm} author={args.author} wall="
          f"{result['wall_seconds']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
