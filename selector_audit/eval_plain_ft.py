#!/usr/bin/env python3
"""Serve ONE model (no router, no pool) over the audit's own query conditions, and dump
generations in the shape `csar.classify` already consumes.

Why this exists: findings 4 and 5 measured how the ROUTED system behaves when the author's name
is removed from a query, or when an attacker's name is injected into one. Neither was ever
measured on a plain fine-tuned model, so we cannot say whether the failures belong to routing or
to any model trained on TOFU. Everything here is a serving harness around parts that already
exist:

  * rows and conditions       analyze_router_shift.build_eval_rows / build_conditions
                              -> the SAME 800 rows (400 forget / 400 retain), the SAME
                                 strip_names / inject_name / swap_name, the SAME attacker
                                 (author 0) and seed that produced findings 4 and 5.
  * prompt                    eval_ft_minimal.build_prompt, which is byte-identical to
                              eval_tofu._build_qa_prompt ("Question: {q}\\nAnswer:"), so the
                              routed pool and this model see exactly the same string. Checked,
                              because a prompt-format difference between the two arms would have
                              silently become the result.
  * ROUGE                     rouge_scorer(["rougeL"], use_stemmer=True).recall — the audit's
                              metric, constructed the same way eval_ft_minimal does.

Sharded with --row_shard i/N so a condition can be spread over many GPUs; merge_plain_ft.py
stitches the pieces and refuses to score a set with holes in it.

  python eval_plain_ft.py --model_name locuslab/tofu_ft_llama2-7b \\
      --conditions original,name_stripped --row_shard 0/8 --out .../ft_shard0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "tofu_sisa_lora"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


def _parse_shard(spec: str) -> tuple:
    if not spec:
        return 0, 1
    i, n = spec.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise SystemExit(f"--row_shard {spec!r}: need 0 <= i < n")
    return i, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter_dir", default=None,
                    help="PEFT adapter dir. Omit for a full fine-tuned model.")
    ap.add_argument("--conditions", default="original,name_stripped",
                    help="Comma list from analyze_router_shift.CONDITIONS.")
    ap.add_argument("--row_shard", default="0/1", metavar="i/N")
    ap.add_argument("--attacker_id", type=int, default=0,
                    help="Must match finding 5's attacker (author 0) for the injection arms.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=100,
                    help="eval_ft_minimal.rouge_score's default, kept so ROUGE is comparable.")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Smoke: first N rows of the shard.")
    args = ap.parse_args()
    if not args.hf_home:
        raise SystemExit("--hf_home or $HF_HOME is required")
    os.environ["HF_HOME"] = args.hf_home

    from analyze_router_shift import build_eval_rows, build_conditions, CONDITIONS
    from eval_ft_minimal import build_prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rouge_score import rouge_scorer as _rs_lib

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; known: {list(CONDITIONS)}")

    shard_i, shard_n = _parse_shard(args.row_shard)

    full, rows, authors, paras = build_eval_rows(args.hf_home)
    cond, names, attacker_name = build_conditions(full, rows, authors, paras,
                                                  args.attacker_id, args.hf_home)
    is_forget = np.isin(authors, np.arange(180, 200))

    # Contiguous-stride sharding: every shard sees a mix of forget and retain rows, so a shard
    # that dies does not silently remove one whole class from the merged set.
    idx = np.arange(len(rows))[shard_i::shard_n]
    if args.limit:
        idx = idx[:args.limit]
    print(f"[plain_ft] model={args.model_name} adapter={args.adapter_dir or '(none)'}", flush=True)
    print(f"[plain_ft] shard {shard_i}/{shard_n}: {len(idx)} of {len(rows)} rows "
          f"({int(is_forget[idx].sum())} forget / {int((~is_forget[idx]).sum())} retain)",
          flush=True)
    print(f"[plain_ft] conditions={conds}  attacker={attacker_name!r}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter_dir, adapter_name="ft")
        model.set_adapter("ft")
    else:
        model = base
    model.eval()
    device = next(model.parameters()).device

    scorer = _rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
    out = {"meta": {"model_name": args.model_name, "adapter_dir": args.adapter_dir,
                    "conditions": conds, "row_shard": args.row_shard,
                    "n_rows_total": int(len(rows)), "n_rows_shard": int(len(idx)),
                    "attacker_id": args.attacker_id, "attacker_name": attacker_name,
                    "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                    "prompt": "Question: {q}\\nAnswer:"},
           "conditions": {}}

    t0 = time.time()
    for c in conds:
        recs = []
        for n_done, i in enumerate(idx):
            i = int(i)
            q = cond[c][i]
            row = int(rows[i])
            gold = full[row]["answer"]
            enc = tok(build_prompt(q), return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(ids[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            recs.append({
                "row": row,
                "author": int(authors[i]),          # the question's TRUE subject
                "is_forget": bool(is_forget[i]),
                "question": q,
                # csar.classify reads `gen_sibling` as "what was served"; there is no sibling
                # here, but keeping the key means the classifier is reused unmodified.
                "gen_sibling": gen,
                "rougeL_recall_vs_own_gold": float(scorer.score(gold, gen)["rougeL"].recall),
            })
            if (n_done + 1) % 25 == 0:
                el = time.time() - t0
                print(f"    [{c}] {n_done + 1}/{len(idx)}  {el / (n_done + 1):.2f}s/gen",
                      flush=True)
        out["conditions"][c] = recs
        print(f"[plain_ft] {c}: {len(recs)} rows, "
              f"mean ROUGE-L recall {np.mean([r['rougeL_recall_vs_own_gold'] for r in recs]):.4f}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[plain_ft] -> {args.out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
