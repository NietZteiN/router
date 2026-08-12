# router

The complete research tree behind **Router-Free Modular Storage for Knowledge Unlearning in Large
Language Models** (MUSR) — code, configs, results, ledger and manuscript — packaged to `git clone`
onto another cluster and keep running.

> **The manuscript is not in this repo.** It is an anonymized AAAI submission under review, whose
> notice forbids distribution, so neither the PDFs nor the LaTeX sources are committed and
> `.gitignore` keeps them out. [`paper/README.md`](paper/README.md) maps each paper claim to the
> code and result file that produces it — which is the part that is useful here anyway.

## The result this exists to support

| | |
|---|---|
| Oracle routing over 200 per-author task vectors (7B, e25) | `model_utility` **0.8236** — best of any track in the project |
| Author deletion under that routing | **Δmu = 0.0000**, forget-ppl 1.05 → 17.72 |
| Post-deletion router leak | **400/400** orphan queries reassigned to a surviving expert, by every one of 12 routers. Confidence refusal caps at AUC 0.57–0.61; only an explicit deletion record reaches 0.98 |
| Replacing the oracle with a real embedding router | −0.064 utility *before* any deletion is requested |

That is the case the paper makes for removing the router entirely. This repo is the evidence, the
comparators it is measured against, and the tooling to re-run all of it.

## Start here

| you want to | read |
|---|---|
| see what the paper claims and which file produces it | [`paper/README.md`](paper/README.md) — a per-claim map from every table to its code, report and result file |
| run it on a new cluster | [`SETUP.md`](SETUP.md), then [`PORTING.md`](PORTING.md) |
| know what is verifiable offline vs needs GPUs | [`STATUS.md`](STATUS.md) |
| know where each file came from | [`PROVENANCE.md`](PROVENANCE.md) |
| see what is deliberately absent, and the open gaps | [`STATUS.md`](STATUS.md) |
| read the research narrative | [`log/README.md`](log/README.md) — 136 dated entries, 19 threads, hypothesis-before-run |
| see where the **follow-up paper** stands | [`SELECTOR_AUDIT_REPORT.md`](SELECTOR_AUDIT_REPORT.md) — *Deleted from the Router, Not from the Model*: findings per section with their load-bearing caveats, what is blocked, and the method constraints that govern how each number may be read |

## Layout — flat siblings

Every project sits at the root, exactly as it did in the home tree. That is not cosmetic: modules
resolve each other as `dirname(dirname(__file__))/<project>`, so the flat layout *is* the import
contract. [`test_repo_selfcontained.py`](test_repo_selfcontained.py) enforces it.

```
tofu_sisa_lora/   the core tree — 134 modules, 43 configs, 62 SLURM drivers, 35 CPU gates,
                  38 reports. Router core, leak audit, scaffold routing, SISA, S3T, merge
                  mechanism, composable task vectors, SIFT/ClAMU masks, attacks, eval spine.
ramole/           RAMoLE — LoraRetriever + per-layer RouterLoRA cross-attention gate
legonet_lora/     LegoNet — keyed adapter bank, MiniLM centroids, top-3 kNN
sea/              BART-MNLI zero-shot domain router (a 10th router family, absent from router.py)
sea_tofu/         SEA on TOFU — per-author experts, rank-vs-deletability
sepmlp_tofu/      MUSR itself — per-source gated FFN adapters, L_out suppression, no router
blocktc_tofu/     block-transcoder successor, 11.8x smaller
memsinks_tofu/    MemSinks routed masks
memadapt_tofu/    product-key memory router with -inf block-list deletion
apa_uniform_sum/  APA / uniform summation, no router — Exp A/B/C
merge_tables_7b/  the 7B k=200 result tables + a stdlib-only verification harness
paper/            claim -> code map for the MUSR paper (the manuscript itself is NOT here)
papers/           22 reference PDFs
log/              the dated research ledger
results_snapshot/ 476 result files (42 MB) — every CPU analysis runs with no GPU and no scratch FS
ou_integration/   open-unlearning fork patches, including 3 files that exist nowhere else
```

## Quickstart

```bash
git clone git@github.com:NietZteiN/router.git && cd router
pip install -r tofu_sisa_lora/requirements.txt

export TOFU_SITE=local
export HF_HOME=$HOME/.cache/huggingface       # must contain hub/
export TOFU_CKPT_ROOT=$HOME/tofu_checkpoints  # needs room for the pools

python test_repo_selfcontained.py             # 13 checks, no GPU, no network
bash fetch_upstream.sh                        # S3T / MemSinks / open-unlearning, at pinned commits
```

Porting to a named cluster is **one file**: copy `cluster_env.local.sh` to
`cluster_env.<yoursite>.sh` and fill in partition, account and paths. Details in
[`PORTING.md`](PORTING.md).

## What is not here

Model weights and adapter pools — 674 GB for `tofu_sisa_lora` alone, ~90 GB across the six sibling
stores. [`STATUS.md`](STATUS.md) says, per result, whether it is checkable from
`results_snapshot/` or needs a retrain, and what the retrain costs.

## Keeping it in sync with the working tree

The code was vendored through an allow-list, not an `rsync -a`. The repo it replaces did the
blanket copy and forked: *"52 files differ, 16 only in the repo, 220 only in home … the repo copy
is not a stale snapshot, it is a fork"*
([VENDOR_DRIFT.md](merge_tables_7b/reproduce/VENDOR_DRIFT.md)).

```bash
bash sync_from_tree.sh --check    # report drift, write nothing (exit 1 if any)
bash sync_from_tree.sh --pull     # tree -> repo, unedited entries only
```

Every vendored `.py`/`.sh` is marked `edited:deabsolutized` in [`MANIFEST.files`](MANIFEST.files),
because the export rewrote its hard-coded paths. `--check` reports that as EDITED, not DRIFT, so a
future sync is a deliberate merge rather than a silent overwrite of the port.
