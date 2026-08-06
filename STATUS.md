# STATUS — what is verifiable here, and what is not

Honest coverage. Every row says whether a result can be re-derived **on this repo alone** (no GPU,
no scratch filesystem, no model weights) or whether it needs something that is not in git.

## What is deliberately not in this public repo

**The manuscript.** `paper/pdf/` and `paper/tex/` are gitignored. The AAAI submission is
anonymized and under review, and its own notice reads *"Distribution, citation, or public sharing
of this manuscript is strictly prohibited."* Both the compiled PDFs and the LaTeX sources are the
manuscript, so both are excluded — dropping one while publishing the other would be pointless.
[`paper/README.md`](paper/README.md) carries the claim → code map instead, citing results that are
already in `tofu_sisa_lora/reports/` and `log/`.
[`test_repo_selfcontained.py`](test_repo_selfcontained.py) fails if either ever reappears, on disk
or in the git index.

**Model weights.** 674 GB for `tofu_sisa_lora` plus ~90 GB across the six sibling stores. See the
retrain costs below.

`papers/` — 22 third-party reference PDFs — **is** committed, at the repo owner's decision.

## Gate results at export

Run on the export machine, `TOFU_SITE=sprint`, `requirements.txt` env.

| suite | result |
|---|---|
| `test_repo_selfcontained.py` (layout, site layer, snapshot, pins) | **13 / 13** |
| `tofu_sisa_lora` — **all 35 gates** | **34 passed, 1 failed** |
| — of which the routing gates (router, family, leak, audit, scaffold, groupb, lazy-adapters, legonet, ramole, eval-rows, cluster-env) | **11 / 11** |
| `sepmlp_tofu/tests` (MUSR) | **77 passed, 1 skipped** |
| `blocktc_tofu/tests` | **91 passed, 1 skipped** |
| `memadapt_tofu/tests` | **24 passed** |
| `legonet_lora/tests` | **9 passed** |
| `memsinks_tofu/test_memsinks.py` | **22 gates green** |
| `ramole/tests` (6 standalone scripts) | **6 / 6** |
| `snapshot_results.py --check` | **476 / 476 files match their sha256** |

**The one failure is pre-existing, not a port regression.** `test_plot_author_tsne.py` fails
identically in the source tree: it imports `matplotlib`, which `requirements.txt` deliberately
omits (plots run under `$TOFU_PLOT_PYTHON` / `requirements-plots.txt`). Install those and it
passes.

Two suites depend on upstream clones and fail **by design** until `bash fetch_upstream.sh` has
run: `memsinks_tofu/test_memsinks.py` (hashes MemSinks' `src/src/SeqTDModel.py` as its reference)
and three `memadapt_tofu/tests/test_data.py` cases (tokenizer parity against open-unlearning).
Both pass when pointed at an existing clone via `MEMSINKS_UPSTREAM_DIR` / `OU_DIR`.

**The open-unlearning pin is unreachable (found 2026-08-06).** `fetch_upstream.sh` pins
`93e9cd5d8808bb43641d133b38bb34466f9aae2e`, but `locuslab/open-unlearning` answers *"upload-pack:
not our ref"* for it: it is a fork-local commit that was never pushed, not a stale clone. S3T and
MemSinks check out at their pins normally. Consequences, in order of severity:

* `ou_integration/patches/model__init__.diff` cannot apply — its second hunk anchors on the
  fork-only `from model.memadapt_registry import …` lines, which upstream main does not have.
  `fetch_upstream.sh` now **skips the patch step entirely** rather than installing
  `sepmlp_registry.py` into an upstream tree and leaving something that looks integrated.
* **The OU eval track — the Table-1 `Agg`/`Priv` rows of sepmlp / blocktc / memadapt — is not
  reproducible from this repo** until that fork commit is recovered. This belongs with GRAM /
  NULL / SGTM in "Known gaps" below, not with the by-design failures above.
* The three `memadapt_tofu/tests/test_data.py` tokenizer-parity cases still pass against an
  upstream-main clone; they do not depend on the fork's contents.

## Verifiable offline — no GPU, no `/storage2`, no HF token

[`results_snapshot/`](results_snapshot/) carries 476 result files (42 MB) from the pools the paper
cites, including the raw `.npz` score matrices, so the analyses recompute rather than just read
back.

| paper table | what the snapshot covers |
|---|---|
| Appendix E Tables 9–10 (orphan destinations) | the full k=10 1B scaffolded battery — 12 routers × drop-audit, plus the tombstone ladder and the ROC/AUC arms |
| Appendix E (7B scale + granularity) | 7B k=10 feature + behavioral, 7B k=50, 1B-plain de-confound |
| Appendix E (the oracle row, 0.8236) | the 7B k=200 e25 pool's routed/deletion evals |
| Appendix E (keyed banks, learned gate) | legonet n=32 audits, RouterLoRA ×3 seeds, base-pin arms |
| Appendix E (RAMoLE / DBpedia) | the RAMoLE run results |
| Appendix D Table 6 (merged vs routed ladder) | 7B pools at k = 4/10/20/50/100/200 |
| Appendix D Tables 7–8 | SIFT / ClAMU-K200 and the four PEFT bake-off pools |

Plus [`merge_tables_7b/reproduce/`](merge_tables_7b/reproduce/) — `verify_report.py` and
`rebuild_tables.py` re-derive the 7B merge-vs-routing master report **with the Python standard
library alone**.

## Needs GPUs and a checkpoint store

Everything that *produces* rather than *analyses*. The adapter pools are 674 GB for
`tofu_sisa_lora` plus ~90 GB across the six sibling stores, and are not in git.

| to re-run | needs |
|---|---|
| any routed eval on a new pool | train the pool first: 200 per-author LoRAs, r32/e25, 7B — the largest single cost in the repo |
| the router-leak battery on a new pool | the pool + one encoder pass per query per router; the four behavioral routers run *every* expert on *every* query and become impractical past ~50 sources |
| MUSR / sepmlp / blocktc training | 1B frozen base + the per-source bank |
| the OU-track Table-1 rows | the second conda env **and** an open-unlearning clone |

## Known gaps — stated, not hidden

**Baselines in the paper with no code in this repo.** Main-paper Tables 1–2 include `GRAM`,
`NULL`, `SGTM`, `NPO`, `SimNPO` and `RMU`. NPO / SimNPO / RMU are open-unlearning methods and are
reachable once `fetch_upstream.sh` has cloned the fork. **GRAM, NULL and SGTM were run outside
this tree and have no implementation here** — those three rows cannot be reproduced from this repo
as it stands.

**The OU eval track.** NPO / SimNPO / RMU and the Table-1 `Agg`/`Priv` rows need the
open-unlearning fork at `93e9cd5`, which is not on the remote — see "Gate results at export"
above. Everything else in the routing and merging tree is independent of it.

**MUSE.** Appendix C's MUSE-News results are not reproducible here either: the benchmark, its
Llama-2-7B fine-tuned/retrained targets, and any MUSE loader are all absent. TOFU is fully covered.

**`sea/`** pulls `facebook/bart-large-mnli` (1.6 GB) on first use, so its router needs network
access on the new cluster even though nothing else does.

**Result files are not rewritten.** Paths inside `results_snapshot/**` still name the original
scratch filesystem. That is deliberate — they are provenance written by the job that produced
them, and rewriting them would falsify the record. The same applies to `log/`.

## Next steps this repo is set up for

- **H-k200-scaf** — per-author experts trained on the scaffolded base + oracle routing, testing
  whether the 0.8236 headline moves. Config and driver are in place; needs the pool.
- Extended + multi-seed replication of the 0.8236 cell (it is currently a single seed).
- The Alpaca-replay matched-FT control for the routing-scaffold arm.
- Bringing GRAM / NULL / SGTM in-tree, which is what would close the Table-1/2 gap above.
