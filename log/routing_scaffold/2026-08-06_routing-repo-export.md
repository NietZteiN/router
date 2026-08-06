### Target Date: 2026-08-06 (portable export of the whole routing tree → `tofu-routing`)
- **Hypotheses / what we're testing:** Infrastructure, not an experiment — no falsifiable
  hypothesis about unlearning. The question is operational: **can the routing/selection tree be
  moved to another cluster and still run?** Success bar stated before the work: a bare `git clone`
  on a machine with no `/home/jack` and no `/storage2` (a) imports every module, (b) passes every
  CPU gate, (c) re-derives the router-leak tables from committed data with no GPU, and (d) needs
  exactly one new file to target a new SLURM site. Failure = any surviving absolute path, any gate
  that only passes because the developer's filesystem happens to exist.
- **Setup:** New repo at `~/tofu-routing`, commit `7f74118`, remote
  `git@github.com:NietZteiN/tofu-routing.git` (not yet pushed — the empty GitHub repo is the
  user's to create). 2,361 files, 586,430 insertions, 96.06 MiB packed. Vendored from four roots
  via an allow-list (`MANIFEST.files` + `sync_from_tree.sh --check`), **not** `rsync -a`:
  `$TOFU_TREE=/home/jack/tofu-unlearning` (tofu_sisa_lora, log, paper/tex), `$TOFU_HOME`
  (ramole, legonet_lora, sea, sea_tofu, sepmlp_tofu, blocktc_tofu, memsinks_tofu, memadapt_tofu,
  papers, paper/pdf, the OU patches), `$APA_TREE=/home/jack/apa-uniform-sum` (whole repo + the
  site layer), `$MT7B_TREE=/home/jack/merge-tables-7b` (only `reproduce/` + 6 owned docs).
  Layout is FLAT — every project at the root — because modules resolve siblings as
  `dirname(dirname(__file__))/<proj>`, so `<repo>/<proj>` is exactly where `/home/jack/<proj>`
  was. Third-party upstreams pinned rather than copied (`fetch_upstream.sh`): S3T `3ecb73b`,
  MemSinks `a005119`, open-unlearning `93e9cd5`. Gates run with the `test-env` interpreter,
  `TOFU_SITE=sprint`.
- **Results:**
  - **Path rewrites:** 113 files had `/home/jack/<proj>` → repo-relative sibling + env override;
    246 files (of which **212 are configs**) had `/storage2/jack/...` → `${HF_HOME}` /
    `${TOFU_CKPT_ROOT}` / `${TOFU_CKPT_STORE}` / `${TOFU_DATA_ROOT}`. Residue in code: **0**.
  - **Gates:** self-containment **12/12**; `tofu_sisa_lora` routing gates **11/11**;
    `sepmlp_tofu` **77 passed, 1 skipped**; `blocktc_tofu` **91 passed, 1 skipped**;
    `memadapt_tofu` **24 passed**; `legonet_lora` **9 passed**; `memsinks_tofu` **22 gates green**;
    `ramole` **6/6**.
  - **Offline evidence:** `results_snapshot/` = **476 files, 41.9 MB**, `--check` **476/476**
    sha256 match. Covers the k=10 orphan battery, the 7B k=200 oracle cell, 7B k=10/k=50 coverage,
    legonet + RouterLoRA×3 seeds, the merged-vs-routed k-ladder, SIFT/ClAMU, the PEFT bake-off,
    and the RAMoLE/DBpedia runs. Zero pools skipped.
  - **Vendor drift:** `sync_from_tree.sh --check` → **same 1490, edited 328, drift 0, missing 0**.
  - **Site layer:** `TOFU_ARRAY_CAP=16` → clamped to 4 with a stderr notice; unset `HF_HOME` →
    hard failure naming the export; all 6 sibling `slurm_nodes.sh` source the site file cleanly.
- **What worked / hypothesis verdict:** All four bars **MET**. (a) every `.py` parses and the
  sibling-resolution check passes; (b) all suites green; (c) the snapshot re-derives with the
  stdlib; (d) porting is `cp cluster_env.local.sh cluster_env.<site>.sh`. The allow-list vendoring
  worked as intended — drift 0 on a 1,800-file surface.
- **Observations:**
  - **The `/storage2` hardcoding was the bigger problem, and it was nearly missed.** The first
    sweep only looked for `/home/jack` and came back clean; the gate then found **145 code files +
    212 configs** naming the scratch filesystem. Configs were the dangerous half: a config with a
    dead absolute path loads fine, submits fine, and fails forty minutes into a run.
  - **`expandvars` silently no-ops on an unset variable**, which is why `repo_env.expand_paths`
    raises instead. This is not hypothetical — a literal `${TOFU_CKPT_ROOT}` directory is sitting
    in `apa-uniform-sum` today, created exactly this way.
  - **Three of my own automated rewrites broke code and were caught by the gates, not by review.**
    (i) inserting an anchor after the last top-level import put it *inside* a multi-line
    `from x import (…)` — 8 syntax errors; fixed by using AST `end_lineno`. (ii) an anchor placed
    below its first use — 13 files. (iii) a regex that moved `_ensure_site_env()` inside a
    `with open(path)` block, leaving `json.load(f)` outside it — "I/O operation on closed file" in
    9 `load_config`s, which only surfaced because `blocktc_tofu` has a config-validation test.
    Mechanical edits at this scale need a gate that runs the code, not just one that parses it.
  - The `#SBATCH`-shadowing check fired on 30 files as a false positive (directives inside
    heredocs that *generate* job scripts). Narrowed to the file prologue, which is the only region
    sbatch reads.
  - **(iv) The worst one: the rewrite gutted the site files themselves.** `cluster_env.sprint.sh`
    had its `TOFU_PYTHON` default rewritten from the real env down to a bare `python3`, because
    the pass treated a site file like any other script. A site file is the *one place* an absolute
    path belongs. Every driver then resolved to system python, and the failure surfaced 300 lines
    into `test_analyze_ctv.py` as `ModuleNotFoundError: No module named 'torch'` — nowhere near
    the cause, and only visible because that one test executes a driver end-to-end. Fixed by
    restoring the four per-project site files verbatim and adding
    `test_site_files_still_name_their_interpreter` to the gate.
  - `test_plot_author_tsne.py` fails here **and in the source tree** (matplotlib is deliberately
    absent from `test-env`; plots run under `$TOFU_PLOT_PYTHON`). Pre-existing, not a regression.
  - `~/sea` turned out to be a distinct project from `sea_tofu` — a BART-MNLI zero-shot domain
    router, a 10th router family not in `router.py`'s inventory. It would have been dropped by a
    name-based sweep.
- **New questions / new hypotheses:**
  - Does the `edited:deabsolutized` marker scale? 328 files now differ from the tree by
    construction. A future `--pull` is a 328-file merge; at some point the port should flow *back*
    home so the two converge.
  - The three uncommitted open-unlearning files (`sepmlp_registry.py`, the yaml, the
    `__init__.py` fp32 diff) were reachable from no git remote. What else in the working tree is
    load-bearing and uncommitted?
- **Next Steps:**
  1. User creates the empty private repo, then `git push -u origin main` (96 MiB).
  2. **Private, not public** — `paper/pdf/` is an anonymized submission under review whose notice
     forbids distribution, and `papers/` is 22 publisher PDFs.
  3. On the new cluster: `cp cluster_env.local.sh cluster_env.<site>.sh`, fill four values, run
     `test_repo_selfcontained.py` then `fetch_upstream.sh`.
  4. Close the coverage gap `STATUS.md` names: GRAM / NULL / SGTM have no implementation in this
     tree, so main-paper Tables 1–2 cannot be fully reproduced from it; MUSE is absent entirely.
