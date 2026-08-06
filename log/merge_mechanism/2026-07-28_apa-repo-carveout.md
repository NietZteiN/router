### Target Date: 2026-07-28 (APA study packaged as a standalone, self-contained repo)

- **Hypotheses / what we're testing:** Engineering entry, not an experiment — no scientific
  hypothesis is settled here. The claim under test is an *infrastructure* one, and it is
  falsifiable: **H-REPO — the Experiment A/B/C code can be carved out of `tofu-unlearning` into a
  single directory that runs on a machine with none of the sibling research trees, no `/storage2`,
  and no SLURM.** CONFIRMED if every CPU gate passes from a fresh `git clone` into a directory
  with no siblings, and the drivers emit valid sbatch for a third site. REFUTED if any module
  needs a path outside the clone. Companion: **H-PORT — a cluster is a file, not a patch**;
  confirmed if `TOFU_SITE=<site> STUB=1` renders another cluster's job scripts correctly from
  here, refuted if any driver still hardcodes a site fact.

- **Setup:** New repo `/home/jack/apa-uniform-sum` (git, 8 commits, private, not yet pushed —
  `gh` is not installed on this box and no API token is present, so repo creation is pending).
  Carved from `/home/jack/tofu-unlearning` @ `6173f3f` through an explicit allow-list
  (`MANIFEST.files`, 62 vendored paths) applied by `sync_from_tree.sh`, direction tree→repo only.
  Contents: the 27-file import-time closure of A/B/C computed with `ast` (top-level vs
  function-level imports distinguished), plus `subspace_overlap.py` (the `overlap` stage) and
  `skill_data.py` (the Alpaca tier); the `cluster_env` site abstraction; 10 CPU gates; four
  reference tables (`nmerge_mu.csv`, `nmerge_subset_mu.csv`, `key_firing_e5.json`(+`.npz`),
  `expA_norms.json`); and this `log/merge_mechanism/` thread. 3.2 MB, 76 files at first commit.
  Verification run from `/tmp/.../clonetest`, a fresh clone whose only siblings are scratch dirs.
  Env: `test-env` (Python 3.12, torch 2.5.1+cu121, transformers 4.48.3, peft 0.14.0). No GPU
  was used; `squeue` empty throughout; nothing submitted.

- **Results:**
  - Fresh-clone gate: **10/10 CPU gates PASS** from the isolated clone —
    `test_repo_selfcontained, test_cluster_env, test_expa, test_merge_subset,
    test_ou_equivalence, test_eval_rows, test_merge_extra, test_expb_selectivity,
    test_mmlu_primitives, test_plot_style --colors-only`. `import eval_mmlu` succeeds with no
    `legonet_lora` anywhere on the filesystem near it.
  - `sync_from_tree.sh --check`: **62 same · 13 edited · 0 drift · 0 missing**.
  - Site rendering, `STUB=1`, same driver, three sites:
    `sprint` → `--partition=all --exclude=sprint4 --gres=gpu:1 --mem=48G`, arrays `%4`;
    `cispa` → `--partition=xe8545 --account=testing`, 5-node `--exclude`, **no `--mem` line**,
    arrays `%6`; `local` (from the clone, `TOFU_CKPT_ROOT` in `/tmp`) → `--partition=local`,
    no `--account`, no `--exclude`, arrays `%1`. CPU-only stages emit no `--gres` at every site.
  - `submit_expb.sh plan`: 16 merges, 38 score conditions, 5 targets, 20 authors; all 16 merge
    author-sets distinct (`uniq -d` empty).
  - Palette validation (six computable checks, both modes): 3 slots on the ALL-PAIRS list —
    light worst normal-vision ΔE **24.0** / CVD **9.2**, dark **20.9** / **9.4**; 8 slots on the
    ADJACENT list — light **19.6** / **9.1**, dark **19.3** / **8.4**. All PASS.
  - Fixed-weight drop-a-term exactness on the factor-cat algebra: max |err| **2.2e-16**;
    the 1/(N−1) renormalization differs from it by **0.501** on the same fixture.

- **What worked / hypothesis verdict:**
  - **H-REPO SUPPORTED.** 10/10 gates from a clone with no siblings, and `import eval_mmlu`
    works where it previously depended on a tree one directory up.
  - **H-PORT SUPPORTED.** One driver rendered three sites correctly, including the CISPA
    `--mem` policy, without a per-driver edit.
  - Five real defects were found *by building the gates*, each of which produced a plausible
    result rather than an error:
    1. `eval_mmlu.py` resolved `legonet_lora` as `dirname(dirname(__file__))/legonet_lora` —
       which on this machine **exists**, so every local run passed and only a clone would fail.
    2. `submit_nmerge.sh` read config paths without `expandvars`, so the portable
       `"${TOFU_CKPT_ROOT}/…"` form yielded a **literal** path and the driver reported a missing
       manifest at a directory whose name contained `${TOFU_CKPT_ROOT}`.
    3. Its `merge`/`eval`/`overlap` stages still hand-wrote `--partition=all` and `--mem=`, so
       3 of 6 stages would fail **at submit** on CISPA. (`norms` was already ported.)
    4. Six modules defaulted `HF_HOME` to `/storage2/jack/data/huggingface` — another cluster's
       disk, which does not fail elsewhere; it points HF at a missing directory and dies later
       as an offline-cache miss. Four of the six were found by the new gate, not by reading.
    5. The first `submit_expb.sh` design gave every target the same three sum4 companions, so
       all five `sum4_drop_a{X}` merges were **the same author set** — one artifact under five
       labels, i.e. one measurement presented as five, plus five redundant merges on disk.
  - Two chart defects were found by rendering the figures and looking at them: the palette
    **cycled** past 3 slots (figB1 painted a186 and a194 the same blue), and figC2's three tiers
    coincide *by construction* — that is the predicted result — so three direct labels printed on
    one pixel. Both are now structurally impossible: `slot_colors()` raises rather than wrapping,
    and `direct_labels()` de-collides in display space.

- **Observations:** The plan estimated this as "carve out ~19 files"; the measured import closure
  is 27 at import time and ~35 files shipped, because `eval_tofu.py` pulls `eval_progress`,
  `merge_lora`, `shard_utils` at module level and `merge_lora` pulls `merge_extra`/`tree_utils`.
  The thing that made a slim tree possible at all is that eval_tofu's *other-track* arms
  (legonet, ramole, sift, clamu, ctv, memsinks, prefix, routing — 16 modules) are imported
  **lazily**, inside the branch that needs them. That was luck rather than design, so it is now a
  gate (`test_absent_arms_are_lazy_only`) and `eval_tofu.py` stays byte-identical to the working
  tree apart from one HF_HOME default — which is what keeps `test_ou_equivalence.py` meaningful
  and every historical number comparable. No silent-failure checks apply (no model ran); the
  campaign itself is unchanged and unstarted.

- **New questions / new hypotheses:**
  - **H-REPO-2 (open):** does the carved-out tree stay in sync? The allow-list plus
    `--check` reports drift, but nothing *forces* a check. `merge-tables-7b` forked in both
    directions (52 files) with the same good intentions. Predict: without a scheduled
    `--check`, drift appears within ~2 weeks of active work in either tree.
  - **H-RANK (open):** `TOFU_MAX_EXACT_RANK=1024` is the A40 figure and is currently inherited
    by both other sites. On a 40 GB A100 a single-GPU task has *less* headroom, so the exact
    ladder may need to stop earlier than N=32 — or, at `TOFU_GPUS_PER_TASK=2` with
    `device_map="auto"`, considerably later. Unmeasured on that hardware.
  - Does `stage_hf_cache.sh`'s ~98% I/O-overhead figure reproduce on a different NFS? It is one
    measurement from one cluster on one day and is currently quoted as if general.

- **Next Steps:**
  1. Create the private GitHub repo and push (blocked on `gh`/token — one command once it exists).
  2. Re-derive `TOFU_MAX_EXACT_RANK` on the target card before any N≥64 merge; update the
     config's `_rank_ceiling` note rather than carrying the A40 number silently.
  3. Run the campaign: `submit_pool.sh anchors` → `pilot` → `r32`/`e25`, then the A/C ladder and
     the Exp-B chain, respecting the 4-GPU global cap (never two GPU arrays queued at once).
  4. Add a periodic `sync_from_tree.sh --check` so H-REPO-2 is observed rather than assumed.
