### Target Date: 2026-08-06 (the export went public — manuscript withheld, history rewritten)
- **Hypotheses / what we're testing:** Correction + completion of
  [2026-08-06_routing-repo-export.md](2026-08-06_routing-repo-export.md), whose "Next Steps" said
  *push to an empty **private** repo `NietZteiN/tofu-routing`*. Both halves of that turned out to
  be wrong. No research hypothesis; the operational question is whether the repo can be published
  **without** disclosing an anonymized submission that is still under review.
- **Setup:** The user created `https://github.com/NietZteiN/router` — a different name, and
  `GET https://api.github.com/repos/NietZteiN/router` returned **HTTP 200 unauthenticated**
  ⇒ `"private": false, "visibility": "public"`. Checked *before* pushing, not after. Remote
  repointed to `git@github.com:NietZteiN/router.git`. Decisions taken with the user: drop the
  AAAI submission (both the two PDFs **and** the three `.tex` sources — publishing the LaTeX while
  withholding the PDF would disclose the same prose); **keep** `papers/` (22 third-party publisher
  PDFs) at the user's explicit call. Single commit `857afe7`, 2,357 files, 95.43 MiB.
- **Results:**
  - **A late-delete is not a delete.** The manuscript was in the first commit (`7f74118`), so
    `git rm` in a follow-up commit still ships the blobs — `git log --all --name-only` listed all
    five. Nothing had been pushed, so history was rebuilt from scratch:
    `git update-ref -d refs/heads/main` → one fresh commit → `reflog expire --expire=now --all` →
    `gc --prune=now`.
  - **Purge verified by content hash, not by absence of a filename:** `git hash-object` on each
    original file, then `git cat-file -e` against the object store — all five report **absent**.
  - **Published tree audited from outside:** the GitHub trees API returns 2,636 paths;
    `paper/` contains exactly `paper/README.md`; grep for `p_unlearn|paper/(pdf|tex)` ⇒ **0
    violations**.
  - New gate `test_manuscript_absent` replaces `test_paper_present` and checks **both** the
    working tree and `git ls-files` — `.gitignore` does not apply to an already-tracked file,
    which is precisely how these got committed. Self-containment now **13/13**.
  - Docs re-cut for a public repo: `README.md`, `STATUS.md`, `PROVENANCE.md`, `paper/README.md`,
    `MANIFEST.files`, `.gitignore`.
- **What worked / hypothesis verdict:** Goal **MET** — the repo is public, the code/results/ledger
  are all there, and the submission is not, in the tree *or* in history. The prior entry's
  "private repo" plan is **superseded**: it was written before the repo existed and assumed a
  visibility that was never chosen.
- **Observations:**
  - **Checking visibility cost one unauthenticated curl and would have been unrecoverable to skip.**
    "The user said they'd make it private" is not evidence about a URL they hand you later. The
    only reason this was caught is that the push was gated on an explicit check rather than on the
    earlier conversation.
  - The instinct to drop only the PDFs was wrong and the user agreed on being shown why: the
    `.tex` files *are* Appendix D/E. Withholding the compiled artifact while publishing its source
    is a null defence — and worse than useless, because the LaTeX is greppable.
  - `paper/README.md` (the claim → code map) survives the removal intact, because it cites
    **results**, which live in `tofu_sisa_lora/reports/` and `log/` and are public anyway. The map
    was always the useful half; the manuscript was the redundant half.
  - `papers/` stays public by the user's decision. It is a copyright question rather than a
    confidentiality one, and it is recorded as such in `STATUS.md` rather than silently.
- **New questions / new hypotheses:**
  - The ledger (`log/`, 136 entries) and `reports/` are now public and contain every number the
    paper reports, including unpublished arms. That is a deliberate consequence, not an oversight
    — but it is worth a decision before the paper's camera-ready, not after.
  - Should `sync_from_tree.sh --check` also assert the manuscript's absence, so a future `--pull`
    cannot silently re-vendor `tofu_sisa_lora/paper/*.tex`? The gate catches it, but only if the
    gate is run.
- **Next Steps:**
  1. `git push` is **done** — `https://github.com/NietZteiN/router`, branch `main`, `857afe7`.
  2. On the new cluster: `git clone`, `cp cluster_env.local.sh cluster_env.<site>.sh`, fill four
     values, `python test_repo_selfcontained.py`, `bash fetch_upstream.sh`.
  3. If the paper is accepted and de-anonymized, the manuscript can be added back — the
     `.gitignore` entries and the gate are the only things to reverse.
  4. Unchanged from the prior entry: GRAM / NULL / SGTM have no implementation in this tree, so
     main-paper Tables 1–2 are not fully reproducible from it; MUSE is absent.
