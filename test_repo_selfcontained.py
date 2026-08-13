#!/usr/bin/env python3
"""Gate: this repo resolves everything inside itself, on a machine that has never seen /home/jack.

    python test_repo_selfcontained.py

Standard library only. No GPU, no network, no model, no cluster, no torch.

WHY THIS EXISTS. "Portable" is a claim that decays silently, and it decays in a specific way:
a module resolves a sibling by absolute path, the developer's machine happens to have that path,
so every local run passes and only a fresh clone on another cluster fails — at the point where
someone is trying to run a job, not read a test. Every check below is a failure this repo
already had before the export, turned into something that fails HERE instead.

The layout contract being tested: the repo is FLAT. Every project sits at the root as a sibling,
exactly as it did in the home tree, so `dirname(dirname(__file__))/<proj>` is the same directory
`/home/jack/<proj>` used to be. Nothing may depend on that being true of the machine.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OK, FAIL = "ok  ", "FAIL"

PROJECTS = [
    "tofu_sisa_lora", "ramole", "legonet_lora", "sea", "sea_tofu",
    "sepmlp_tofu", "blocktc_tofu", "memsinks_tofu", "memadapt_tofu", "apa_uniform_sum",
]

# Prose, not code. A /home/jack path in the ledger is a historical record of where a job ran and
# must NOT be rewritten; one in an import is a portability bug.
PROSE_DIRS = {"log", "papers", "paper", ".git"}
# merge_tables_7b is another repo vendored verbatim; rewriting it would fork it (the exact
# mistake its own VENDOR_DRIFT.md documents). Its paths are documented in its own SETUP.md.
VERBATIM_DIRS = {"merge_tables_7b"}
# This file names the forbidden patterns, so it necessarily contains them.
SELF = os.path.basename(__file__)

# Files where an absolute path is the POINT, not a bug:
#   cluster_env.<site>.sh  a site file's whole job is to name that cluster's paths. Two sites
#                          (sprint, cispa) are shipped as worked examples; cluster_env.local.sh
#                          is the template and names nothing.
#   sync_from_tree.sh      points at the source trees it vendors FROM.
#   repo_env.py            documents which literal each variable replaced.
ABSOLUTE_PATH_OK = {"cluster_env.sprint.sh", "cluster_env.cispa.sh",
                    "sync_from_tree.sh", "repo_env.py"}

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{OK if ok else FAIL}] {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def walk_code(skip_verbatim: bool = True):
    skip = PROSE_DIRS | {"__pycache__", ".pytest_cache", "results_snapshot",
                         "S3T", "MemSinks", "open-unlearning"}
    if skip_verbatim:
        skip |= VERBATIM_DIRS
    for dp, dn, fn in os.walk(HERE):
        dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            if f.endswith((".py", ".sh")) and f != SELF:
                yield os.path.join(dp, f)


# ── 1. no absolute developer paths in code ────────────────────────────────────────────────────

def test_no_absolute_home_paths() -> None:
    pats = ("/home/jack", "/storage2/", "/home/c03jale")
    hits = []
    for p in walk_code():
        if os.path.basename(p) in ABSOLUTE_PATH_OK:
            continue
        src = open(p, encoding="utf-8", errors="surrogateescape").read()
        for pat in pats:
            if pat in src:
                for i, line in enumerate(src.splitlines(), 1):
                    if pat in line:
                        hits.append(f"{os.path.relpath(p, HERE)}:{i}")
    check("no /home/jack or /storage2 literal in code", not hits,
          f"{len(hits)} hit(s): " + ", ".join(hits[:5]))


# ── 2. every python file parses ───────────────────────────────────────────────────────────────

def test_all_python_parses() -> None:
    bad = []
    for p in walk_code(skip_verbatim=False):
        if not p.endswith(".py"):
            continue
        try:
            ast.parse(open(p, encoding="utf-8", errors="surrogateescape").read())
        except SyntaxError as e:
            bad.append(f"{os.path.relpath(p, HERE)}:{e.lineno}")
    check("every .py parses", not bad, f"{len(bad)}: " + ", ".join(bad[:5]))


# ── 3. every sibling lookup lands inside the repo ─────────────────────────────────────────────

def test_sibling_lookups_resolve() -> None:
    """`os.environ.get("X_DIR", os.path.join(_REPO_ROOT, "proj"))` must name a real sibling, and
    the _REPO_ROOT anchor must have the right dirname() depth for the file it sits in."""
    pat = re.compile(r'os\.environ\.get\("(\w+)",\s*os\.path\.join\(_REPO_ROOT,\s*"([\w.-]+)"')
    anchor = re.compile(r"^_REPO_ROOT = ((?:os\.path\.dirname\()+)os\.path\.abspath\(__file__\)")
    bad = []
    for p in walk_code():
        if not p.endswith(".py"):
            continue
        src = open(p, encoding="utf-8", errors="surrogateescape").read()
        targets = pat.findall(src)
        if not targets:
            continue
        rel = os.path.relpath(p, HERE)
        m = None
        for line in src.splitlines():
            m = anchor.match(line) or m
        if m is None:
            bad.append(f"{rel}: uses _REPO_ROOT with no anchor definition")
            continue
        depth = m.group(1).count("os.path.dirname(")
        want = len(rel.split(os.sep))          # <proj>/f.py -> 2, <proj>/tests/f.py -> 3
        if depth != want:
            bad.append(f"{rel}: anchor depth {depth}, needs {want}")
            continue
        for _env, proj in targets:
            # Upstreams are cloned by fetch_upstream.sh and are .gitignore'd, so "not present"
            # is a legitimate state; what must hold is that the path is INSIDE the repo.
            if proj in ("open-unlearning", "MemSinks", "S3T"):
                continue
            if not os.path.isdir(os.path.join(HERE, proj)):
                bad.append(f"{rel}: -> {proj}/ which does not exist")
    check("every sibling lookup resolves inside the repo", not bad, "; ".join(bad[:5]))


# ── 4. shell drivers anchor on BASH_SOURCE, not cwd ───────────────────────────────────────────

def test_shell_repo_root_anchor() -> None:
    bad = []
    for p in walk_code():
        if not p.endswith(".sh"):
            continue
        src = open(p, encoding="utf-8", errors="surrogateescape").read()
        if "${REPO_ROOT}" not in src:
            continue
        if "REPO_ROOT=" not in src and "cluster_env.sh" not in src:
            bad.append(os.path.relpath(p, HERE) + ": uses $REPO_ROOT without defining it")
    check("shell drivers define $REPO_ROOT before use", not bad, "; ".join(bad[:5]))


def test_sbatch_directives_not_shadowed() -> None:
    """sbatch stops reading #SBATCH at the first non-comment line, so an assignment inserted
    above a directive block silently disables every directive below it — no error, just a job
    on the wrong partition with no GPU. The export inserted a `REPO_ROOT=` line into 30 drivers,
    which is exactly the edit that could cause this.

    Only the PROLOGUE is examined — from the shebang to the first line of code. A driver's own
    directives can only live there, because that is the only region sbatch reads. Past it a
    `#SBATCH` is generated content: most drivers build a job script inside a heredoc or a quoted
    string, and flagging those is a false positive that buries the real thing (an earlier
    heredoc-tracking version fired on 30 files, all of them fine).
    """
    bad = []
    for p in walk_code():
        if not p.endswith(".sh"):
            continue
        lines = open(p, encoding="utf-8", errors="surrogateescape").read().splitlines()
        seen_code = False
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#SBATCH"):
                if seen_code:
                    bad.append(f"{os.path.relpath(p, HERE)}:{i}")
                break                      # the prologue ends at the first directive either way
            if s and not s.startswith("#"):
                if seen_code:
                    break                  # two code lines in: this file has no prologue block
                seen_code = True
    check("no #SBATCH directive shadowed by earlier code", not bad, "; ".join(bad[:5]))


# ── 5. the site layer works with nothing preset ───────────────────────────────────────────────

def test_local_site_resolves() -> None:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
           "TOFU_SITE": "local", "HF_HOME": "/tmp/_gate_hf", "TOFU_CKPT_ROOT": "/tmp/_gate_ck"}
    r = subprocess.run(["bash", "-c", f'source "{HERE}/cluster_env.sh" && '
                                      'echo "$TOFU_PYTHON|$TOFU_OU_PYTHON|$TOFU_ARRAY_CAP"'],
                       capture_output=True, text=True, env=env)
    ok = r.returncode == 0 and r.stdout.count("|") == 2 and "" not in r.stdout.strip().split("|")
    check("cluster_env.local.sh resolves from a bare env", ok, r.stderr.strip()[:120])


def test_missing_required_env_fails_loudly() -> None:
    """The failure mode this prevents: HF_HOME unset, the site file silently defaults to a path
    that does not exist, and the first job fails 40 minutes in with a cache miss."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
           "TOFU_SITE": "local"}
    r = subprocess.run(["bash", "-c", f'source "{HERE}/cluster_env.sh"'],
                       capture_output=True, text=True, env=env)
    check("unset HF_HOME fails loudly, naming the export",
          r.returncode != 0 and "HF_HOME" in r.stderr, r.stderr.strip()[:80])


def test_gpu_cap_ceiling_is_opt_in() -> None:
    """TOFU_GPU_CAP_CEILING clamps when set, and does nothing when it is not.

    Until 2026-08-06 a ceiling of 4 was hardcoded for every site — a sprint-cluster courtesy rule
    (~/CLAUDE.md §1) that silently overrode site files that knew their own scheduler's limits.
    What is worth gating now is the mechanism, not the number: an unset ceiling must honour the
    site's cap verbatim, and a set one must still clamp in EVERY cluster_env.sh, because the
    drivers that submit source their own project's copy through slurm_nodes.sh — a ceiling
    present only at the root is enforced nowhere that matters.
    """
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
            "TOFU_SITE": "local", "HF_HOME": "/tmp/_gate_hf", "TOFU_CKPT_ROOT": "/tmp/_gate_ck"}
    for rel in ["cluster_env.sh", "tofu_sisa_lora/cluster_env.sh",
                "apa_uniform_sum/cluster_env.sh"]:
        path = os.path.join(HERE, rel)
        if not os.path.isfile(path):
            continue
        cmd = ["bash", "-c", f'source "{path}" && echo "$TOFU_ARRAY_CAP"']

        r = subprocess.run(cmd, capture_output=True, text=True,
                           env={**base, "TOFU_ARRAY_CAP": "16"})
        check(f"no ceiling set => TOFU_ARRAY_CAP=16 is honoured by {rel}",
              r.stdout.strip() == "16", r.stdout.strip())

        r = subprocess.run(cmd, capture_output=True, text=True,
                           env={**base, "TOFU_ARRAY_CAP": "16", "TOFU_GPU_CAP_CEILING": "4"})
        check(f"ceiling 4 => TOFU_ARRAY_CAP=16 is clamped by {rel}",
              r.stdout.strip() == "4", r.stdout.strip())


def test_site_files_still_name_their_interpreter() -> None:
    """A site file that no longer names its cluster's interpreter is worse than useless.

    This is a regression the export actually caused: the de-absolutization pass rewrote
    `TOFU_PYTHON` inside cluster_env.sprint.sh from the real env down to a bare `python3`,
    because it treated a site file like any other script. Every driver then resolved to system
    python, and the failure surfaced 300 lines into one config-validation test as
    `ModuleNotFoundError: No module named 'torch'` — nowhere near the cause.

    A site file is the ONE place an absolute path belongs. `local` is exempt: it deliberately
    names nothing and takes everything from the environment.
    """
    bad = []
    for f in sorted(os.listdir(HERE)):
        if not (f.startswith("cluster_env.") and f.endswith(".sh")) or f in (
                "cluster_env.sh", "cluster_env.local.sh"):
            continue
        src = open(os.path.join(HERE, f), encoding="utf-8").read()
        m = re.search(r'TOFU_PYTHON="\$\{TOFU_PYTHON:-([^}]*)\}"', src)
        if not m:
            bad.append(f"{f}: no TOFU_PYTHON default")
        elif not m.group(1).startswith("/"):
            bad.append(f"{f}: TOFU_PYTHON defaults to {m.group(1)!r}, not an absolute path")
    check("named site files still carry their absolute interpreter", not bad, "; ".join(bad))


def test_every_project_slurm_env_sources():
    ok_all, bad = True, []
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.path.expanduser("~"),
           "TOFU_SITE": "local", "HF_HOME": "/tmp/_gate_hf", "TOFU_CKPT_ROOT": "/tmp/_gate_ck"}
    for proj in PROJECTS:
        f = os.path.join(HERE, proj, "slurm_nodes.sh")
        if not os.path.isfile(f):
            continue
        r = subprocess.run(["bash", "-c", f'source "{f}"'], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            ok_all = False
            bad.append(f"{proj}: {r.stderr.strip()[:60]}")
    check("every project's slurm_nodes.sh sources cleanly", ok_all, "; ".join(bad[:4]))


# ── 6. the offline evidence is intact ─────────────────────────────────────────────────────────

def test_results_snapshot_intact() -> None:
    r = subprocess.run([sys.executable, os.path.join(HERE, "snapshot_results.py"), "--check"],
                       capture_output=True, text=True)
    check("results_snapshot matches its MANIFEST.tsv", r.returncode == 0,
          r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:100])


def test_upstream_pins_are_full_shas() -> None:
    src = open(os.path.join(HERE, "fetch_upstream.sh")).read()
    pins = re.findall(r"\|([0-9a-f]{7,64})\|", src)
    check("fetch_upstream.sh pins are full 40-char SHAs",
          len(pins) == 3 and all(len(p) == 40 for p in pins), str(pins))


# The ONE place under paper/ where LaTeX is allowed: the follow-up draft
# ("Deleted from the Router, Not from the Model"). It is our own unsubmitted work, carries no
# distribution notice, and cites only results already public in reports/. Everything else under
# paper/ stays banned. See test_manuscript_absent for what still guards this directory.
FOLLOWUP_DIR = "paper/followup/"


def test_manuscript_absent() -> None:
    """The AAAI submission must NEVER be committed to this repo.

    It is anonymized and under review, and its own notice reads "Distribution, citation, or
    public sharing of this manuscript is strictly prohibited". This repo is public. Both the
    PDFs and the LaTeX sources are the manuscript — dropping one while publishing the other
    would be self-defeating, so both are gitignored and checked for here. `paper/README.md`
    (the claim → code map) stays: it cites results, which are already public in reports/.

    Checked on disk AND in the index, because .gitignore does not apply to a file that is
    already tracked — which is exactly how these got committed in the first place.

    NARROWED 2026-08-13, deliberately and with the hole named. `paper/followup/` holds the
    follow-up draft and is exempt from the .tex ban, so this check would have gone from
    "no LaTeX under paper/, ever" to "no LaTeX except in one directory" — which is a weaker
    invariant that an accidental `cp` into that directory could defeat. Three things keep it
    strong instead of merely narrower:
      * .pdf is still banned EVERYWHERE under paper/, exempt directory included. A built PDF is
        not a source file, and the AAAI artifact most likely to be copied around is a PDF.
      * the exempt directory's own .tex files are read and rejected if they carry the
        submission's prohibition notice or its filename marker.
      * everything outside the exempt directory is unchanged.
    """
    strays, notices = [], []
    for root, _dirs, files in os.walk(os.path.join(HERE, "paper")):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), HERE)
            exempt = rel.replace(os.sep, "/").startswith(FOLLOWUP_DIR)
            if f.endswith(".pdf") or "p_unlearn" in f.lower():
                strays.append(rel)                      # banned everywhere, exemption included
            elif f.endswith(".tex") and not exempt:
                strays.append(rel)
            elif f.endswith(".tex"):
                body = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
                if "strictly prohibited" in body or "p_unlearn" in body.lower():
                    notices.append(rel)                 # the submission wearing a new path
    r = subprocess.run(["git", "-C", HERE, "ls-files"], capture_output=True, text=True)
    tracked = [l for l in r.stdout.splitlines()
               if (l.startswith("paper/") and l != "paper/README.md"
                   and not l.startswith(FOLLOWUP_DIR))
               or "p_unlearn" in l.lower()]
    check("the manuscript is absent from disk and from the index",
          not strays and not tracked and not notices,
          f"on disk: {strays[:3]}  tracked: {tracked[:3]}  carries the notice: {notices[:3]}")


def main() -> int:
    print("Repo self-containment gate\n")
    for fn in sorted(
        (v for k, v in globals().items() if k.startswith("test_") and callable(v)),
        key=lambda f: f.__code__.co_firstlineno,
    ):
        fn()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: " + ", ".join(failures))
        return 1
    print("all checks passed — this repo runs from a bare clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
