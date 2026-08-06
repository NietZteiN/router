"""Gate: this repo runs from a single directory, with nothing but requirements.txt installed.

"Self-contained" is a claim that decays silently. The specific way it decayed before was
`eval_mmlu.py` resolving a sibling `legonet_lora/` tree as dirname(dirname(__file__))/... —
which on the development machine happened to exist one level up, so every local run passed and
only a fresh clone failed. This gate makes that class of failure fail HERE.

    python test_repo_selfcontained.py

No GPU, no network, no model, no cluster.
"""
from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OK = "ok  "

# Plot scripts import matplotlib, which requirements.txt deliberately omits (it runs under
# ${TOFU_PLOT_PYTHON}). They are import-checked in a separate, tolerant pass.
PLOT_MODULES = {"plot_style", "plot_nmerge", "plot_expa", "plot_expb", "plot_expc"}

# Arms of eval_tofu / merge_lora / train_lora_shard that belong to OTHER research tracks
# (legonet, ramole, sift-masks, clamu, composable-tv, memsinks, prefix-compose, routing,
# entangled-facts). They are deliberately NOT vendored — see ARMS.md. The contract that makes
# that safe is not "they are unused" but "they are imported LAZILY", inside the branch that
# needs them, so `import eval_tofu` never touches them and only passing the corresponding CLI
# flag does. test_absent_arms_are_lazy_only enforces exactly that.
ABSENT_ARMS = {
    "clamu_model", "ds_support", "ensemble", "entangle_data", "legonet_model", "legonet_tofu",
    "linear_tv", "memsinks_routed_model", "prefix_concat", "ramole_tofu", "router",
    "sift_masks", "sift_masks_data", "sift_masks_model", "train_ds_support", "train_sift_masks",
}


def _repo_modules():
    return sorted(f[:-3] for f in os.listdir(HERE)
                  if f.endswith(".py") and not f.startswith("_"))


def test_no_absolute_home_paths():
    """No source file may hardcode a path into a specific user's home or storage volume.

    Comments and docstrings are exempt — several of them legitimately record WHERE a measurement
    was taken. Only live code is checked, by walking the AST's string constants.
    """
    bad = []
    # This file necessarily CONTAINS the patterns it searches for. Scanning itself would make
    # the gate permanently red for the wrong reason.
    for m in (x for x in _repo_modules() if x != "test_repo_selfcontained"):
        p = os.path.join(HERE, m + ".py")
        try:
            tree = ast.parse(open(p).read())
        except SyntaxError as e:
            bad.append(f"{m}.py: unparseable ({e})")
            continue
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v in docstrings:
                    continue
                for pat in ("/home/jack", "/storage2/", "/home/c03jale"):
                    if pat in v:
                        bad.append(f"{m}.py:{node.lineno}: {pat!r} in {v[:70]!r}")
    assert not bad, ("hardcoded machine paths in live code:\n  " + "\n  ".join(bad) +
                     "\n  Resolve via tofu_env.hf_home() / ${TOFU_CKPT_ROOT} instead.")
    print(OK + f"no hardcoded /home/<user> or /storage2 paths in {len(_repo_modules())} modules")


def test_imports_without_sibling_trees():
    """Every module imports with ONLY this directory on sys.path.

    Run in a SUBPROCESS with a scrubbed sys.path: importing in-process would let modules already
    resolved by the test runner mask a missing dependency, which is exactly how the legonet
    coupling stayed invisible.
    """
    mods = [m for m in _repo_modules() if m not in PLOT_MODULES]
    code = (
        "import importlib, sys, os\n"
        f"sys.path = [p for p in sys.path if os.path.abspath(p) != {os.path.dirname(HERE)!r}]\n"
        f"sys.path.insert(0, {HERE!r})\n"
        "bad = []\n"
        f"for m in {mods!r}:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        "print('\\n'.join(bad))\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=HERE, env=env, timeout=600)
    out = (r.stdout or "").strip()
    assert r.returncode == 0, f"import probe crashed:\n{r.stderr[-2000:]}"
    assert not out, ("modules that do NOT import from a bare clone:\n  " +
                     "\n  ".join(out.splitlines()) +
                     "\n  A clone has no sibling projects — vendor what you need.")
    print(OK + f"all {len(mods)} runtime modules import with only this directory on sys.path")


def test_no_sibling_tree_path_injection():
    """No module may sys.path-inject a directory outside this repo."""
    bad = []
    for m in _repo_modules():
        src = open(os.path.join(HERE, m + ".py")).read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", None)
            if name not in ("insert", "append"):
                continue
            val = getattr(getattr(fn, "value", None), "attr", None)
            owner = getattr(getattr(getattr(fn, "value", None), "value", None), "id", None)
            if val == "path" and owner == "sys":
                # A path injection is allowed only if it is derived from __file__ AND stays
                # inside this directory; the safest rule is simply to forbid it and vendor.
                seg = ast.get_source_segment(src, node) or "sys.path.insert(...)"
                if "dirname(dirname" in seg or ".." in seg:
                    bad.append(f"{m}.py:{node.lineno}: {seg[:90]}")
    assert not bad, ("sys.path injection reaching outside the repo:\n  " + "\n  ".join(bad))
    print(OK + "no module sys.path-injects a parent or sibling directory")


def test_absent_arms_are_lazy_only():
    """Every import of a non-vendored arm module must be function-level, never module-level.

    This is what lets the repo carry only the A/B/C closure while keeping eval_tofu.py
    byte-identical to the working tree (its metrics are frozen and test_ou_equivalence.py is the
    guarantee). A module-level import of any of these would make `import eval_tofu` — and
    therefore every gate and every driver — fail on a clone.
    """
    offenders = []
    for m in _repo_modules():
        tree = ast.parse(open(os.path.join(HERE, m + ".py")).read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in ABSENT_ARMS and node.col_offset == 0:
                    offenders.append(f"{m}.py:{node.lineno}: module-level `import {n}`")
    assert not offenders, (
        "a non-vendored arm is imported at module level, which breaks a clone:\n  " +
        "\n  ".join(offenders) + "\n  Move it inside the branch that uses it, or vendor it.")
    print(OK + f"all {len(ABSENT_ARMS)} non-vendored arms are imported lazily only")


def test_requirements_covers_third_party_imports():
    """Every third-party top-level import is either in requirements.txt or a stdlib module."""
    req_p = os.path.join(HERE, "requirements.txt")
    if not os.path.exists(req_p):
        raise AssertionError("requirements.txt is missing — a clone cannot build an env")
    declared = set()
    for line in open(req_p):
        line = line.split("#")[0].strip()
        if not line:
            continue
        declared.add(line.split("==")[0].split(">=")[0].split("[")[0].strip().lower())
    # import name -> distribution name, where they differ
    ALIAS = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow",
             "sentence_transformers": "sentence-transformers", "rouge_score": "rouge_score",
             "huggingface_hub": "huggingface_hub", "dateutil": "python-dateutil"}
    local = set(_repo_modules()) | ABSENT_ARMS
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"setuptools", "pkg_resources"}
    missing = {}
    for m in _repo_modules():
        tree = ast.parse(open(os.path.join(HERE, m + ".py")).read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in local or n in stdlib or n.startswith("_"):
                    continue
                dist = ALIAS.get(n, n).lower()
                if dist not in declared and n.lower() not in declared:
                    missing.setdefault(n, set()).add(m)
    # matplotlib is intentionally absent (see requirements-plots.txt); anything else is a bug.
    missing.pop("matplotlib", None)
    assert not missing, ("third-party imports absent from requirements.txt:\n  " +
                         "\n  ".join(f"{k} (used by {', '.join(sorted(v))})"
                                     for k, v in sorted(missing.items())))
    print(OK + f"every third-party import is declared ({len(declared)} pinned distributions)")


def test_plot_scripts_fail_with_an_actionable_message():
    """Without matplotlib the plot scripts must say WHICH interpreter to use, not traceback."""
    src = open(os.path.join(HERE, "plot_style.py")).read()
    assert "TOFU_PLOT_PYTHON" in src and "ImportError" in src, \
        "plot_style.py must catch the matplotlib ImportError and name ${TOFU_PLOT_PYTHON}"
    print(OK + "plot scripts explain the interpreter split instead of tracebacking")


def main():
    tests = [test_no_absolute_home_paths, test_no_sibling_tree_path_injection,
             test_absent_arms_are_lazy_only, test_imports_without_sibling_trees,
             test_requirements_covers_third_party_imports,
             test_plot_scripts_fail_with_an_actionable_message]
    for t in tests:
        t()
    print(f"\nALL test_repo_selfcontained.py GATES PASS ({len(tests)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
