# ou_integration — the open-unlearning fork state

The OU-track evals (the `Agg` / `Util` / `Priv` rows of `sepmlp_tofu`, `blocktc_tofu` and
`memadapt_tofu`) run inside [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning),
not in this tree. That fork is not vendored — it is pinned by
[`../fetch_upstream.sh`](../fetch_upstream.sh) at commit `93e9cd5`.

## What is in `patches/`, and why it had to be captured

Two of the fork's commits are on its own branch and come down with the clone:

| commit | what |
|---|---|
| `93e9cd5` | *"Fix offline cache lookup: cache_dir must be HF_HOME/hub, not HF_HOME"* |
| `7cea9fe` | the MemAdapt eval integration (registry shim + model/eval configs) |

The **sepmlp registry was never committed**. On the original cluster it existed only as a dirty
working tree:

```
 M src/model/__init__.py
?? src/model/sepmlp_registry.py
?? configs/model/SepMlp-Llama-3.2-1B.yaml
```

Three files, reachable from no git remote, that the entire `sepmlp` OU track depends on. Without
them `--model SepMlp-Llama-3.2-1B` raises a registry `KeyError`. They are captured here:

```
patches/sepmlp_registry.py           -> open-unlearning/src/model/sepmlp_registry.py
patches/SepMlp-Llama-3.2-1B.yaml     -> open-unlearning/configs/model/SepMlp-Llama-3.2-1B.yaml
patches/model__init__.diff           -> git apply, against src/model/__init__.py
```

`fetch_upstream.sh` installs all three after cloning. If the diff does not apply cleanly it says
so and names the file rather than continuing silently — the likely causes are "already applied" or
"upstream moved that line".

The `__init__.py` change is an fp32-logits fix: `transformers` ≥ 4.49 moved the fp32 cast, and
without the fix the evals either crash or diverge from the reference numbers. It is
deliberately-dirty state in the source tree, which is why it trips clean-tree guards there.

## Per-project registries

Each OU-track project also ships its own `ou_integration/` with an installer that extends the same
branch:

| project | files |
|---|---|
| [`../sepmlp_tofu/ou_integration/`](../sepmlp_tofu/ou_integration/) | `sepmlp_registry.py`, `SepMlp-Llama-3.2-1B.yaml`, `install_branch.sh` |
| [`../blocktc_tofu/ou_integration/`](../blocktc_tofu/ou_integration/) | `blocktc_registry.py`, `BlockTc-Llama-3.2-1B.yaml`, `install_branch.sh` |
| [`../memadapt_tofu/ou_integration/`](../memadapt_tofu/ou_integration/) | `memadapt_registry.py`, `MemAdapt-Llama-3.2-1B.yaml`, `tofu_grimes.yaml`, `install_branch.sh` |

The registries are `sys.path` shims — the model code is never copied into the fork, so there is
one implementation, not two.

```bash
bash ../fetch_upstream.sh
OU_DIR=$PWD/../open-unlearning bash ../sepmlp_tofu/ou_integration/install_branch.sh
```

## The second environment

These evals need `requirements-ou.txt` (torch 2.4.1 / transformers 4.51.3), **not**
`requirements.txt`. The two `transformers` majors resolve the Llama-3.2 chat template differently,
so running an OU eval under the wrong one yields a plausible wrong number rather than an error.
`cluster_env.sh` exposes it as `$TOFU_OU_PYTHON`.

Two traps that are easy to hit and hard to notice, both recorded in the projects' own `CLAUDE.md`:

- the OU experiment file is `@package _global_` and merges **after** the model group, silently
  resetting `pretrained_model_name_or_path` to the TOFU-finetuned model. Every eval command must
  carry the explicit override.
- `tofu_grimes.yaml` has `overwrite: false`, so re-running into a reused `paths.output_dir` serves
  the **stale cached** `TOFU_EVAL.json`. New checkpoint → new eval label/dir, always.
