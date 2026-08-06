# ARMS — what is deliberately not here

`eval_tofu.py` is the shared evaluation harness of a much larger research tree. It can serve a
dozen different architectures, selected by mutually-exclusive CLI flags. This repo carries only
the arms Experiments A, B and C use:

| supported here | how |
|---|---|
| a materialized merge | `--preloaded_adapter DIR` (every A/B/C condition) |
| the plain base model | no arm flag, or `eval_baseline.py` |
| one isolated adapter | `--preloaded_adapter .../shard_<a>` (the Exp-B `iso_a{X}` ceiling) |

Everything else was left out. The modules are absent, so passing their flag raises
`ModuleNotFoundError`:

| flag | arm | its home track |
|---|---|---|
| `--legonet_config` | LegoNet: frozen k-means keys + top-k routing | legonet_lora |
| `--ramole_router` | RAMoLE: retrieval over LoRAs + a learned RouterLoRA | ramole |
| `--sift_masks_config` | SIFT-Masks: sign-constrained full-FT + bit masks | sift_masks |
| `--clamu_config` | ClAMU: per-cluster STE-optimized masks | clamu |
| `--linear_tv_config` | linearized (tangent-space) task vectors | composable_tv |
| `--ds_config` | disjoint-support sparse full-FT | composable_tv |
| `--prefix_pool_dir` | KV-prefix concatenation | peft_compose |
| `--memsinks_config` | MemSinks routed masks | memsinks |
| `routed_*` labels | the nine `router.py` strategies | router_leak |
| `ensemble_*` labels | prediction-level ensembling | sisa_lora |

## Why they can be absent

Because every one of those imports is **lazy** — inside the branch that needs it, not at module
level. So `import eval_tofu` never touches them, and every gate, driver and analysis in this
repo works on a clone with none of those trees present.

That is a contract, not an accident, and it is enforced:

```bash
python test_repo_selfcontained.py    # test_absent_arms_are_lazy_only
```

A module-level import of any absent arm would make `import eval_tofu` fail — and therefore
every gate and every driver — so the gate fails loudly at edit time rather than at clone time.

## Why not just delete the branches

Keeping them costs nothing and buys two things:

1. **`eval_tofu.py` stays byte-identical to the working tree.** Its metrics are frozen;
   `test_ou_equivalence.py` proves they reproduce open-unlearning's formulas numerically. Every
   number this project has ever recorded is comparable only while that stays true, and the
   cheapest way to keep it true is not to touch the file.
2. **Drift stays diffable.** `sync_from_tree.sh --check` can report a one-line difference
   instead of a rewrite, so a fix made upstream can be pulled in mechanically.

If you need one of those arms, vendor its module (and its own closure) and add it to
`MANIFEST.files` — do not add a `sys.path` injection pointing at a sibling directory. That is
the exact failure this repo was built to remove: it resolves silently to whatever happens to sit
one level up, so it works on the machine where it was written and nowhere else.
