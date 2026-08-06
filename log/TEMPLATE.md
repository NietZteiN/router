# Log templates — copy from here

Blank, copy-paste scaffolds for the experiment ledger. Canonical spec: [CLAUDE.md](../CLAUDE.md) §6.
**Append-only:** never edit a historical entry — corrections go in a *new* dated entry that
references the old one. Ground every claim in real data; never invent numbers.

---

## 1. Daily entry — copy into `log/<experiment>/YYYY-MM-DD_<slug>.md`

One file per working day per thread (`<slug>` = 2–4 word kebab-case headline). The first line
is the only header. Fill all seven fields; the loop is: state hypotheses **before** the run,
make the setup reproducible, record raw results, then close the loop with the verdict,
observations, and the new questions/hypotheses the result raises.

```markdown
### Target Date: YYYY-MM-DD (<short title>)
- **Hypotheses / what we're testing:** [Explicit, falsifiable statement(s). Number them H1, H2… when there are several. For each, state the prediction: what result would CONFIRM it and what would REFUTE it. If today is exploratory (no firm hypothesis yet), say so and state the question.]
- **Setup:** [Explicitly reproducible — exact commands, config paths, seed, git commit hash / script sha256, SLURM Job IDs, model + dataset, hardware. Enough to re-run without guessing.]
- **Results:** [Key quantitative metrics, loss curves, or generation outputs — the numbers only, no interpretation.]
- **What worked / hypothesis verdict:** [Map each result back to its hypothesis: SUPPORTED / REFUTED / INCONCLUSIVE, with the evidence (quote the metric). State plainly what worked and what didn't.]
- **Observations:** [What do these results mean? Surprises, bottlenecks, or errors hit; silent-failure checks (NaNs, frozen loss, empty/repetitive generations, out-of-bounds metrics).]
- **New questions / new hypotheses:** [Additional questions the result raises; new hypotheses to test next. These feed the thread README's "Hypotheses — open" ledger.]
- **Next Steps:** [Concrete tactical adjustments or experiments for the next session]
```

Links inside an entry point to repo files two levels up — use `../../<path>`.

---

## 2. Thread README — copy into `log/<experiment>/README.md` when starting a new thread

Living summary of one research thread (updated in place, not append-only). Refresh **Status**,
the **Hypotheses — open / resolved** ledger, and the **What worked / What didn't / Open ideas**
bullets on every new entry; add each new entry to the chronological list.

```markdown
# <experiment> — <one-line what-it-is>

**Status:** <active | complete | reference> · **Project:** [`<project_dir>/`](../../<project_dir>/) · **Entries:** <N> (YYYY-MM-DD → YYYY-MM-DD)

<1–3 paragraph description of the thread: the method, the unlearning tie-in, and the arc so far.>

## Hypotheses — open / resolved
- **[resolved ✓ supported]** H: <claim> — confirmed by <entry + metric>.
- **[resolved ✗ refuted]** H: <claim> — refuted by <entry + metric>.
- **[open]** H: <claim> — pending <what test would settle it>.

## What worked
- <finding, with the metric and the entry it came from>

## What didn't / open problems
- <failure or constraint, with evidence>

## Open ideas / next steps
- <future direction or decision gate>

## Entries (chronological)
- [YYYY-MM-DD — <title>](YYYY-MM-DD_<slug>.md) — <headline>
```
