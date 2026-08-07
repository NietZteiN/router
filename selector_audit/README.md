# selector_audit — the metric suite for deletion under a selector

The reusable half of the MUSR follow-up. `tofu_sisa_lora/` holds the experiments; this directory
holds the metrics they are read through, because those are what another paper would want to
re-use and what an OpenUnlearning contribution would ship.

Research narrative: [`../log/selector_audit/`](../log/selector_audit/). Decision rules are
pre-registered there before each run, not chosen after the numbers land.

## What is here

| file | what it measures |
|---|---|
| [`csar.py`](csar.py) | **CSAR** — Cross-Source Attribution Rate. Classifies each orphan generation as refusal / cross-source attribution / base-generic / unattributable, by matching *facts* against the routed survivor's gold rather than n-gram overlap. |
| [`test_csar.py`](test_csar.py) | CPU gate, including the two real-TOFU controls that make the metric trustworthy. |

## Why CSAR is not ROUGE-L

The existing sibling-content audit scores three ROUGE-L axes and finds a **95.5%** confabulation
rate at k=10 — orphan answers that match neither the deleted author's gold nor the survivor's.
That is evidence about n-gram overlap, not about whose facts were served. An answer that says
*"Alice writes magical realism and was born in Lisbon"* — Bob's genre, Bob's city, asserted about
Alice — scores low ROUGE-L against every gold answer in the corpus, and is exactly the harm the
paper is about.

CSAR counts a generation as cross-source attribution when it contains at least one fact that is

1. **distinctive to the routed survivor** — appears in at most `max_adf` (default 2) of the 200
   authors' gold sets, so "literary fiction" never counts and "Addis Ababa" does;
2. **not the deleted author's own** — serving the deleted author's real facts is a disclosure of
   the deleted source, a different and worse failure, and must not be folded into this number;
3. **not present in the base model's answer** to the same question — otherwise the frozen base's
   own knowledge would be credited to the selector.

Distinctiveness is measured on the corpus, never hand-listed. Three filters cooperate: tokens
that also occur lowercase across authors are ordinary words; tokens capitalized *only* at
sentence starts are punctuation artifacts (this is what removes "Moreover", which never appears
lowercase in TOFU and which a frequency filter alone cannot catch); and the author-document
frequency removes shared vocabulary.

## Validation

`test_csar.py` runs two controls over the real 200-author corpus and refuses to pass without them:

| control | served text | expected | measured |
|---|---|---|---|
| negative | the **deleted** author's own gold answer | ~0.00 | **0.000** |
| positive | the **survivor's** own gold answer, as a reply about the deleted author | ~1.00 | **0.970** |

plus generic prose ("Moreover, this author is known for their work.") matching **no** author.

A metric this one-sided on controls is still a proxy. Before any CSAR is quoted in the paper,
`--sample_for_labeling 300` emits records for hand labelling and `--labels` reports agreement and
per-category precision/recall against them.

## Usage

```bash
python csar.py --self_test
python test_csar.py                       # add HF_HOME to include the real-TOFU controls

# score a per-strategy sibling-content audit
python csar.py --audit_json <dump_generations_routed --strategies output> \
    --out_json csar_k200.json --out_md csar_k200.md

# hand-label 300 records, then report agreement
python csar.py --audit_json A.json --sample_for_labeling 300 --out_jsonl label_me.jsonl
python csar.py --audit_json A.json --labels label_me.jsonl --out_json validation.json
```

## Granularity caveat

CSAR asks *whose* facts were served. Below per-author routing units the survivor is a group of
authors and the question has no single answer; at k=10 a shard hosts 20 people. `csar.py` warns
and scores against the shard's pooled gold, which reads as an upper bound. The metric is at home
on the k=200 per-author pool.

## Not yet here

The rest of the suite from the paper plan — ORR, Concentration, RDR, DD-AUC, RIP, FMD, PDG, SEA —
still lives in `tofu_sisa_lora/` next to the producers that feed it
(`analyze_orphan_destinations.py`, `analyze_router_family.py`, `analyze_router_probe.py`).
Consolidating them here is the step that turns this directory into the released harness; it is
deliberately not done before the pilots decide which metrics the paper actually needs.
