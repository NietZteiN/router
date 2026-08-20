# Deleted from the Router, Not from the Model — short summary

*A condensation of the whole of [`SELECTOR_AUDIT_REPORT.md`](SELECTOR_AUDIT_REPORT.md) (campaign
2026-08-07 → 2026-08-12, plus the 2026-08-18 baseline control). Every claim below is stated in full,
with its table and caveats, in the section it links to.*

**The setting.** Train one small expert per person, let a router pick which expert answers, and
delete a person by throwing their expert away. The deletion is provably real — nothing left was
trained on their data. But their questions keep arriving, and the router still sends them
*somewhere*. **A system can honour a deletion request, score well on the field's standard forgetting
metric, and still hand the next user a stranger's biography under the deleted person's name.**

Measured on `Llama-2-7B-chat-hf_k200_r32_e25_lr1e4` — 200 per-author experts, authors 180–199
deleted (400 orphaned questions) — across 8 routers and 5 question phrasings, with variants at 5/50
epochs and rank 8 for controls.

## The seven findings

1. **The benchmark cannot see substitution** ([§11](SELECTOR_AUDIT_REPORT.md#11-finding-1-the-benchmark-cannot-tell-deletion-from-substitution)).
   A deliberately fake method that deletes nothing and only redirects orphaned questions matched or
   beat genuine deletion on `forget_quality` in **6 of 7** destinations, utility unchanged to four
   decimals. Destination choice alone moved the score **0.53** on a 0–1 scale.

2. **The harm is real, and not a naming glitch** ([§12](SELECTOR_AUDIT_REPORT.md#12-finding-2-what-the-system-actually-says-to-an-orphan)).
   Answers about a deleted author assert a specific *surviving* author's facts at **0.24–0.30**; a
   random substitute still gives **0.17**, so most of it comes from substituting anybody and no
   better router removes it. Refusal never exceeds **1.3%** of 1600 answers — including from a
   router that knows it has no match for 100% of them.

3. **Everything reassuring is a property of the name** ([§13](SELECTOR_AUDIT_REPORT.md#13-finding-3-where-orphans-go-and-what-deletion-disturbs), [§14](SELECTOR_AUDIT_REPORT.md#14-finding-4-orphans-are-only-detectable-because-of-the-name)).
   Verbatim questions look clean. Strip the author's name and orphan detection falls **0.991 →
   0.623**, routing accuracy **0.966 → 0.343**, and deletion silently displaces **9.2%** of other
   authors' questions. True of every router but one. The "magnet" prediction — orphans concentrating
   on one survivor — was refuted; they spread over 23–29 destinations.

4. **A lexical router can be steered** ([§15](SELECTOR_AUDIT_REPORT.md#15-finding-5-a-router-that-reads-names-can-be-steered-by-an-attacker)).
   Appending one chosen name captures `key_exact` **97.7%** of the time (structural: it returns the
   lowest-index matching shard). Composed with Finding 2, the attacker chooses *whose* facts get
   attributed to the erased person.

5. **The one surviving defense is not deployable** ([§16](SELECTOR_AUDIT_REPORT.md#16-finding-6-the-one-defense-that-survives-is-not-deployable)).
   The `ppl` gate — ask each expert which is least surprised — survives anonymisation and we made it
   **45–90× cheaper** for free. At 90% orphan recall on anonymous queries it refuses **41.8%** of
   legitimate traffic, and is flawless only where the name already told you the answer.

6. **Training longer moves the leak** ([§17](SELECTOR_AUDIT_REPORT.md#17-finding-7-training-longer-moves-the-leak-instead-of-removing-it)).
   5 → 25 → 50 epochs drives the cheap magnitude routers to chance (`activation_norm` 0.934 → 0.608
   → 0.515) while `ppl` holds (1.000 → 0.999 → 0.996). That improves the audit result, not the
   privacy — it must not be offered as mitigation.

7. **Much of it is TOFU, not routing** ([§18](SELECTOR_AUDIT_REPORT.md#18-the-baseline-control-is-this-failure-routing-or-is-it-tofu)).
   A routerless fine-tune loses the *same* quality to name removal (−0.4962 vs our −0.4850) and
   already follows an injected name (0.0550 append / 0.2050 substitute vs our 0.2288 / 0.4487).
   Finding 3's answering half and Finding 4 are an amplification over a model-level floor, 2–4×.
   Findings 1 and 2 have no routerless analogue and are untouched. The control also bought a
   deletion-size ladder: a deleted user's own degradation is flat in deletion volume, but
   **collateral damage to everyone else grows** (RDR 0 → 0.0925 anonymised, 1 → 20 deletions), so
   locality decays as requests accumulate.

## Trust

Findings 1–7 are settled. One item is **blocked**: ~300 human labels validating the CSAR
classifier. Known defect — `name_stripped` leaves a name in **31.2%** of rows, so every name-free
number is an upper bound. Nine of our own errors are recorded in
[§20](SELECTOR_AUDIT_REPORT.md#20-defect-record-nine-things-we-got-wrong); 28 of 31 hypotheses were
adjudicated.

**Not claimed:** that one substitute destination beats another (the spread reproduces, the ordering
does not) · that CSAR is human-validated · that longer training mitigates anything · that no
record-free defense exists (we bounded one) · that these rates transfer off TOFU's uniform synthetic
English corpus.
