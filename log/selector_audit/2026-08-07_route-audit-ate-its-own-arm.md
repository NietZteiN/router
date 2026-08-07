### Target Date: 2026-08-07 (DEFECT: the route audit destroyed the arm it was auditing)

Eighth entry today. A defect in the E5 tooling from
[e1-router-probe-and-preregistration](2026-08-07_e1-router-probe-and-preregistration.md), whose
pre-registration required a route audit "before any metric is read". The audit was right to exist
and wrong in both its invariant and its placement.

- **Hypotheses / what we're testing:** none. Defect record.

- **Setup:** job **3191693** (`e5-reroute`), the `delete` arm, k=200 e25 pool, forget10 deleted.

- **Results:**

  ```
  route audit FAILED: 630 orphan queries took the deletion path, expected 400 (20 authors x 20).
  stats={'routed': 360, 'ood': 1208, 'deleted': 630, 'rerouted': 0}
  ```

  **The routing was correct.** `stats` counts **forward passes**, and one question is forwarded
  several times per eval — for perplexity, for the greedy generation, and for the truth-ratio
  paraphrased/perturbed variants. Comparing a pass counter against a question count is off by
  however many passes the metric suite happens to make; 630 vs 400 is that, not a misroute.

  The second error was worse. The raise sat **before** `json.dump`, so an arm that had run for
  1h15m computed every metric and then threw them away. The remaining three arms of that job, and
  both `ood_gate` arms, were guaranteed to reach the same assertion, so they were cancelled rather
  than left to burn to a certain failure.

- **What worked / hypothesis verdict:**
  - **Invariant fixed.** `path_authors` records the distinct AUTHORS that took each path. The
    claim an arm actually makes is: every deleted author took the deletion path, no retained
    author did, and no deleted author was served normally. That is stable under any number of
    forwards. The audit result is now a dict — `missing`, `unexpected`,
    `deleted_authors_served_normally` — rather than a single count comparison.
  - **Placement fixed.** The artifact is written first with the verdict inside it as
    `route_audit`, and the non-zero exit comes after. **An audit that destroys the artifact it
    audits is worse than no audit**: the failure is exactly the moment you most want the numbers
    to look at.
  - The gate reproduces the original mistake directly: three forwards of one deleted author's
    question leave `stats["deleted"] == 3` and `path_authors["deleted"] == {185}`.

- **Observations:**
  - The check was added for a real reason — a plausible-but-wrong route is invisible to every
    metric — and I wrote it in the same breath as the pre-registration that demanded it. Writing
    a check and validating the check are different acts, and I did only the first. It had never
    run against a real eval before today; the stub gate it did have used one forward per query,
    which is precisely the case that hides the bug.
  - Cost: roughly 1h15m of A100 time on the killed arm, plus the cancelled remainder. Cheap
    relative to trusting a wrong number, which is what the audit exists to prevent — but entirely
    self-inflicted, and avoidable by writing before raising.

- **New questions / new hypotheses:** none.

- **Next Steps:** both jobs resubmitted (**3192096** e5-reroute, **3192097** ood-gate) with the
  corrected audit. Check `route_audit.ok` in each output rather than the exit status alone.
