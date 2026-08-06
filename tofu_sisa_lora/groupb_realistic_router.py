"""Attach a REALISTIC (MiniLM embedding) selector to a Group-B served model, replacing its
oracle author-ID lookup — for the router-leak (b) test.

Group B (SIFT-Masks / ClAMU / MemSinks) serve a single merged/masked model and pick the
per-task mask by an ORACLE exact author-ID lookup (`legonet_tofu.build_q2author`). That is
clean by construction. The question: if you must select WITHOUT author labels — a realistic
embedding router over the selection-unit centroids — do these methods inherit the Group-A
orphan leak, and does misrouting actually leak deleted content?

`attach_realistic_router` swaps the model's `_route` for an embedding router over the
SURVIVING units' question centroids, keeping `q2author` only for the OOD gate (a TOFU-author
query embedding-routes; a real/world query still serves base) — exactly EmbedRoutedModel's
oracle-gated OOD split, so the OOD path is not a confound. Deleted authors' queries route
among survivors → a surviving sibling unit, the leak under test.

Works for all three by monkey-patching the bound `_route` (their `_apply`/`_mask_for` are
keyed on the returned author id and unchanged):
  - SIFT / MemSinks: units are AUTHORS; the router returns the argmax author directly.
  - ClAMU: units are feature CLUSTERS; the router returns a representative member author of
    the argmax cluster, which the model's `_apply` maps back to that cluster's mask.
"""
from __future__ import annotations

import types

import numpy as np

import legonet_tofu as lt
from eval_routed_scaffold import build_unit_centroids


def attach_realistic_router(model, hf_home, unit_to_authors, device="cuda",
                            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                            embed=None, unit_repr_author=None):
    """Replace `model._route` with a MiniLM embedding router over `unit_to_authors` (the
    SURVIVING units only — deleted units must be excluded by the caller). `unit_repr_author`
    maps unit-id -> the author id `model._apply` should receive (default: the unit id itself,
    correct when units ARE authors). Records `model.route_stats`. Returns the model."""
    cents, uids, embed = build_unit_centroids(hf_home, unit_to_authors, device,
                                              encoder_name=encoder_name, embed=embed)
    repr_author = dict(unit_repr_author) if unit_repr_author else {u: u for u in uids}
    model._rr_centroids = cents
    model._rr_uids = uids
    model._rr_embed = embed
    model._rr_repr = repr_author
    model.route_stats = {"routed": 0, "ood": 0, "orphan_misrouted": 0}
    # authors whose own unit survived (routing to self is possible) vs orphans (unit dropped)
    surviving_authors = {int(a) for authors in unit_to_authors.values() for a in authors}
    model._rr_surviving_authors = surviving_authors

    def _route(self, text):
        q = lt.parse_question(text)
        author = self.q2author.get(lt._norm(q))
        if author is None:
            self.route_stats["ood"] += 1
            return None                                  # OOD -> base (oracle-gated)
        v = self._rr_embed([q])[0]
        uid = self._rr_uids[int(np.argmax(self._rr_centroids @ v))]
        routed_author = self._rr_repr[uid]
        self.route_stats["routed"] += 1
        if author not in self._rr_surviving_authors:     # a deleted author = an orphan
            self.route_stats["orphan_misrouted"] += 1
        return routed_author

    model._route = types.MethodType(_route, model)
    return model


def per_author_units(num_authors, forget_authors):
    """SIFT / MemSinks: one surviving unit per retained author."""
    forget = set(int(a) for a in forget_authors)
    return {a: [a] for a in range(num_authors) if a not in forget}


def clamu_cluster_units(assignment):
    """ClAMU: surviving units = the (retain re-clustered) feature clusters. Returns
    (unit_to_authors, unit_repr_author) — repr = the lowest-id member so `_apply` maps it
    back to the cluster via author_to_cluster."""
    members = {int(c): [int(a) for a in ms] for c, ms in assignment["members"].items() if ms}
    repr_author = {c: min(ms) for c, ms in members.items()}
    return members, repr_author
