"""CPU gate for eval_routed_scaffold.OODAwareRoutedModel --merged_label arm (no GPU, no HF hub).

Contract under test (scaffold x composition 2x2, arm B):
  - merged_adapter set  -> EVERY TOFU-author query serves the ONE merged adapter
                           (incl. forget-shard authors: Fig-8/H8 maskless-merged serving),
                           OOD queries serve scaffold-only (adapters disabled).
  - merged_adapter None -> legacy behavior byte-unchanged (author -> shard_{sid},
                           delete_shard -> disabled path + stats["deleted"]).
  - merged_adapter + delete_shard -> ValueError (deletion = a remerge_* label).
Run before any SLURM job: python test_routed_scaffold_merged.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from eval_routed_scaffold import OODAwareRoutedModel
from legonet_tofu import _norm

SEQ_LEN, VOCAB = 4, 7


class StubModel(nn.Module):
    """Records set_adapter/disable_adapter calls; forward returns fixed-shape outputs."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.active = "shard_0"

    def set_adapter(self, name):
        self.calls.append(("set", name))
        self.active = name

    @contextmanager
    def disable_adapter(self):
        self.calls.append(("disable", None))
        prev, self.active = self.active, None
        try:
            yield
        finally:
            self.active = prev

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        loss = torch.tensor(1.0) if labels is not None else None
        return CausalLMOutputWithPast(loss=loss, logits=torch.zeros(1, SEQ_LEN, VOCAB))

    def generate(self, input_ids, **kw):
        # echo which adapter served this call via the active name recorded in calls
        return input_ids


class StubTokenizer:
    """decode(ids) -> texts[int(ids[0])]; each row's first token id indexes its prompt."""

    def __init__(self, texts):
        self.texts = texts

    def decode(self, ids, skip_special_tokens=True):
        return self.texts[int(ids.reshape(-1)[0])]


def make_fixture(merged_adapter=None, delete_shard=None, reroute_to=None, k=10,
                 ood_route=None):
    q_retain = "Who is Author A?"      # author 45 -> shard 2
    q_forget = "Who is Author Z?"      # author 185 -> shard 9 (forget shard)
    q_ood = "What is the capital of France?"  # not in q2author
    texts = [f"Question: {q_retain}\nAnswer: x",
             f"Question: {q_forget}\nAnswer: y",
             f"Question: {q_ood}\nAnswer: z"]
    q2author = {_norm(q_retain): 45, _norm(q_forget): 185}
    model = StubModel()
    tok = StubTokenizer(texts)
    routed = OODAwareRoutedModel(model, tok, q2author, k=k,
                                 delete_shard=delete_shard, merged_adapter=merged_adapter,
                                 reroute_to=reroute_to, ood_route=ood_route)
    ids = lambda i: torch.tensor([[i] + [0] * (SEQ_LEN - 1)])
    return routed, model, ids


def test_legacy_routing_unchanged():
    routed, model, ids = make_fixture()
    routed(ids(0), labels=ids(0))                       # retain author -> shard_2
    assert model.calls[-1] == ("set", "shard_2"), model.calls
    routed(ids(2), labels=ids(2))                       # OOD -> disabled
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats == {"routed": 1, "ood": 1, "deleted": 0, "rerouted": 0, "ood_routed": 0}, routed.stats


def test_legacy_delete_shard_unchanged():
    routed, model, ids = make_fixture(delete_shard=9)
    routed(ids(1), labels=ids(1))                       # forget author, shard dropped -> disabled
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats["deleted"] == 1, routed.stats
    routed(ids(0), labels=ids(0))                       # retain author still -> shard_2
    assert model.calls[-1] == ("set", "shard_2"), model.calls
    assert routed.delete_shards == frozenset({9})       # int still accepted


def test_multi_unit_delete_set():
    """A per-author pool deletes 20 units to express forget10; `delete_shard` took one int."""
    routed, model, ids = make_fixture(delete_shard=[8, 9], k=10)
    assert routed.delete_shards == frozenset({8, 9})
    routed(ids(1), labels=ids(1))                       # author 185 -> shard 9, deleted
    assert model.calls[-1] == ("disable", None), model.calls
    routed(ids(0), labels=ids(0))                       # author 45 -> shard 2, survives
    assert model.calls[-1] == ("set", "shard_2"), model.calls
    assert routed.stats["deleted"] == 1 and routed.stats["routed"] == 1


def test_reroute_serves_a_survivor_and_drops_nothing():
    """E5: the deleted author's queries are answered by a fixed SURVIVING expert. No expert is
    dropped, so the deletion path must never disable adapters and never count `deleted`."""
    routed, model, ids = make_fixture(delete_shard=9, reroute_to=0)
    routed(ids(1), labels=ids(1))                       # forget author -> shard_0, not base
    assert model.calls[-1] == ("set", "shard_0"), model.calls
    routed(ids(0), labels=ids(0))                       # retain author unaffected
    assert model.calls[-1] == ("set", "shard_2"), model.calls
    routed(ids(2), labels=ids(2))                       # OOD still scaffold-only
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats == {"routed": 1, "ood": 1, "deleted": 0, "rerouted": 1, "ood_routed": 0}, routed.stats
    # generate() takes the same path — an eval that only checked forward would miss a
    # divergence here, and forget_rouge is computed from generations
    routed.generate(ids(1))
    assert model.calls[-1] == ("set", "shard_0"), model.calls


def test_reroute_rejects_incoherent_targets():
    for kw, why in (
        (dict(reroute_to=0), "no delete set"),
        (dict(delete_shard=9, reroute_to=9), "target is itself deleted"),
        (dict(delete_shard=9, reroute_to=10), "target out of range"),
        (dict(delete_shard=[8, 9], reroute_to=8), "target inside the deleted set"),
    ):
        try:
            make_fixture(**kw)
        except ValueError:
            continue
        raise AssertionError(f"reroute config accepted but should raise: {why} ({kw})")


def test_path_authors_survive_repeated_forwards():
    """The route audit's invariant is over AUTHORS, not forward passes.

    One question is forwarded several times per eval — ppl, generation, and the truth-ratio
    paraphrased/perturbed variants — so a counter compared against a question count is wrong by
    however many passes the metric suite happens to make. That mistake cost a completed k=200
    arm: the assert fired on `deleted == 630 != 400` and, sitting before the json.dump, threw
    away metrics that had taken over an hour to compute.
    """
    routed, model, ids = make_fixture(delete_shard=9)
    for _ in range(3):                                  # same question, three forwards
        routed(ids(1), labels=ids(1))
    routed(ids(0), labels=ids(0))
    assert routed.stats["deleted"] == 3, routed.stats   # counter follows passes ...
    assert routed.path_authors["deleted"] == {185}      # ... the author set does not
    assert routed.path_authors["routed"] == {45}
    assert not (routed.path_authors["routed"] & routed.path_authors["deleted"])

    rr, _, rids = make_fixture(delete_shard=9, reroute_to=0)
    for _ in range(2):
        rr(rids(1), labels=rids(1))
    assert rr.stats["rerouted"] == 2 and rr.path_authors["rerouted"] == {185}
    assert rr.path_authors["deleted"] == set()
    print("  [ok] path_authors is per-author and stable under repeated forwards")


def test_ungated_ood_routes_instead_of_abstaining():
    """The q2author lookup deciding "is this about one of my sources" is an ORACLE.

    Default (`ood_route=None`) is unchanged: a miss serves base+scaffold, which is what every
    published number assumes. With a fallback router the miss is handed to a surviving expert
    instead — what happens when nobody tells the system the query is out of domain. Source
    routing is untouched either way, so the delta prices this oracle alone.
    """
    routed, model, ids = make_fixture(ood_route=lambda q: 5)
    routed(ids(2), labels=ids(2))                       # OOD -> shard_5, NOT disabled
    assert model.calls[-1] == ("set", "shard_5"), model.calls
    routed(ids(0), labels=ids(0))                       # in-domain routing unchanged
    assert model.calls[-1] == ("set", "shard_2"), model.calls
    assert routed.stats["ood"] == 1 and routed.stats["ood_routed"] == 1, routed.stats
    # generate() takes the same path
    routed.generate(ids(2))
    assert model.calls[-1] == ("set", "shard_5"), model.calls

    # default still abstains
    gated, gmodel, gids = make_fixture()
    gated(gids(2), labels=gids(2))
    assert gmodel.calls[-1] == ("disable", None), gmodel.calls
    assert gated.stats["ood_routed"] == 0


def test_merged_serves_all_authors():
    routed, model, ids = make_fixture(merged_adapter="remerge_additive_mean")
    routed(ids(0), labels=ids(0))                       # retain author -> merged adapter
    assert model.calls[-1] == ("set", "remerge_additive_mean"), model.calls
    routed(ids(1), labels=ids(1))                       # FORGET author -> merged adapter too (Fig-8)
    assert model.calls[-1] == ("set", "remerge_additive_mean"), model.calls
    routed(ids(2), labels=ids(2))                       # OOD -> scaffold-only, never the merge
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats == {"routed": 2, "ood": 1, "deleted": 0, "rerouted": 0, "ood_routed": 0}, routed.stats


def test_merged_generate_and_batch():
    routed, model, ids = make_fixture(merged_adapter="merged_additive_mean")
    routed.generate(ids(0))
    assert model.calls[-1] == ("set", "merged_additive_mean"), model.calls
    routed.generate(ids(2))
    assert model.calls[-1] == ("disable", None), model.calls
    batch = torch.cat([ids(0), ids(2)], dim=0)          # mixed author+OOD batch
    out = routed(batch, attention_mask=torch.ones_like(batch), labels=batch)
    assert out.logits.shape == (2, SEQ_LEN, VOCAB), out.logits.shape
    assert out.loss is not None and torch.isfinite(out.loss)
    set_calls = [c for c in model.calls if c[0] == "set"]
    assert all(name == "merged_additive_mean" for _, name in set_calls), set_calls


def test_merged_plus_delete_raises():
    try:
        make_fixture(merged_adapter="merged_additive_mean", delete_shard=9)
    except ValueError:
        return
    raise AssertionError("merged_adapter + delete_shard must raise ValueError")


if __name__ == "__main__":
    test_legacy_routing_unchanged()
    test_legacy_delete_shard_unchanged()
    test_multi_unit_delete_set()
    test_reroute_serves_a_survivor_and_drops_nothing()
    test_reroute_rejects_incoherent_targets()
    test_path_authors_survive_repeated_forwards()
    test_ungated_ood_routes_instead_of_abstaining()
    test_merged_serves_all_authors()
    test_merged_generate_and_batch()
    test_merged_plus_delete_raises()
    print("test_routed_scaffold_merged: ALL GREEN (10/10)")
