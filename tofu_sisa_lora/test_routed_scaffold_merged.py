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


def make_fixture(merged_adapter=None, delete_shard=None):
    q_retain = "Who is Author A?"      # author 45 -> shard 2
    q_forget = "Who is Author Z?"      # author 185 -> shard 9 (forget shard)
    q_ood = "What is the capital of France?"  # not in q2author
    texts = [f"Question: {q_retain}\nAnswer: x",
             f"Question: {q_forget}\nAnswer: y",
             f"Question: {q_ood}\nAnswer: z"]
    q2author = {_norm(q_retain): 45, _norm(q_forget): 185}
    model = StubModel()
    tok = StubTokenizer(texts)
    routed = OODAwareRoutedModel(model, tok, q2author, k=10,
                                 delete_shard=delete_shard, merged_adapter=merged_adapter)
    ids = lambda i: torch.tensor([[i] + [0] * (SEQ_LEN - 1)])
    return routed, model, ids


def test_legacy_routing_unchanged():
    routed, model, ids = make_fixture()
    routed(ids(0), labels=ids(0))                       # retain author -> shard_2
    assert model.calls[-1] == ("set", "shard_2"), model.calls
    routed(ids(2), labels=ids(2))                       # OOD -> disabled
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats == {"routed": 1, "ood": 1, "deleted": 0}, routed.stats


def test_legacy_delete_shard_unchanged():
    routed, model, ids = make_fixture(delete_shard=9)
    routed(ids(1), labels=ids(1))                       # forget author, shard dropped -> disabled
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats["deleted"] == 1, routed.stats
    routed(ids(0), labels=ids(0))                       # retain author still -> shard_2
    assert model.calls[-1] == ("set", "shard_2"), model.calls


def test_merged_serves_all_authors():
    routed, model, ids = make_fixture(merged_adapter="remerge_additive_mean")
    routed(ids(0), labels=ids(0))                       # retain author -> merged adapter
    assert model.calls[-1] == ("set", "remerge_additive_mean"), model.calls
    routed(ids(1), labels=ids(1))                       # FORGET author -> merged adapter too (Fig-8)
    assert model.calls[-1] == ("set", "remerge_additive_mean"), model.calls
    routed(ids(2), labels=ids(2))                       # OOD -> scaffold-only, never the merge
    assert model.calls[-1] == ("disable", None), model.calls
    assert routed.stats == {"routed": 2, "ood": 1, "deleted": 0}, routed.stats


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
    test_merged_serves_all_authors()
    test_merged_generate_and_batch()
    test_merged_plus_delete_raises()
    print("test_routed_scaffold_merged: ALL GREEN (5/5)")
