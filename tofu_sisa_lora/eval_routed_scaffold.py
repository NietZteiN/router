"""OOD-aware routed+scaffold eval (fixes the composition bug).

TOFU-author queries -> their shard expert (exact q2author lookup, not name-substring);
OOD queries (real_authors / world_facts) -> scaffold-only base (adapters disabled). Run against the
*scaffolded* base (base+scaffold baked in) so author queries serve base+scaffold+expert and OOD serve
base+scaffold. This stops a TOFU expert from being slapped onto general-knowledge queries and
corrupting the scaffold's answer.

  python eval_routed_scaffold.py --model_name <scaffolded_base> --shards_dir checkpoints/Llama-3.2-1B-Instruct \
      --k 10 --forget_shard_id 9 --out reports/routed_scaffold_ood.json --smoke
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

import eval_tofu as et
from eval_tofu import (load_tofu_data, load_all_shard_adapters, evaluate_model,
                       SMOKE_ROUGE_MAX, SMOKE_RETAIN_MAX, SMOKE_TRUTH_MAX,
                       EXTENDED_ROUGE_MAX, EXTENDED_RETAIN_MAX, EXTENDED_TRUTH_MAX)
from eval_progress import ProgressLogger
from shard_utils import get_author_shard, parse_author_ids
from legonet_tofu import build_q2author, _norm

import sys

# ── site env bootstrap (added on export) ─────────────────────────────────────────────────────
# This module reads os.environ["TOFU_*"] at import. A script launched by a submit_*.sh inherits
# those from cluster_env.<site>.sh; one run by hand does not, and would die with a bare KeyError
# naming a variable the reader has never heard of. ensure_site_env() sources the site file once
# so both entry points behave the same.
_REPO_ROOT_FOR_ENV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_ENV not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_ENV)
try:
    from repo_env import ensure_site_env as _ensure_site_env
    _ensure_site_env()
except ImportError:
    pass


def _extract_question(text):
    """Recover the raw question from a "Question: {q}\nAnswer: {a}" prompt."""
    if text.startswith("Question:"):
        text = text[len("Question:"):]
    idx = text.find("\nAnswer")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def build_unit_centroids(hf_home, unit_to_authors, device,
                         encoder_name="sentence-transformers/all-MiniLM-L6-v2", embed=None):
    """Training-free embedding router over ARBITRARY author groupings. `unit_to_authors` maps
    each routing-unit id -> list of member author ids; the unit centroid = L2-normed mean of
    its members' TOFU question embeddings. Returns (centroids (U,D), [unit_ids], embed_fn).
    Generalizes build_shard_centroids (contiguous shards) to Group-B selection units:
    per-author (SIFT/MemSinks: {a:[a]}) or feature clusters (ClAMU: assignment['members']).
    Pass `embed=` to reuse an already-loaded encoder fn (same MiniLM/normalize convention as
    the router-leak centroid audit, so results are directly comparable)."""
    data_full = et.load_tofu_data(hf_home)["full"]
    if embed is None:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(encoder_name, device=device)
        embed = lambda ts: np.asarray(enc.encode(ts, normalize_embeddings=True), dtype="float32")
    cents, uids = [], []
    for uid, authors in unit_to_authors.items():
        qs = [data_full[int(a) * 20 + w]["question"] for a in authors for w in range(20)]
        v = embed(qs).mean(0)
        cents.append(v / (np.linalg.norm(v) + 1e-12))
        uids.append(int(uid))
    return np.stack(cents), uids, embed


def build_shard_centroids(hf_home, k, shard_ids, device,
                          encoder_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Training-free embedding router: per shard, centroid = L2-normed mean of its member
    authors' TOFU question embeddings. Returns (centroids (S,D), [shard_ids], embed_fn).
    Shared by the router-leak serving arms here and eval_entangled_probe's served_embedsim
    (pass the SURVIVING shard list there; pass ALL shards for a tombstone policy, which
    keeps the deleted shard's centroid as an identity sentinel). Thin wrapper over
    build_unit_centroids with contiguous shards as the units."""
    from shard_utils import get_author_shard
    return build_unit_centroids(hf_home, {j: list(get_author_shard(k, j)) for j in shard_ids},
                                device, encoder_name=encoder_name)


class OODAwareRoutedModel(nn.Module):
    """Author query -> shard_{author//per_shard}; OOD -> disable adapters (scaffold-only base).

    merged_adapter: scaffold×composition 2x2 control (arm B) — every TOFU-author query is
    served by this ONE merged adapter instead of its shard expert (forgotten authors included,
    i.e. Fig-8/H8 maskless-merged serving under a remerge label); OOD stays scaffold-only.
    Mutually exclusive with delete_shard: merged-mode deletion is expressed by the
    remerge_* label itself, not by dropping a route.
    """

    def __init__(self, model, tokenizer, q2author, k, num_authors=200, delete_shard=None,
                 merged_adapter=None, reroute_to=None):
        """`delete_shard` accepts an int (legacy) or an iterable of shard ids — a per-author pool
        needs to delete twenty units to express TOFU's forget10.

        `reroute_to` turns the deletion into a **reassignment**: instead of the deleted authors'
        queries falling through to base+scaffold (which is what a weight-absent model would
        serve), they are answered by one fixed SURVIVING expert. Nothing is dropped. This is the
        E5 arm — a "method" that deletes nothing and only edits the serving policy — and it
        exists to measure whether TOFU's forget metric can tell the two apart.
        """
        super().__init__()
        if merged_adapter is not None and delete_shard is not None:
            raise ValueError("merged_adapter and delete_shard are mutually exclusive")
        self.model = model
        self.tokenizer = tokenizer
        self.q2author = q2author
        self.per_shard = num_authors // k
        if delete_shard is None:
            self.delete_shards = frozenset()
        elif isinstance(delete_shard, int):
            self.delete_shards = frozenset({delete_shard})
        else:
            self.delete_shards = frozenset(int(s) for s in delete_shard)
        self.delete_shard = delete_shard  # kept for callers/labels that read it back
        if reroute_to is not None:
            if not self.delete_shards:
                raise ValueError("reroute_to needs a non-empty delete set")
            if int(reroute_to) in self.delete_shards:
                raise ValueError(f"reroute_to={reroute_to} is itself deleted")
            if not (0 <= int(reroute_to) < k):
                raise ValueError(f"reroute_to={reroute_to} out of range [0,{k})")
            reroute_to = int(reroute_to)
        self.reroute_to = reroute_to
        self.merged_adapter = merged_adapter
        self.stats = {"routed": 0, "ood": 0, "deleted": 0, "rerouted": 0}

    @property
    def config(self):
        return self.model.config

    def set_adapter(self, name):
        pass

    def _shard_for(self, ids_1d):
        q = self.tokenizer.decode(ids_1d, skip_special_tokens=True)
        author = self.q2author.get(_norm(_extract_question(q)))
        if author is None:
            self.stats["ood"] += 1
            return None
        sid = author // self.per_shard
        if sid in self.delete_shards:
            if self.reroute_to is not None:
                # nothing was dropped: a surviving expert answers for the deleted author
                self.stats["rerouted"] += 1
                return self.reroute_to
            # expert dropped -> serve base+scaffold only (exact O(1) deletion of these authors)
            self.stats["deleted"] += 1
            return None
        self.stats["routed"] += 1
        return sid

    def _adapter_for(self, sid):
        return self.merged_adapter if self.merged_adapter is not None else f"shard_{sid}"

    def _call_one(self, inp, mask, lab, **kw):
        sid = self._shard_for(inp[0])
        if sid is None:
            with self.model.disable_adapter():
                return self.model(inp, attention_mask=mask, labels=lab, **kw)
        self.model.set_adapter(self._adapter_for(sid))
        return self.model(inp, attention_mask=mask, labels=lab, **kw)

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        B = input_ids.shape[0]
        if B == 1:
            return self._call_one(input_ids, attention_mask, labels, **kw)
        all_logits, loss_sum, toks = [], 0.0, 0
        for i in range(B):
            inp = input_ids[i:i+1]
            mask = attention_mask[i:i+1] if attention_mask is not None else None
            lab = labels[i:i+1] if labels is not None else None
            out = self._call_one(inp, mask, lab, **kw)
            all_logits.append(out.logits)
            if out.loss is not None and lab is not None:
                n = (lab != -100).sum().item()
                loss_sum += out.loss.item() * n
                toks += n
        logits = torch.cat(all_logits, dim=0)
        loss = torch.tensor(loss_sum / toks, device=input_ids.device) if toks > 0 else None
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids, **kw):
        sid = self._shard_for(input_ids[0])
        if sid is None:
            with self.model.disable_adapter():
                return self.model.generate(input_ids, **kw)
        self.model.set_adapter(self._adapter_for(sid))
        return self.model.generate(input_ids, **kw)


class EmbedRoutedModel(OODAwareRoutedModel):
    """Realistic embedding-routed serving (router-leak R2 arm): TOFU-author queries are
    routed by nearest shard CENTROID (MiniLM, `build_shard_centroids`) instead of the exact
    q2author shard lookup. OOD detection stays oracle-gated (q2author decides TOFU vs OOD)
    so the deletion cells aren't confounded by the known merged-onto-OOD composition bug.

    Deletion policies (`policy`, active only with delete_shard set):
      sibling   — the deleted shard's centroid is REMOVED from the pool; its authors'
                  queries fall to the nearest surviving sibling shard (the pure leak).
      tombstone — the deleted centroid is KEPT as an identity sentinel; a query whose
                  top-1 is the deleted shard serves base+scaffold (stats['deleted']),
                  everything else routes normally. Weights of the deleted shard are
                  never applied under either policy.
    Routing depends only on question embeddings, never on expert weights — so route
    stats must be BIT-IDENTICAL across expert pools (the R4 silent-failure assert)."""

    def __init__(self, model, tokenizer, q2author, k, centroids, centroid_sids, embed_fn,
                 num_authors=200, delete_shard=None, policy="sibling",
                 author_sentinels=None, tombstone_tau=None):
        super().__init__(model, tokenizer, q2author, k, num_authors=num_authors,
                         delete_shard=None)   # base-class deletion path unused
        if policy not in ("sibling", "tombstone", "tombstone_author"):
            raise ValueError(f"unknown embed-route policy {policy!r}")
        self.embed_delete_shard = delete_shard
        self.policy = policy
        self.embed_fn = embed_fn
        sids = [int(s) for s in centroid_sids]
        # tombstone_author (H3 closer): route by the SURVIVING shard centroids, but first gate on a
        # per-deleted-AUTHOR identity sentinel with a calibrated threshold — abstain to base+scaffold
        # when (best deleted-author-sentinel sim − best surviving-centroid sim) > tau. The deleted
        # centroid is NOT kept (survivors only); the sentinels are recomputable from the deletion
        # request, and tau is calibrated on RETAIN margins (never forget). Predicted served retain
        # cost ~0 at 90% catch, vs the shard-argmax rung's Δmu −0.0061 (the H3 gap).
        if policy == "tombstone_author":
            if delete_shard is None or author_sentinels is None or tombstone_tau is None:
                raise ValueError("tombstone_author needs delete_shard, author_sentinels, tombstone_tau")
            keep = [i for i, s in enumerate(sids) if s != delete_shard]
            centroids, sids = centroids[keep], [sids[i] for i in keep]
            self.author_sentinels = np.asarray(author_sentinels, dtype="float32")
            self.tombstone_tau = float(tombstone_tau)
            self.stats.update({"deleted": 0})
        elif delete_shard is not None and policy == "sibling":
            keep = [i for i, s in enumerate(sids) if s != delete_shard]
            if len(keep) == len(sids):
                raise ValueError(f"sibling policy: delete_shard {delete_shard} not in centroid pool")
            centroids, sids = centroids[keep], [sids[i] for i in keep]
        elif delete_shard is not None and policy == "tombstone" and delete_shard not in sids:
            raise ValueError(f"tombstone policy needs the deleted centroid in the pool")
        self.centroids, self.centroid_sids = centroids, sids
        self.stats.update({"route_mismatch": 0})

    def _shard_for(self, ids_1d):
        q_text = self.tokenizer.decode(ids_1d, skip_special_tokens=True)
        q_raw = _extract_question(q_text)
        author = self.q2author.get(_norm(q_raw))
        if author is None:
            self.stats["ood"] += 1
            return None
        v = self.embed_fn([q_raw])[0]
        surv_sims = self.centroids @ v
        if self.policy == "tombstone_author":
            # identity gate first: abstain if the query matches a deleted-author sentinel
            # distinctly better than any surviving shard centroid.
            margin = float((self.author_sentinels @ v).max()) - float(surv_sims.max())
            if margin > self.tombstone_tau:
                self.stats["deleted"] += 1
                return None
            sid = self.centroid_sids[int(np.argmax(surv_sims))]
            if sid != author // self.per_shard:
                self.stats["route_mismatch"] += 1
            self.stats["routed"] += 1
            return sid
        sid = self.centroid_sids[int(np.argmax(surv_sims))]
        if (self.policy == "tombstone" and self.embed_delete_shard is not None
                and sid == self.embed_delete_shard):
            self.stats["deleted"] += 1
            return None
        if sid != author // self.per_shard:
            self.stats["route_mismatch"] += 1
        self.stats["routed"] += 1
        return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="scaffolded base (base+scaffold merged)")
    ap.add_argument("--shards_dir", required=True, help="dir with shard_0..k-1 (author experts)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--forget_shard_id", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delete_shard", type=int, default=None,
                    help="Exact O(1) deletion demo: drop this shard's expert -> its authors serve "
                         "base+scaffold only (== a model that never trained on them). Set = forget_shard_id "
                         "and read forget_quality on those authors.")
    ap.add_argument("--delete_shards", default=None,
                    help="Multi-unit deletion, e.g. '180-199' (inclusive ranges and commas). A "
                         "per-author pool (k=200) needs 20 units to express TOFU's forget10, "
                         "which --delete_shard cannot. Mutually exclusive with --delete_shard.")
    ap.add_argument("--reroute_to", type=int, default=None,
                    help="E5 (selector_audit): delete NOTHING and send the deleted authors' "
                         "queries to this fixed SURVIVING expert instead of to base+scaffold. The "
                         "arm exists to ask whether TOFU's forget metric can distinguish 'the "
                         "source is gone' from 'a stranger answers for it'. Requires a delete set.")
    ap.add_argument("--forget_author_ids", default=None,
                    help="Explicit forget authors for the METRICS, e.g. '180-199' (see "
                         "eval_tofu.split_eval_indices). At k=200 --forget_shard_id scores one "
                         "author's 20 questions; forget10 is 400.")
    ap.add_argument("--merged_label", default=None,
                    help="scaffold-x-composition control (arm B): build this merge label over the "
                         "loaded shard experts (merged_*/remerge_* via merge_lora.activate_label) and "
                         "serve ALL TOFU-author queries with it (OOD stays scaffold-only). remerge_* = "
                         "the merged-deployment deletion condition (forgotten authors get the retain "
                         "merge, Fig-8-style). Mutually exclusive with --delete_shard.")
    ap.add_argument("--embed_route", default=None, choices=["sibling", "tombstone", "tombstone_author"],
                    help="Router-leak R2 arm: route TOFU-author queries by nearest shard "
                         "centroid (MiniLM) instead of exact q2author. With --delete_shard: "
                         "'sibling' drops the deleted centroid (orphans fall to the nearest "
                         "surviving sibling = the leak), 'tombstone' keeps it as an identity "
                         "sentinel (its top-1 hits serve base+scaffold). Without "
                         "--delete_shard both give the embed-full baseline. OOD stays "
                         "oracle-gated. Mutually exclusive with --merged_label.")
    ap.add_argument("--router_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tombstone_tau", type=float, default=None,
                    help="tombstone_author only: abstain to base when the deleted-author identity "
                         "margin exceeds this. Calibrate on RETAIN margins (author-rung); 0.1944 = "
                         "the k=10 MiniLM 90%-catch / 0.11%-retain-FPR operating point.")
    ap.add_argument("--lazy_adapter_cache", type=int, default=0,
                    help="High-k memory-wall fix: keep at most N shard adapters resident "
                         "(eval_tofu.lazify_shard_adapters; loads on demand + LRU-evicts). "
                         "0 (default) = eager load of all k shards. Use e.g. 8 for k=200 r32 "
                         "on an A40. Incompatible with --merged_label (merging enumerates the "
                         "full adapter set).")
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", os.environ["HF_HOME"]))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extended", action="store_true")
    args = ap.parse_args()
    if args.lazy_adapter_cache and args.merged_label is not None:
        raise SystemExit("--lazy_adapter_cache is incompatible with --merged_label")
    if args.delete_shards is not None and args.delete_shard is not None:
        raise SystemExit("--delete_shard and --delete_shards are mutually exclusive")
    try:
        delete_set = parse_author_ids(args.delete_shards) if args.delete_shards else None
        forget_author_ids = parse_author_ids(args.forget_author_ids)
    except ValueError as e:
        raise SystemExit(str(e))
    if delete_set is not None and any(s >= args.k for s in delete_set):
        raise SystemExit(f"--delete_shards has ids >= k={args.k}: "
                         f"{[s for s in delete_set if s >= args.k]}")
    delete_arg = delete_set if delete_set is not None else args.delete_shard
    if args.reroute_to is not None:
        if delete_arg is None:
            raise SystemExit("--reroute_to needs --delete_shard or --delete_shards")
        if args.embed_route is not None:
            raise SystemExit("--reroute_to is an oracle-route arm; --embed_route already "
                             "reassigns orphans by nearest surviving centroid")
    os.environ["HF_HOME"] = args.hf_home
    forget_id = args.forget_shard_id if args.forget_shard_id is not None else args.k - 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prog = ProgressLogger(args.out.replace(".json", ".progress.json"), "routed_scaffold_ood")
    data = load_tofu_data(args.hf_home)
    shards = {i: get_author_shard(args.k, i) for i in range(args.k)}
    q2author = build_q2author(data["full"], 200, 20)

    model, tokenizer = load_all_shard_adapters(args.model_name, args.shards_dir, args.k,
                                               lazy_cache=args.lazy_adapter_cache)
    merged_name = None
    if args.merged_label is not None:
        if args.delete_shard is not None:
            raise SystemExit("--merged_label and --delete_shard are mutually exclusive "
                             "(merged-mode deletion = a remerge_* label)")
        if args.embed_route is not None:
            raise SystemExit("--merged_label and --embed_route are mutually exclusive")
        from merge_lora import activate_label
        merged_name = activate_label(model, args.k, forget_id, args.merged_label)
        if not isinstance(merged_name, str):
            raise SystemExit(f"--merged_label must resolve to a single adapter, got {type(merged_name)}")
    if args.embed_route is not None:
        cents, sids, embed_fn = build_shard_centroids(
            args.hf_home, args.k, list(range(args.k)), "cuda" if torch.cuda.is_available() else "cpu",
            encoder_name=args.router_encoder)
        author_sents = None
        if args.embed_route == "tombstone_author":
            if args.delete_shard is None or args.tombstone_tau is None:
                raise SystemExit("tombstone_author needs --delete_shard and --tombstone_tau")
            # per-author identity sentinels for the DELETED shard: mean question embedding per
            # author (recomputable from the deletion request). Reuses the router encoder.
            full = data["full"]
            rows = []
            for a in get_author_shard(args.k, args.delete_shard):
                vecs = embed_fn([full[a * 20 + w]["question"] for w in range(20)])
                m = np.asarray(vecs).mean(0)
                rows.append(m / (np.linalg.norm(m) + 1e-12))
            author_sents = np.stack(rows).astype("float32")
        eval_model = EmbedRoutedModel(model, tokenizer, q2author, args.k, cents, sids, embed_fn,
                                      delete_shard=args.delete_shard, policy=args.embed_route,
                                      author_sentinels=author_sents, tombstone_tau=args.tombstone_tau)
    else:
        eval_model = OODAwareRoutedModel(model, tokenizer, q2author, args.k,
                                         delete_shard=delete_arg, merged_adapter=merged_name,
                                         reroute_to=args.reroute_to)

    results_sub = "smoke" if args.smoke else ("extended" if args.extended else "")
    retain_tr_path = os.path.join(args.shards_dir, "results", results_sub, "retain_tr_scores.npy")
    retain_ref = np.load(retain_tr_path) if os.path.exists(retain_tr_path) else None

    rouge_n, retain_n, truth_n = None, 500, None
    if args.smoke:
        rouge_n, retain_n, truth_n = SMOKE_ROUGE_MAX, SMOKE_RETAIN_MAX, SMOKE_TRUTH_MAX
    elif args.extended:
        rouge_n, retain_n, truth_n = EXTENDED_ROUGE_MAX, EXTENDED_RETAIN_MAX, EXTENDED_TRUTH_MAX

    if args.embed_route is not None:
        run_label = ("embedrouted_full" if args.delete_shard is None
                     else f"embedrouted_{args.embed_route}_del{args.delete_shard}")
    elif args.merged_label:
        run_label = f"scafmerged_{args.merged_label}"
    elif args.reroute_to is not None:
        run_label = f"routed_reroute_s{args.reroute_to}"
    elif delete_set is not None:
        run_label = f"routed_oracle_del{len(delete_set)}units"
    else:
        run_label = "routed_scaffold_ood"
    row = evaluate_model(
        eval_model, tokenizer, run_label, forget_shard_id=forget_id,
        full_ds=data["full"], shards=shards, forget10_pert=data["forget10_pert"],
        real_authors=data["real_authors"], world_facts=data["world_facts"],
        retain_ref_tr_scores=retain_ref, rouge_max_samples=rouge_n, prog=prog,
        smoke=args.smoke, extended=args.extended, retain_max_samples=retain_n, truth_max_rows=truth_n,
        full_pert=data["full_pert"], real_authors_pert=data["real_authors_pert"],
        world_facts_pert=data["world_facts_pert"], forget_author_ids=forget_author_ids)
    row["route_stats"] = eval_model.stats
    row["deletion"] = {"delete_shards": sorted(eval_model.delete_shards),
                       "reroute_to": eval_model.reroute_to,
                       "forget_author_ids": forget_author_ids}
    # The failure mode these arms are most exposed to is a plausible-but-wrong route, which no
    # metric would flag. Assert the served policy matches the requested one before anything is
    # read off the numbers.
    n_orphan_q = 20 * len(forget_author_ids) if forget_author_ids else None
    if n_orphan_q is not None and eval_model.delete_shards:
        served = (eval_model.stats["rerouted"] if args.reroute_to is not None
                  else eval_model.stats["deleted"])
        if served != n_orphan_q:
            raise SystemExit(
                f"route audit FAILED: {served} orphan queries took the deletion path, expected "
                f"{n_orphan_q} ({len(forget_author_ids)} authors x 20). stats={eval_model.stats}")
    import json
    json.dump(row, open(args.out, "w"), indent=2)
    print(f"[{run_label}] mu={row['model_utility']:.4f} forget_rouge={row.get('forget_rouge',float('nan')):.3f} "
          f"routed={eval_model.stats['routed']} ood={eval_model.stats['ood']} -> {args.out}")


if __name__ == "__main__":
    main()
