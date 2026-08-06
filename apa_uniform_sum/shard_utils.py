"""Shared SISA shard utilities — single source of truth for author-to-shard mapping.

With k=10, shard_9 = authors 180-199 = exactly TOFU's forget10 split, giving full
perturbed-answer coverage for truth-ratio evaluation.
"""


def get_author_shard(k: int, shard_id: int) -> list:
    """Return sorted author indices (0-based out of 200) assigned to this shard."""
    assert 200 % k == 0, f"k={k} must evenly divide 200"
    authors_per_shard = 200 // k
    start = shard_id * authors_per_shard
    return list(range(start, start + authors_per_shard))


# ── S³T (Bourtoule-style slices inside SISA shards; ICLR'25 S3T) ─────────────
#
# Each S3T shard's authors are split into L natural slices; training proceeds in
# L sequential stages. The ordering (a permutation of natural slice indices)
# decides which natural slice is trained at which stage. Stage data is
# CUMULATIVE (stage j trains on the slices at positions 0..j), matching the
# official implementation (S3T/src/s3t_llm.py: train[0%:(j+1)*100/L%]).
#
# Ordering policy (single source of truth for the m=5/L=4 TOFU instantiation):
#   - shard without forget authors: cyclic rotation by shard_id (the paper's
#     uniform-prior sequence-selection strategy; cost-free diversity).
#   - shard containing forget authors: non-forget slices first (natural order),
#     forget slices LAST — the paper's BMS outcome for a point-mass deletion
#     prior. Deleting the forget data then = revert to the snapshot taken
#     before the first forget slice; earlier stages never saw it.

DEFAULT_FORGET_AUTHORS = tuple(range(180, 200))  # TOFU forget10


def get_s3t_shard_authors(m: int, shard_id: int) -> list:
    """Author indices of S3T shard `shard_id` out of `m` equal author shards."""
    assert 200 % m == 0, f"m={m} must evenly divide 200"
    assert 0 <= shard_id < m, f"shard_id={shard_id} out of range for m={m}"
    per = 200 // m
    return list(range(shard_id * per, (shard_id + 1) * per))


def _s3t_natural_slices(m: int, shard_id: int, L: int) -> list:
    """List of L author-lists: natural slice i of the shard (contiguous authors)."""
    authors = get_s3t_shard_authors(m, shard_id)
    assert len(authors) % L == 0, f"L={L} must evenly divide {len(authors)} shard authors"
    per = len(authors) // L
    return [authors[i * per:(i + 1) * per] for i in range(L)]


def get_s3t_ordering(m: int, shard_id: int, L: int,
                     forget_authors=DEFAULT_FORGET_AUTHORS) -> list:
    """Permutation: ordering[j] = natural slice index trained at stage j."""
    slices = _s3t_natural_slices(m, shard_id, L)
    forget = set(forget_authors)
    # Every natural slice must be entirely forget or entirely non-forget,
    # otherwise no ordering can make the deletion exact at slice granularity.
    is_forget = []
    for i, sl in enumerate(slices):
        inter = forget.intersection(sl)
        assert not inter or inter == set(sl), (
            f"slice {i} of shard {shard_id} straddles the forget set: {sorted(inter)}"
        )
        is_forget.append(bool(inter))
    if not any(is_forget):
        return [(shard_id + j) % L for j in range(L)]
    return [i for i in range(L) if not is_forget[i]] + [i for i in range(L) if is_forget[i]]


def get_s3t_slice_authors(m: int, shard_id: int, L: int, stage_j: int,
                          ordering=None) -> list:
    """Authors of the natural slice trained at stage `stage_j` under `ordering`."""
    if ordering is None:
        ordering = get_s3t_ordering(m, shard_id, L)
    return _s3t_natural_slices(m, shard_id, L)[ordering[stage_j]]


def get_s3t_stage_authors(m: int, shard_id: int, L: int, stage_j: int,
                          ordering=None) -> list:
    """Cumulative author set for stage `stage_j` = slices at positions 0..stage_j."""
    if ordering is None:
        ordering = get_s3t_ordering(m, shard_id, L)
    authors = []
    for j in range(stage_j + 1):
        authors.extend(get_s3t_slice_authors(m, shard_id, L, j, ordering))
    return sorted(authors)


def get_s3t_layer_block(stage_j: int, num_loras: int, num_layers: int) -> list:
    """Decoder-layer indices whose LoRA params stage `stage_j` trains (top-down).

    Official mapping (S3T/src/s3t_llm.py get_layers): stage 0 gets the TOP
    `num_loras` layers, each later stage the next block down. Blocks are
    disjoint; with L*num_loras == num_layers they cover the whole stack.
    """
    block = [num_layers - 1 - (stage_j * num_loras + i) for i in range(num_loras)]
    assert min(block) >= 0, (
        f"stage {stage_j}: layer block {block} below layer 0 "
        f"(num_loras={num_loras} too large for {num_layers} layers)"
    )
    return block
