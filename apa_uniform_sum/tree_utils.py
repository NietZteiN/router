"""Tree structure helpers for hierarchical SISA LoRA merging.

Shards are arranged in a balanced binary tree built by recursively splitting
contiguous shard-index ranges at the midpoint. Each internal node covers a
range [lo, hi] (inclusive) and its adapter is the merged result of its two
children. Leaves are the existing shard_{i} adapters; they are never renamed.

Node naming
-----------
  leaf:  shard_{i}              (lo == hi, adapter already exists)
  internal: tnode_{lo}_{hi}_{method}  (e.g. tnode_0_9_linear)

The method suffix prevents collisions when multiple merge methods are
evaluated on the same loaded PeftModel.

Invariants
----------
  split(lo, hi) always produces two non-empty halves (lo <= hi required).
  path_to_root returns nodes from the immediate parent of shard_id up to the
  root [0, k-1], in that order (parent first, root last).
  internal_nodes_postorder guarantees children appear before parents so a
  simple linear scan builds the tree bottom-up without recursion.
"""


def node_name(lo: int, hi: int, method: str) -> str:
    """Adapter name for the tree node covering shards lo..hi (inclusive)."""
    if lo == hi:
        return f"shard_{lo}"
    return f"tnode_{lo}_{hi}_{method}"


def split(lo: int, hi: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Split [lo, hi] into left [lo, mid] and right [mid+1, hi]."""
    assert lo < hi, f"split requires lo < hi, got lo={lo} hi={hi}"
    mid = (lo + hi) // 2
    return (lo, mid), (mid + 1, hi)


def path_to_root(k: int, shard_id: int) -> list[tuple[int, int]]:
    """Return internal-node ranges from shard_id's parent up to root.

    Result is ordered parent-first, root-last:
      path_to_root(4, 1) -> [(0, 1), (0, 3)]
      path_to_root(4, 2) -> [(2, 3), (0, 3)]
    """
    assert 0 <= shard_id < k, f"shard_id={shard_id} out of range [0, {k})"
    ancestors: list[tuple[int, int]] = []
    lo, hi = 0, k - 1
    while lo < hi:
        ancestors.append((lo, hi))
        mid = (lo + hi) // 2
        if shard_id <= mid:
            hi = mid
        else:
            lo = mid + 1
    # ancestors is root-first; reverse to parent-first
    return list(reversed(ancestors))


def internal_nodes_postorder(k: int) -> list[tuple[int, int]]:
    """All internal (non-leaf) nodes in post-order (children before parents).

    Suitable for bottom-up tree construction:
      internal_nodes_postorder(4) -> [(0,1), (2,3), (0,3)]
    """
    result: list[tuple[int, int]] = []

    def _visit(lo: int, hi: int) -> None:
        if lo == hi:
            return
        (llo, lhi), (rlo, rhi) = split(lo, hi)
        _visit(llo, lhi)
        _visit(rlo, rhi)
        result.append((lo, hi))

    _visit(0, k - 1)
    return result
