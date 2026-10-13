"""
Reciprocal Rank Fusion (RRF).

For each ranked list, an item at 1-based rank r contributes  weight / (k + r).
Contributions are summed across lists; higher total = better.

RRF uses ranks only, so it needs no score normalisation between BM25 scores
(unbounded) and cosine similarities (roughly 0..1). That is why it is a
common default for hybrid search.

Note: with k=60 the fused scores are tiny (a top-1 hit in both lists scores
2/61 ~= 0.033). They are only meaningful for ordering. Do not threshold on them.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[Hashable, float]]:
    """
    Args:
        rankings: one list of item ids per retriever, best first.
        k: RRF constant (damps the influence of top ranks). Must be > 0.
        weights: optional per-list weights (default: all 1.0).
        top_n: truncate the fused list to this many items.

    Returns:
        [(item_id, fused_score), ...] sorted best first. Ties are broken by
        which item appeared earliest across the input lists, so results are
        deterministic.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have the same length as rankings")

    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}

    for list_idx, ranking in enumerate(rankings):
        w = weights[list_idx] if weights is not None else 1.0
        seen_in_list: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen_in_list:  # a duplicate within one list counts once (first rank)
                continue
            seen_in_list.add(item)
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            first_seen.setdefault(item, len(first_seen))

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return fused[:top_n] if top_n is not None else fused