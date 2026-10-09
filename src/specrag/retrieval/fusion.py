"""Deterministic reciprocal-rank fusion for lexical and semantic candidates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedRanking:
    ordered_chunk_ids: tuple[str, ...]
    semantic_rank: dict[str, int]
    semantic_distance: dict[str, float]
    lexical_rank: dict[str, int]
    lexical_score: dict[str, float]
    rrf_score: dict[str, float]


def reciprocal_rank_fusion(
    semantic: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    *,
    rrf_k: int,
    limit: int,
) -> FusedRanking:
    semantic_rank = {
        chunk_id: rank for rank, (chunk_id, _) in enumerate(semantic, 1)
    }
    semantic_distance = dict(semantic)
    lexical_rank = {
        chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, 1)
    }
    lexical_score = dict(lexical)
    scores: dict[str, float] = defaultdict(float)
    for chunk_id, rank in semantic_rank.items():
        scores[chunk_id] += 1 / (rrf_k + rank)
    for chunk_id, rank in lexical_rank.items():
        scores[chunk_id] += 1 / (rrf_k + rank)
    ordered = tuple(sorted(scores, key=lambda item: (-scores[item], item))[:limit])
    return FusedRanking(
        ordered_chunk_ids=ordered,
        semantic_rank=semantic_rank,
        semantic_distance=semantic_distance,
        lexical_rank=lexical_rank,
        lexical_score=lexical_score,
        rrf_score=dict(scores),
    )


__all__ = ["FusedRanking", "reciprocal_rank_fusion"]
