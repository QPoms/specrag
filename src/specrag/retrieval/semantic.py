"""Provider-neutral dense candidate retrieval."""

from __future__ import annotations

from typing import Any

from ..config import RetrievalCfg


def dense_search(
    collection: Any,
    cfg: RetrievalCfg,
    text: str,
    source_collection: str,
    allowed_specifications: tuple[str, ...] = (),
) -> list[tuple[str, float]]:
    where_clauses: tuple[dict[str, Any], ...]
    if source_collection == "3gpp_markdown" and allowed_specifications:
        where_clauses = tuple(
            {
                "$and": [
                    {"source_collection": source_collection},
                    {"specification": specification},
                ]
            }
            for specification in allowed_specifications
        )
    else:
        where_clauses = ({"source_collection": source_collection},)

    candidates: dict[str, float] = {}
    for where in where_clauses:
        result = collection.query(
            query_texts=[text],
            n_results=cfg.semantic_candidates_per_collection,
            where=where,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, distance in zip(ids, distances):
            value = float(distance)
            candidates[chunk_id] = min(candidates.get(chunk_id, value), value)
    return sorted(candidates.items(), key=lambda item: (item[1], item[0]))[
        : cfg.semantic_candidates_per_collection
    ]


__all__ = ["dense_search"]
