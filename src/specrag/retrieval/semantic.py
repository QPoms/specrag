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
    allowed_document_hashes: tuple[str, ...] = (),
) -> list[tuple[str, float]]:
    where_clauses: tuple[dict[str, Any], ...]
    document_filter = (
        [{"document_sha256": {"$in": list(allowed_document_hashes)}}]
        if allowed_document_hashes else []
    )
    if source_collection == "3gpp_markdown" and allowed_specifications:
        where_clauses = tuple(
            {
                "$and": [
                    {"source_collection": source_collection},
                    {"specification": specification},
                    *document_filter,
                ]
            }
            for specification in allowed_specifications
        )
    elif document_filter:
        where_clauses = ({
            "$and": [
                {"source_collection": source_collection},
                *document_filter,
            ]
        },)
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
