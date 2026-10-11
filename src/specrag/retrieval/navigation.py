"""Deterministic clause lookup and same-section source navigation."""

from __future__ import annotations

import re

from .models import MarkdownChunk


def lookup_clause_chunks(
    chunks: tuple[MarkdownChunk, ...],
    clause: str,
    *,
    specifications: tuple[str, ...] = (),
    limit: int = 10,
) -> tuple[MarkdownChunk, ...]:
    """Find chunks whose heading contains one exact clause identifier."""

    if not clause.strip():
        raise ValueError("clause must not be blank")
    if limit < 1:
        raise ValueError("limit must be positive")
    allowed = frozenset(specifications)
    pattern = re.compile(
        rf"(?<![0-9.]){re.escape(clause.strip())}(?![0-9.])",
        re.IGNORECASE,
    )
    matches = [
        chunk
        for chunk in chunks
        if (not allowed or chunk.specification in allowed)
        and any(pattern.search(heading) for heading in chunk.heading_path)
    ]
    return tuple(sorted(
        matches,
        key=lambda chunk: (
            chunk.specification or "",
            chunk.source_path,
            chunk.line_start,
            chunk.chunk_id,
        ),
    )[:limit])


def expand_same_section_chunks(
    chunks: tuple[MarkdownChunk, ...],
    seed_chunk_ids: tuple[str, ...],
    *,
    limit: int = 8,
) -> tuple[MarkdownChunk, ...]:
    """Return bounded source neighbours from each seed's exact section."""

    if not seed_chunk_ids:
        raise ValueError("at least one seed chunk is required")
    if limit < 1:
        raise ValueError("limit must be positive")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    unknown = tuple(chunk_id for chunk_id in seed_chunk_ids if chunk_id not in by_id)
    if unknown:
        raise ValueError("unknown seed chunks: " + ", ".join(unknown))
    seed_ids = frozenset(seed_chunk_ids)
    selected: list[MarkdownChunk] = []
    seen: set[str] = set()
    for seed_id in seed_chunk_ids:
        seed = by_id[seed_id]
        siblings = sorted(
            (
                chunk
                for chunk in chunks
                if chunk.source_path == seed.source_path
                and chunk.specification == seed.specification
                and chunk.version == seed.version
                and chunk.heading_path == seed.heading_path
            ),
            key=lambda chunk: (chunk.line_start, chunk.line_end, chunk.chunk_id),
        )
        seed_index = next(
            index for index, chunk in enumerate(siblings) if chunk.chunk_id == seed_id
        )
        candidate_indexes = tuple(dict.fromkeys((
            seed_index - 1,
            seed_index + 1,
            0,
        )))
        for index in candidate_indexes:
            if not 0 <= index < len(siblings):
                continue
            candidate = siblings[index]
            if candidate.chunk_id in seed_ids or candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            selected.append(candidate)
            if len(selected) >= limit:
                return tuple(selected)
    return tuple(selected)


__all__ = ["expand_same_section_chunks", "lookup_clause_chunks"]
