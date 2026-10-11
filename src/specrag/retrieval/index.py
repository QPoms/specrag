"""Immutable retrieval snapshot persistence and fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import RetrievalCfg
from .corpus import corpus_scope_from_config
from .models import (
    CorpusScope,
    MarkdownChunk,
    RetrievalIndexManifest,
    SourceDocument,
)


def index_fingerprint(
    documents: tuple[SourceDocument, ...],
    cfg: RetrievalCfg,
    scope: CorpusScope | None = None,
) -> str:
    active_scope = scope or corpus_scope_from_config(cfg)
    payload = {
        "schema": "specification-retrieval-index/v1.1",
        "embedding_model": cfg.embedding_model,
        "corpus_scope": active_scope.model_dump(mode="json"),
        "chunk_characters": cfg.chunk_characters,
        "chunk_overlap_characters": cfg.chunk_overlap_characters,
        "documents": [item.model_dump(mode="json") for item in documents],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def write_snapshot(
    directory: Path,
    chunks: tuple[MarkdownChunk, ...],
    manifest: RetrievalIndexManifest,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for item in chunks
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_chunks(path: Path) -> tuple[MarkdownChunk, ...]:
    return tuple(
        MarkdownChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def searchable_text(chunk: MarkdownChunk) -> str:
    if not chunk.heading_path:
        return chunk.text
    return "\n".join((*chunk.heading_path, chunk.text))


__all__ = [
    "index_fingerprint",
    "load_chunks",
    "searchable_text",
    "write_snapshot",
]
