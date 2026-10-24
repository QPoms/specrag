"""Provider adapter for the local Chroma vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _metadata(chunk: Any) -> dict[str, str | int]:
    return {
        "source_collection": chunk.source_collection,
        "specification": chunk.specification or "",
        "version": chunk.version or "",
        "source_path": chunk.source_path,
        "document_sha256": chunk.document_sha256,
        "heading_path": json.dumps(chunk.heading_path, ensure_ascii=False),
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
    }


def open_chroma_collection(
    directory: Path,
    chunks: tuple[Any, ...],
    *,
    embedding_model: str,
    document_text: Callable[[Any], str],
    retain_extra: bool = False,
    add_missing: bool = True,
):
    """Open a local collection and optionally add missing source chunks."""

    import chromadb
    from chromadb.utils import embedding_functions

    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        normalize_embeddings=True,
        local_files_only=True,
    )
    client = chromadb.PersistentClient(path=str(directory / "chroma"))
    name = "specification_evidence"
    if add_missing:
        collection = client.get_or_create_collection(
            name,
            embedding_function=embedding,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        try:
            collection = client.get_collection(
                name,
                embedding_function=embedding,
            )
        except Exception as exc:
            raise RuntimeError("vector index is not ready") from exc
    expected_ids = {item.chunk_id for item in chunks}
    existing_ids = set(collection.get(include=[]).get("ids", []))
    if not retain_extra and not existing_ids.issubset(expected_ids):
        client.delete_collection(name)
        collection = client.create_collection(
            name,
            embedding_function=embedding,
            metadata={"hnsw:space": "cosine"},
        )
        existing_ids = set()
    missing = tuple(item for item in chunks if item.chunk_id not in existing_ids)
    if missing and not add_missing:
        raise RuntimeError(
            f"vector index is not ready: {len(missing)} chunks are missing"
        )
    for offset in range(0, len(missing), 128):
        batch = missing[offset : offset + 128]
        collection.add(
            ids=[item.chunk_id for item in batch],
            documents=[document_text(item) for item in batch],
            metadatas=[_metadata(item) for item in batch],
        )
    return collection


__all__ = ["open_chroma_collection"]
