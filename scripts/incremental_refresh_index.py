"""Create a specification-retrieval snapshot by reusing an older vector index.

The source-document fingerprint changes when a specification is added.  This
utility copies a compatible immutable snapshot, rewrites its manifest/chunks,
and lets Chroma embed only chunk IDs that are new in the refreshed corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from specrag.config import load_config
from specrag.retrieval.hybrid import (
    RetrievalIndexManifest,
    _fingerprint,
    _open_vector_collection,
    _repository_path,
    _write_snapshot,
    chunk_source_documents,
    discover_source_documents,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fingerprint", required=True)
    parser.add_argument("--release", type=int, required=True)
    args = parser.parse_args()

    base_cfg = load_config().retrieval
    cfg = base_cfg.model_copy(update={
        "version_policy": "highest_version_within_release",
        "target_release": args.release,
    })
    documents = discover_source_documents(cfg)
    fingerprint = _fingerprint(documents, cfg)
    index_root = _repository_path(cfg.index_dir, "retrieval index directory")
    base_snapshot = index_root / args.base_fingerprint
    snapshot = index_root / fingerprint
    if not base_snapshot.is_dir():
        raise FileNotFoundError(f"base snapshot not found: {base_snapshot}")
    if snapshot.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {snapshot}")

    base_chunk_ids = {
        json.loads(line)["chunk_id"]
        for line in (base_snapshot / "chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    chunks = chunk_source_documents(documents, cfg)
    expected_ids = {chunk.chunk_id for chunk in chunks}
    if not base_chunk_ids.issubset(expected_ids):
        removed = sorted(base_chunk_ids - expected_ids)
        raise ValueError(
            "base snapshot is not append-compatible; "
            f"{len(removed)} existing chunk IDs would be removed"
        )

    shutil.copytree(base_snapshot, snapshot)
    manifest = RetrievalIndexManifest(
        index_fingerprint=fingerprint,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        chunk_characters=cfg.chunk_characters,
        chunk_overlap_characters=cfg.chunk_overlap_characters,
        collection_name="specification_evidence",
        documents=documents,
        chunk_count=len(chunks),
    )
    _write_snapshot(snapshot, chunks, manifest)
    collection = _open_vector_collection(snapshot, chunks, cfg)
    indexed_ids = set(collection.get(include=[]).get("ids", []))
    missing = expected_ids - indexed_ids
    if missing:
        raise RuntimeError(f"vector index is missing {len(missing)} chunks")

    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in base_chunk_ids]
    report = {
        "base_fingerprint": args.base_fingerprint,
        "index_fingerprint": fingerprint,
        "release": args.release,
        "documents": len(documents),
        "chunks": len(chunks),
        "reused_chunk_embeddings": len(base_chunk_ids),
        "new_chunk_embeddings": len(new_chunks),
        "new_chunks_by_specification": {
            specification: sum(
                chunk.specification == specification for chunk in new_chunks
            )
            for specification in sorted({
                chunk.specification
                for chunk in new_chunks
                if chunk.specification is not None
            })
        },
        "snapshot": str(snapshot),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
