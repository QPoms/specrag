from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrag.config import RetrievalCfg
from specrag.retrieval import corpus
from specrag.retrieval.hybrid import (
    HybridMarkdownRetriever,
    MarkdownChunk,
    SourceDocument,
    chunk_source_documents,
    discover_source_documents,
)
from specrag.retrieval.lexical import BM25Index


def _write_specification(
    root: Path,
    specification: str,
    version: str,
    text: str,
) -> Path:
    source = root / "data" / "3gpp" / specification / f"v{version}"
    source.mkdir(parents=True, exist_ok=True)
    path = source / f"TS_{specification.replace('.', '')}.md"
    path.write_text(text, encoding="utf-8")
    return path


def _cfg(**overrides: object) -> RetrievalCfg:
    values: dict[str, object] = {
        "specification_root": "data/3gpp",
        "index_dir": ".local/indexes/specifications",
        "embedding_model": "unused-in-characterization-tests",
        "chunk_characters": 80,
        "chunk_overlap_characters": 10,
    }
    values.update(overrides)
    return RetrievalCfg(**values)


def _chunk(chunk_id: str, text: str, specification: str) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        source_collection="3gpp_markdown",
        specification=specification,
        version="17.1.0",
        source_path=f"data/3gpp/{specification}/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=("5 Procedure",),
        line_start=1,
        line_end=2,
        text=text,
    )


def test_latest_policy_selects_one_highest_version_per_specification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = _write_specification(tmp_path, "24.501", "17.9.0", "older")
    latest = _write_specification(tmp_path, "24.501", "18.2.0", "latest")
    other = _write_specification(tmp_path, "33.501", "16.4.0", "other")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)

    documents = discover_source_documents(_cfg())

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "18.2.0"),
        ("33.501", "16.4.0"),
    ]
    selected = documents[0]
    assert selected.source_path == latest.relative_to(tmp_path).as_posix()
    assert selected.document_sha256 == hashlib.sha256(latest.read_bytes()).hexdigest().upper()
    assert older.relative_to(tmp_path).as_posix() not in {
        item.source_path for item in documents
    }
    assert other.relative_to(tmp_path).as_posix() in {
        item.source_path for item in documents
    }


def test_release_policy_selects_highest_version_inside_requested_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_specification(tmp_path, "24.501", "17.3.0", "release 17 lower")
    _write_specification(tmp_path, "24.501", "17.9.0", "release 17 higher")
    _write_specification(tmp_path, "24.501", "18.1.0", "release 18")
    _write_specification(tmp_path, "33.501", "18.1.0", "no release 17")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)

    documents = discover_source_documents(_cfg(
        version_policy="highest_version_within_release",
        target_release=17,
    ))

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "17.9.0"),
    ]


def test_chunking_preserves_heading_and_exact_source_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_specification(
        tmp_path,
        "24.501",
        "17.1.0",
        "ignored preamble\n# 5 Procedure\nThe UE shall start authentication.\n"
        "## 5.1 Failure\nThe network shall reject the request.\n",
    )
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    document = SourceDocument(
        source_collection="3gpp_markdown",
        specification="24.501",
        version="17.1.0",
        source_path=source.relative_to(tmp_path).as_posix(),
        document_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),
    )

    chunks = chunk_source_documents((document,), _cfg(chunk_characters=200))

    assert [chunk.heading_path for chunk in chunks] == [
        ("5 Procedure",),
        ("5 Procedure", "5.1 Failure"),
    ]
    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (2, 3),
        (4, 5),
    ]
    assert "ignored preamble" not in "\n".join(chunk.text for chunk in chunks)


def test_lexical_results_survive_unavailable_semantic_backend() -> None:
    chunks = (
        _chunk("C1", "registration authentication procedure", "24.501"),
        _chunk("C2", "security key derivation", "33.501"),
    )
    retriever = HybridMarkdownRetriever.__new__(HybridMarkdownRetriever)
    retriever.cfg = SimpleNamespace(
        semantic_candidates_per_collection=10,
        lexical_candidates_per_collection=10,
        final_hits_per_collection=5,
        rrf_k=60,
    )
    retriever.chunks = chunks
    retriever.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    retriever.bm25 = BM25Index(chunks)
    retriever.collection = None
    retriever.vector_error = "offline model"
    item = SimpleNamespace(
        retrieval_id="RET-1",
        query_id="Q-1",
        retrieval_text="authentication",
        source_collections=("3gpp_markdown",),
    )

    result = retriever.retrieve(item)

    assert result.status == "partial"
    assert [hit.chunk_id for hit in result.hits] == ["C1"]
    assert result.hits[0].semantic_rank is None
    assert result.hits[0].lexical_rank == 1
    assert result.hits[0].lexical_score is not None
    assert any("semantic" in error and "offline model" in error for error in result.errors)
