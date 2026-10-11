from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrag.config import AppConfig, RetrievalCfg
from specrag.retrieval import corpus
from specrag.retrieval import planner
from specrag.retrieval.hybrid import (
    CorpusScope,
    HybridMarkdownRetriever,
    MarkdownChunk,
    SourceDocument,
    SpecificationVersion,
    chunk_source_documents,
    discover_source_documents,
)
from specrag.retrieval.lexical import BM25Index
from specrag.retrieval.index import index_fingerprint
from specrag.retrieval.execution_models import PlannedRetrievalExecution


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


def test_default_scope_selects_release_17_uniformly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_24 = _write_specification(tmp_path, "24.501", "17.9.0", "R17")
    _write_specification(tmp_path, "24.501", "18.2.0", "R18")
    selected_33 = _write_specification(tmp_path, "33.501", "17.4.0", "R17")
    _write_specification(tmp_path, "33.501", "16.4.0", "R16")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)

    documents = discover_source_documents(_cfg())

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "17.9.0"),
        ("33.501", "17.4.0"),
    ]
    assert {item.version.split(".", 1)[0] for item in documents} == {"17"}
    assert {item.source_path for item in documents} == {
        selected_24.relative_to(tmp_path).as_posix(),
        selected_33.relative_to(tmp_path).as_posix(),
    }
    assert documents[0].document_sha256 == hashlib.sha256(
        selected_24.read_bytes()
    ).hexdigest().upper()
    assert documents[1].document_sha256 == hashlib.sha256(
        selected_33.read_bytes()
    ).hexdigest().upper()


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


def test_explicit_release_scope_applies_one_release_to_every_specification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_specification(tmp_path, "24.501", "16.8.0", "NAS release 16")
    _write_specification(tmp_path, "24.501", "17.9.0", "NAS release 17")
    _write_specification(tmp_path, "33.501", "16.5.0", "security release 16")
    _write_specification(tmp_path, "33.501", "17.6.0", "security release 17")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)

    documents = discover_source_documents(
        _cfg(),
        CorpusScope(
            version_policy="highest_version_within_release",
            release=16,
        ),
    )

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "16.8.0"),
        ("33.501", "16.5.0"),
    ]
    assert {item.version.split(".", 1)[0] for item in documents} == {"16"}


def test_latest_scope_selects_latest_version_per_specification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_specification(tmp_path, "24.501", "17.9.0", "NAS R17")
    _write_specification(tmp_path, "24.501", "18.3.0", "NAS R18")
    _write_specification(tmp_path, "33.501", "16.4.0", "Security R16")
    _write_specification(tmp_path, "33.501", "17.8.0", "Security R17")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)

    documents = discover_source_documents(
        _cfg(),
        CorpusScope(version_policy="latest_per_spec"),
    )

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "18.3.0"),
        ("33.501", "17.8.0"),
    ]


def test_exact_scope_selects_only_requested_specification_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_specification(tmp_path, "24.501", "17.1.0", "selected")
    _write_specification(tmp_path, "24.501", "17.9.0", "not selected")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    selected = SpecificationVersion(
        specification="24.501",
        version="17.1.0",
    )

    documents = discover_source_documents(
        _cfg(),
        CorpusScope(version_policy="exact_version", versions=(selected,)),
    )

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "17.1.0"),
    ]


def test_mixed_scope_selects_each_explicit_specification_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_specification(tmp_path, "24.501", "16.8.0", "NAS R16")
    _write_specification(tmp_path, "24.501", "17.9.0", "NAS R17")
    _write_specification(tmp_path, "33.501", "17.6.0", "Security R17")
    _write_specification(tmp_path, "33.501", "18.2.0", "Security R18")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)

    documents = discover_source_documents(
        _cfg(),
        CorpusScope(
            version_policy="mixed_versions",
            versions=(
                SpecificationVersion(
                    specification="24.501",
                    version="16.8.0",
                ),
                SpecificationVersion(
                    specification="33.501",
                    version="18.2.0",
                ),
            ),
        ),
    )

    assert [(item.specification, item.version) for item in documents] == [
        ("24.501", "16.8.0"),
        ("33.501", "18.2.0"),
    ]


def test_explicit_scope_fails_when_requested_version_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_specification(tmp_path, "24.501", "17.1.0", "available")
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    scope = CorpusScope(
        version_policy="exact_version",
        versions=(SpecificationVersion(
            specification="24.501",
            version="18.1.0",
        ),),
    )

    with pytest.raises(ValueError, match=r"24\.501 v18\.1\.0"):
        discover_source_documents(_cfg(), scope)


def test_release_scope_requires_release_number() -> None:
    with pytest.raises(ValueError, match="corpus_release is required"):
        CorpusScope(version_policy="highest_version_within_release")


def test_release_scope_is_part_of_index_snapshot_identity() -> None:
    documents = (
        SourceDocument(
            source_collection="3gpp_markdown",
            specification="24.501",
            version="17.1.0",
            source_path="data/3gpp/24.501/v17.1.0/source.md",
            document_sha256="A" * 64,
        ),
    )
    release_16 = CorpusScope(
        version_policy="highest_version_within_release",
        release=16,
    )
    release_17 = release_16.model_copy(update={"release": 17})

    assert index_fingerprint(documents, _cfg(), release_16) != index_fingerprint(
        documents,
        _cfg(),
        release_17,
    )


def test_release_comparison_runs_two_isolated_uniform_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_scopes: list[CorpusScope] = []

    def fake_execute(*args: object, **kwargs: object) -> PlannedRetrievalExecution:
        scope = kwargs["corpus_scope"]
        assert isinstance(scope, CorpusScope)
        observed_scopes.append(scope)
        return PlannedRetrievalExecution(
            status="completed",
            embedding_model="test",
            corpus_version_policy=scope.version_policy,
            corpus_release=scope.release,
        )

    monkeypatch.setattr(
        planner,
        "execute_planned_discovery_retrieval",
        fake_execute,
    )
    result = planner.execute_release_comparison(
        SimpleNamespace(),
        base_release=16,
        target_release=17,
        app_config=AppConfig(retrieval=_cfg()),
    )

    assert observed_scopes == [
        CorpusScope(
            version_policy="highest_version_within_release",
            release=16,
        ),
        CorpusScope(
            version_policy="highest_version_within_release",
            release=17,
        ),
    ]
    assert result.status == "completed"
    assert result.base_execution.corpus_release == 16
    assert result.target_execution.corpus_release == 17


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
