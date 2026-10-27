from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from specrag.config import RetrievalCfg
from specrag.retrieval import corpus
from specrag.retrieval.clauses import (
    ClauseIndex,
    build_clause_records,
    split_clause_heading,
)
from specrag.retrieval.corpus import chunk_source_documents
from specrag.retrieval.index import load_clauses, write_snapshot
from specrag.retrieval.models import RetrievalIndexManifest, SourceDocument


def _document(
    root: Path, specification: str, text: str
) -> SourceDocument:
    relative = Path("data/3gpp") / specification / "v17.1.0" / "source.md"
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")
    return SourceDocument(
        source_collection="3gpp_markdown",
        specification=specification,
        version="17.1.0",
        source_path=relative.as_posix(),
        document_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    first = _document(
        tmp_path,
        "24.501",
        """# 5Parent procedure
The behavior is specified in clause 6.1 of TS 33.501 V17.1.0.

## 5.1First child
First child direct content line one.
First child direct content line two.
First child direct content line three.

## 5.2aLetter suffix child
Letter suffix content.

# AAnnex
## A.1Annex detail
Annex content.
""",
    )
    second = _document(
        tmp_path,
        "33.501",
        """# 6Security procedures
Security parent.

## 6.1Referenced procedure
Referenced direct content.
""",
    )
    cfg = RetrievalCfg(
        embedding_model="unused",
        chunk_characters=55,
        chunk_overlap_characters=0,
    )
    documents = (first, second)
    chunks = chunk_source_documents(documents, cfg)
    records = build_clause_records(documents, chunks)
    return cfg, documents, chunks, records


def test_clause_heading_parser_handles_letter_suffixes_without_stealing_titles() -> None:
    assert split_clause_heading("4.1General") == ("4.1", "General")
    assert split_clause_heading("4.2.5aRadio Capabilities") == (
        "4.2.5a", "Radio Capabilities",
    )
    assert split_clause_heading("19.4.2.9AePDG FQDN") == (
        "19.4.2.9A", "ePDG FQDN",
    )
    assert split_clause_heading("19.4.2.9A.1General") == (
        "19.4.2.9A.1", "General",
    )
    assert split_clause_heading("4.15.6.3b5G VN group data") == (
        "4.15.6.3b", "5G VN group data",
    )
    assert split_clause_heading("8.2.6.35GMM capability") == (
        "8.2.6.35", "GMM capability",
    )
    assert split_clause_heading("9.9.4.2APN aggregate maximum bit rate") == (
        "9.9.4.2", "APN aggregate maximum bit rate",
    )
    assert split_clause_heading("8.10.1.AUnsuccessful Operation") == (
        "8.10.1.A", "Unsuccessful Operation",
    )


def test_clause_build_disambiguates_joined_numeric_title_and_merges_true_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    document = _document(
        tmp_path,
        "24.501",
        """# 8Messages
## 8.2Mobility messages
### 8.2.6Registration request
#### 8.2.6.2Previous field
Previous content.
#### 8.2.6.35GMM capability
Capability content.
#### 8.2.6.4Next field
Next content.
#### 8.2.6.34Field before duplicate
Before duplicate.
#### 8.2.6.35Paging restriction
Paging content.
#### 8.2.6.35Service-level-AA container
Service content.
#### 8.2.6.36Field after duplicate
After duplicate.
""",
    )
    cfg = RetrievalCfg(
        embedding_model="unused",
        chunk_characters=80,
        chunk_overlap_characters=0,
    )
    chunks = chunk_source_documents((document,), cfg)

    records = build_clause_records((document,), chunks)
    by_id = {record.clause_id: record for record in records}

    assert by_id["8.2.6.3"].title == "5GMM capability"
    assert by_id["8.2.6.35"].title == (
        "Paging restriction / Service-level-AA container"
    )
    assert len(by_id["8.2.6.35"].direct_chunk_ids) == 2
    assert len(records) == len(by_id)


def test_clause_index_builds_numeric_suffix_annex_and_reference_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, chunks, records = _fixture(tmp_path, monkeypatch)
    index = ClauseIndex(records, chunks)

    assert [record.clause_id for record in records] == [
        "5", "5.1", "5.2a", "A.1", "6", "6.1",
    ]
    first_child = index.get("24.501", "17.1.0", "5.1")
    assert first_child is not None
    assert first_child.parent_clause_id == "5"
    assert len(first_child.direct_chunk_ids) > 1
    parent = index.get("24.501", "17.1.0", "5")
    assert parent is not None
    assert parent.child_clause_ids == ("5.1", "5.2a")
    assert set(parent.direct_chunk_ids) < set(parent.chunk_ids)
    assert index.get("24.501", "17.1.0", "A.1") is not None

    assert len(parent.cross_references) == 1
    reference = parent.cross_references[0]
    assert (reference.target_specification, reference.target_clause_id) == (
        "33.501", "6.1",
    )
    assert reference.target_version == "17.1.0"
    target = index.resolve_reference(
        parent,
        target_specification="33.501",
        target_clause_id="6.1",
        target_version="17.1.0",
    )
    assert target is not None
    assert (target.specification, target.version) == ("33.501", "17.1.0")
    assert index.resolve_reference(
        parent,
        target_specification="33.501",
        target_clause_id="6.2",
        target_version="17.1.0",
    ) is None


def test_clause_snapshot_round_trip_is_additive_to_chunk_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, documents, chunks, records = _fixture(tmp_path, monkeypatch)
    snapshot = tmp_path / "snapshot"
    manifest = RetrievalIndexManifest(
        index_fingerprint="FINGERPRINT",
        embedding_model=cfg.embedding_model,
        corpus_version_policy="highest_version_within_release",
        corpus_release=17,
        chunk_characters=cfg.chunk_characters,
        chunk_overlap_characters=cfg.chunk_overlap_characters,
        collection_name="specification_evidence",
        documents=documents,
        chunk_count=len(chunks),
        clause_count=len(records),
    )

    write_snapshot(snapshot, chunks, manifest, records)

    assert load_clauses(snapshot / "clauses.jsonl") == records
    persisted = RetrievalIndexManifest.model_validate_json(
        (snapshot / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted.schema_version == "specification-retrieval-index/v1.2"
    assert persisted.clause_count == len(records)
