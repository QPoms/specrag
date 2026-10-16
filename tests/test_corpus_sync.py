from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from specrag.models import (
    ActiveCorpusVersion,
    CorpusBuildResult,
    CorpusVersionRecord,
    SpecificationSourceCandidate,
)
from specrag.infrastructure.corpus_store import SQLiteCorpusStore
from specrag.infrastructure.database import SQLiteDatabase
from specrag.retrieval.corpus_sync import CorpusSyncService, active_snapshot
from specrag.retrieval.models import SourceDocument
from specrag.retrieval.semantic import dense_search
from specrag.config import RetrievalCfg


NOW = datetime.now(timezone.utc)


def _document(version: str, digest: str) -> SourceDocument:
    return SourceDocument(
        source_collection="3gpp_markdown",
        specification="24.501",
        version=version,
        source_path=f"data/3gpp/24.501/v{version}/24501.md",
        document_sha256=digest * 64,
    )


class _Store:
    def __init__(self) -> None:
        self.records: dict[str, CorpusVersionRecord] = {}
        self.active: dict[tuple[int, str], str] = {}

    def get(self, version_id):
        return self.records.get(version_id)

    def find(self, release, specification, version):
        return next((
            item for item in self.records.values()
            if (item.release, item.specification, item.version)
            == (release, specification, version)
        ), None)

    def list_versions(self, release=None):
        return tuple(
            item for item in self.records.values()
            if release is None or item.release == release
        )

    def put(self, record):
        self.records[record.version_id] = record
        return record

    def activate(self, version_id):
        record = self.records[version_id]
        key = (record.release, record.specification)
        previous_id = self.active.get(key)
        if previous_id and previous_id != version_id:
            previous = self.records[previous_id]
            self.records[previous_id] = previous.model_copy(
                update={"status": "draining"}
            )
        active = record.model_copy(update={
            "status": "active",
            "activated_at": NOW,
        })
        self.records[version_id] = active
        self.active[key] = version_id
        return active

    def active_versions(self, release):
        return tuple(
            ActiveCorpusVersion(
                release=record.release,
                specification=record.specification,
                version_id=record.version_id,
                version=record.version,
            )
            for key, version_id in sorted(self.active.items())
            if key[0] == release
            for record in (self.records[version_id],)
        )


def _built(document: SourceDocument) -> CorpusBuildResult:
    return CorpusBuildResult(
        index_fingerprint=f"INDEX-{document.version}",
        chunk_count=3,
        lexical_record_count=3,
        vector_record_count=3,
    )


def test_sync_registers_history_but_builds_only_latest_document() -> None:
    store = _Store()
    built: list[str] = []

    def builder(document):
        built.append(document.version)
        return _built(document)

    result = CorpusSyncService(store, builder).synchronize(
        (_document("17.17.0", "A"), _document("17.18.0", "B")),
        apply=True,
    )

    assert built == ["17.18.0"]
    assert [item.version for item in store.list_versions()] == ["17.17.0", "17.18.0"]
    assert [item.action for item in result] == ["new", "activated"]
    assert active_snapshot(store, 17)[0].version == "17.18.0"


def test_new_activation_keeps_old_version_as_draining() -> None:
    store = _Store()
    service = CorpusSyncService(store, _built)
    service.synchronize((_document("17.17.0", "A"),), apply=True)
    service.synchronize(
        (_document("17.17.0", "A"), _document("17.18.0", "B")),
        apply=True,
    )

    old = store.find(17, "24.501", "17.17.0")
    new = store.find(17, "24.501", "17.18.0")
    assert old is not None and old.status == "draining"
    assert new is not None and new.status == "active"
    assert len(store.records) == 2


def test_failed_incremental_build_does_not_displace_active_version() -> None:
    store = _Store()
    CorpusSyncService(store, _built).synchronize(
        (_document("17.17.0", "A"),), apply=True
    )

    def fail(_document):
        raise RuntimeError("embedding unavailable")

    result = CorpusSyncService(store, fail).synchronize(
        (_document("17.17.0", "A"), _document("17.18.0", "B")),
        apply=True,
    )

    assert result[-1].status == "failed"
    assert active_snapshot(store, 17)[0].version == "17.17.0"
    failed = store.find(17, "24.501", "17.18.0")
    assert failed is not None and failed.status == "failed"


def test_dry_run_has_no_persistent_or_build_side_effects() -> None:
    store = _Store()
    service = CorpusSyncService(
        store,
        lambda _document: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = service.synchronize((_document("17.18.0", "B"),))

    assert result[0].message == "would build, validate, and activate"
    assert store.records == {}


def test_sqlite_activation_is_atomic_and_retains_previous_record(
    tmp_path: Path,
) -> None:
    database = SQLiteDatabase(tmp_path / "corpus-activation.sqlite3")
    store = SQLiteCorpusStore(database)
    try:
        old = CorpusVersionRecord(
            version_id="CV-OLD",
            specification="24.501",
            version="17.17.0",
            release=17,
            source_path="data/3gpp/24.501/v17.17.0/24501.md",
            document_sha256="A" * 64,
            status="ready",
            discovered_at=NOW,
            indexed_at=NOW,
            index_fingerprint="INDEX-OLD",
            chunk_count=2,
        )
        new = old.model_copy(update={
            "version_id": "CV-NEW",
            "version": "17.18.0",
            "source_path": "data/3gpp/24.501/v17.18.0/24501.md",
            "document_sha256": "B" * 64,
            "index_fingerprint": "INDEX-NEW",
        })
        store.put(old)
        store.activate(old.version_id)
        store.put(new)
        store.activate(new.version_id)

        assert store.get(old.version_id).status == "draining"
        assert store.get(new.version_id).status == "active"
        assert [item.version for item in store.list_versions(17)] == [
            "17.17.0", "17.18.0"
        ]
    finally:
        database.close()


def test_provider_discovery_is_side_effect_free_until_apply() -> None:
    store = _Store()
    materialized: list[str] = []

    class Provider:
        def discover(self, releases):
            assert releases == (17, 18, 19)
            return (SpecificationSourceCandidate(
                specification="24.501",
                version="17.18.0",
                release=17,
                source_uri="https://example.invalid/24501.zip",
                target_source_path="data/3gpp/24.501/v17.18.0/24501.md",
                source_format="zip",
            ),)

    class Materializer:
        def materialize(self, candidate):
            materialized.append(candidate.version)
            return _document(candidate.version, "B")

    service = CorpusSyncService(store, _built)
    preview = service.synchronize_from_provider(
        Provider(), Materializer(), releases=(17, 18, 19)
    )
    assert preview[0].message == (
        "would download, convert, build, validate, and activate"
    )
    assert materialized == []
    assert store.records == {}

    applied = service.synchronize_from_provider(
        Provider(), Materializer(), releases=(17, 18, 19), apply=True
    )
    assert materialized == ["17.18.0"]
    assert applied[0].action == "activated"


def test_dense_search_filters_append_only_cache_by_snapshot_hash() -> None:
    class Collection:
        where = None

        def query(self, **kwargs):
            self.where = kwargs["where"]
            return {"ids": [["CH-NEW"]], "distances": [[0.1]]}

    collection = Collection()
    result = dense_search(
        collection,
        RetrievalCfg(),
        "registration",
        "3gpp_markdown",
        ("24.501",),
        ("B" * 64,),
    )

    assert result == [("CH-NEW", 0.1)]
    assert {"document_sha256": {"$in": ["B" * 64]}} in collection.where["$and"]
