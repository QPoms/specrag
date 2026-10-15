"""Explicit, non-destructive document-level corpus synchronization."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
from typing import Protocol

from specrag.models import (
    ActiveCorpusVersion,
    CorpusBuildResult,
    CorpusSyncItem,
    CorpusVersionRecord,
    CorpusVersionRef,
    SpecificationSourceCandidate,
)

from .models import SourceDocument


class CorpusVersionStore(Protocol):
    def find(
        self, release: int, specification: str, version: str
    ) -> CorpusVersionRecord | None: ...
    def get(self, version_id: str) -> CorpusVersionRecord | None: ...
    def list_versions(
        self, release: int | None = None
    ) -> tuple[CorpusVersionRecord, ...]: ...
    def put(self, record: CorpusVersionRecord) -> CorpusVersionRecord: ...
    def activate(self, version_id: str) -> CorpusVersionRecord: ...
    def active_versions(self, release: int) -> tuple[ActiveCorpusVersion, ...]: ...


DocumentBuilder = Callable[[SourceDocument], CorpusBuildResult]


class SpecificationVersionProvider(Protocol):
    """Discover provider-owned current versions without mutating the corpus."""

    def discover(
        self, releases: tuple[int, ...]
    ) -> tuple[SpecificationSourceCandidate, ...]: ...


class SpecificationSourceMaterializer(Protocol):
    """Download and convert one candidate into the canonical local Markdown."""

    def materialize(self, candidate: SpecificationSourceCandidate) -> SourceDocument: ...


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(value) for value in version.split("."))  # type: ignore[return-value]


def _version_id(document: SourceDocument) -> str:
    assert document.specification is not None and document.version is not None
    identity = "|".join((
        document.specification,
        document.version,
        document.source_path,
        document.document_sha256,
    ))
    return "CV-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24].upper()


class CorpusSyncService:
    """Register history and build only the newest changed document per spec."""

    def __init__(self, store: CorpusVersionStore, builder: DocumentBuilder) -> None:
        self.store = store
        self.builder = builder

    def synchronize(
        self,
        documents: tuple[SourceDocument, ...],
        *,
        apply: bool = False,
    ) -> tuple[CorpusSyncItem, ...]:
        ordered = tuple(sorted(
            documents,
            key=lambda item: (
                item.release or 0,
                item.specification or "",
                _version_key(item.version or "0.0.0"),
            ),
        ))
        latest: dict[tuple[int, str], SourceDocument] = {}
        for document in ordered:
            assert document.release is not None
            assert document.specification is not None
            key = (document.release, document.specification)
            current = latest.get(key)
            if current is None or _version_key(
                document.version or "0.0.0"
            ) > _version_key(current.version or "0.0.0"):
                latest[key] = document

        active_by_key = {
            (item.release, item.specification): item
            for release in sorted({
                item.release for item in ordered if item.release is not None
            })
            for item in self.store.active_versions(int(release))
        }
        results: list[CorpusSyncItem] = []
        for document in ordered:
            assert document.release is not None
            assert document.specification is not None
            assert document.version is not None
            existing = self.store.find(
                document.release, document.specification, document.version
            )
            if existing is not None and existing.document_sha256 != document.document_sha256:
                results.append(CorpusSyncItem(
                    specification=document.specification,
                    version=document.version,
                    source_path=document.source_path,
                    action="changed",
                    status=existing.status,
                    message="same version has different source content; refusing overwrite",
                ))
                continue

            record = existing or CorpusVersionRecord(
                version_id=_version_id(document),
                specification=document.specification,
                version=document.version,
                release=document.release,
                source_path=document.source_path,
                document_sha256=document.document_sha256,
                discovered_at=datetime.now(timezone.utc),
            )
            if apply and existing is None:
                self.store.put(record)

            key = (document.release, document.specification)
            active = active_by_key.get(key)
            is_latest = latest[key] == document
            needs_activation = is_latest and (
                active is None or _version_key(document.version) > _version_key(active.version)
            )
            if not needs_activation:
                results.append(CorpusSyncItem(
                    specification=document.specification,
                    version=document.version,
                    source_path=document.source_path,
                    action="new" if existing is None else "unchanged",
                    status=record.status,
                ))
                continue
            if not apply:
                results.append(CorpusSyncItem(
                    specification=document.specification,
                    version=document.version,
                    source_path=document.source_path,
                    action="new" if existing is None else "changed",
                    status=record.status,
                    message="would build, validate, and activate",
                ))
                continue

            indexing = record.model_copy(update={
                "status": "indexing",
                "failure_reason": None,
            })
            self.store.put(indexing)
            try:
                built = self.builder(document)
            except Exception as exc:
                failed = indexing.model_copy(update={
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })
                self.store.put(failed)
                results.append(CorpusSyncItem(
                    specification=document.specification,
                    version=document.version,
                    source_path=document.source_path,
                    action="failed",
                    status="failed",
                    message=failed.failure_reason,
                ))
                continue
            ready = indexing.model_copy(update={
                "status": "ready",
                "indexed_at": datetime.now(timezone.utc),
                "index_fingerprint": built.index_fingerprint,
                "chunk_count": built.chunk_count,
            })
            self.store.put(ready)
            activated = self.store.activate(ready.version_id)
            active_by_key[key] = ActiveCorpusVersion(
                release=activated.release,
                specification=activated.specification,
                version_id=activated.version_id,
                version=activated.version,
            )
            results.append(CorpusSyncItem(
                specification=document.specification,
                version=document.version,
                source_path=document.source_path,
                action="activated",
                status="active",
            ))
        return tuple(results)

    def synchronize_from_provider(
        self,
        provider: SpecificationVersionProvider,
        materializer: SpecificationSourceMaterializer,
        *,
        releases: tuple[int, ...],
        apply: bool = False,
    ) -> tuple[CorpusSyncItem, ...]:
        """Discover remotely, then materialize only candidates needing activation."""

        candidates = tuple(sorted(
            provider.discover(releases),
            key=lambda item: (item.release, item.specification, _version_key(item.version)),
        ))
        active = {
            (item.release, item.specification): item
            for release in releases
            for item in self.store.active_versions(release)
        }
        pending: list[SpecificationSourceCandidate] = []
        unchanged: list[CorpusSyncItem] = []
        for candidate in candidates:
            current = active.get((candidate.release, candidate.specification))
            if current is not None and _version_key(
                current.version
            ) >= _version_key(candidate.version):
                record = self.store.get(current.version_id)
                unchanged.append(CorpusSyncItem(
                    specification=candidate.specification,
                    version=candidate.version,
                    source_path=candidate.target_source_path,
                    action="unchanged",
                    status=record.status if record is not None else "active",
                ))
            else:
                pending.append(candidate)
        if not apply:
            return tuple((*unchanged, *(
                CorpusSyncItem(
                    specification=item.specification,
                    version=item.version,
                    source_path=item.target_source_path,
                    action="new",
                    status="discovered",
                    message="would download, convert, build, validate, and activate",
                )
                for item in pending
            )))
        documents = tuple(materializer.materialize(item) for item in pending)
        return (*unchanged, *self.synchronize(documents, apply=True))


def active_snapshot(
    store: CorpusVersionStore,
    release: int,
) -> tuple[CorpusVersionRef, ...]:
    """Resolve the exact active versions for a new immutable Run binding."""

    active = store.active_versions(release)
    return tuple(
        CorpusVersionRef(
            specification=item.specification,
            version=item.version,
            release=item.release,
        )
        for item in active
    )


__all__ = [
    "CorpusSyncService",
    "CorpusVersionStore",
    "SpecificationSourceMaterializer",
    "SpecificationVersionProvider",
    "active_snapshot",
]
