"""SQLite persistence for immutable specification versions and active pointers."""

from __future__ import annotations

from datetime import datetime, timezone

from specrag.models import ActiveCorpusVersion, CorpusVersionRecord

from .database import SQLiteDatabase


class SQLiteCorpusStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def get(self, version_id: str) -> CorpusVersionRecord | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM corpus_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        return CorpusVersionRecord.model_validate_json(row[0]) if row else None

    def find(
        self,
        release: int,
        specification: str,
        version: str,
    ) -> CorpusVersionRecord | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM corpus_versions "
            "WHERE release = ? AND specification = ? AND version = ?",
            (release, specification, version),
        ).fetchone()
        return CorpusVersionRecord.model_validate_json(row[0]) if row else None

    def list_versions(
        self,
        release: int | None = None,
    ) -> tuple[CorpusVersionRecord, ...]:
        if release is None:
            rows = self.database.connection.execute(
                "SELECT payload_json FROM corpus_versions "
                "ORDER BY release, specification, version"
            ).fetchall()
        else:
            rows = self.database.connection.execute(
                "SELECT payload_json FROM corpus_versions WHERE release = ? "
                "ORDER BY specification, version",
                (release,),
            ).fetchall()
        return tuple(CorpusVersionRecord.model_validate_json(row[0]) for row in rows)

    def put(self, record: CorpusVersionRecord) -> CorpusVersionRecord:
        existing = self.find(record.release, record.specification, record.version)
        if existing is not None and existing.version_id != record.version_id:
            raise ValueError(
                "the same specification version was discovered with different content"
            )
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO corpus_versions(version_id, release, specification, version, "
                "status, payload_json) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(version_id) DO UPDATE SET status=excluded.status, "
                "payload_json=excluded.payload_json",
                (
                    record.version_id,
                    record.release,
                    record.specification,
                    record.version,
                    record.status,
                    record.model_dump_json(),
                ),
            )
        return record

    def activate(self, version_id: str) -> CorpusVersionRecord:
        record = self.get(version_id)
        if record is None:
            raise KeyError(f"unknown corpus version: {version_id}")
        if record.status != "ready":
            raise ValueError("only a ready corpus version can be activated")
        now = datetime.now(timezone.utc)
        active = record.model_copy(update={"status": "active", "activated_at": now})
        with self.database.connection:
            previous_row = self.database.connection.execute(
                "SELECT version_id FROM active_corpus_versions "
                "WHERE release = ? AND specification = ?",
                (record.release, record.specification),
            ).fetchone()
            if previous_row is not None and previous_row[0] != version_id:
                previous = self.get(previous_row[0])
                if previous is not None:
                    draining = previous.model_copy(update={"status": "draining"})
                    self.database.connection.execute(
                        "UPDATE corpus_versions SET status = ?, payload_json = ? "
                        "WHERE version_id = ?",
                        (draining.status, draining.model_dump_json(), draining.version_id),
                    )
            self.database.connection.execute(
                "UPDATE corpus_versions SET status = ?, payload_json = ? WHERE version_id = ?",
                (active.status, active.model_dump_json(), active.version_id),
            )
            self.database.connection.execute(
                "INSERT INTO active_corpus_versions(release, specification, version_id) "
                "VALUES (?, ?, ?) ON CONFLICT(release, specification) DO UPDATE "
                "SET version_id=excluded.version_id",
                (active.release, active.specification, active.version_id),
            )
        return active

    def active_versions(self, release: int) -> tuple[ActiveCorpusVersion, ...]:
        rows = self.database.connection.execute(
            "SELECT c.release, c.specification, c.version_id, c.version "
            "FROM active_corpus_versions a JOIN corpus_versions c "
            "ON c.version_id = a.version_id WHERE c.release = ? "
            "ORDER BY c.specification",
            (release,),
        ).fetchall()
        return tuple(ActiveCorpusVersion(**dict(row)) for row in rows)

    def bind_run(self, run_id: str, version_ids: tuple[str, ...]) -> None:
        with self.database.connection:
            for version_id in version_ids:
                self.database.connection.execute(
                    "INSERT OR IGNORE INTO run_corpus_bindings(run_id, version_id) "
                    "VALUES (?, ?)",
                    (run_id, version_id),
                )

    def referencing_runs(self, version_id: str) -> tuple[str, ...]:
        rows = self.database.connection.execute(
            "SELECT run_id FROM run_corpus_bindings WHERE version_id = ? ORDER BY run_id",
            (version_id,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)


__all__ = ["SQLiteCorpusStore"]
