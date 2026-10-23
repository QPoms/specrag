"""SQLite persistence for immutable Research Results."""

from __future__ import annotations

import json
import sqlite3

from specrag.models import ResearchResult

from .database import SQLiteDatabase


def _json(result: ResearchResult) -> str:
    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class SQLiteResearchResultStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def put(self, result: ResearchResult) -> ResearchResult:
        try:
            with self.database.connection:
                self.insert(self.database.connection, result)
            return result
        except sqlite3.IntegrityError:
            existing = self.get_by_run_attempt(result.run_id, result.run_attempt)
            if existing is None or existing.content_sha256 != result.content_sha256:
                raise ValueError("Run attempt Result content conflict")
            return existing

    @staticmethod
    def insert(
        connection: sqlite3.Connection, result: ResearchResult
    ) -> ResearchResult:
        """Insert inside a caller-owned transaction, validating replay."""

        try:
            connection.execute(
                "INSERT INTO research_results(result_id, run_id, run_attempt, "
                "revision, workspace_id, owner_user_id, conversation_id, "
                "publication_kind, evidence_status, content_sha256, payload_json, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.result_id,
                    result.run_id,
                    result.run_attempt,
                    result.revision,
                    result.workspace_id,
                    result.owner_user_id,
                    result.conversation_id,
                    result.publication_kind,
                    result.evidence_status,
                    result.content_sha256,
                    _json(result),
                    result.created_at.isoformat(),
                ),
            )
            return result
        except sqlite3.IntegrityError:
            row = connection.execute(
                "SELECT payload_json FROM research_results WHERE run_id=? "
                "AND run_attempt=?",
                (result.run_id, result.run_attempt),
            ).fetchone()
            existing = (
                ResearchResult.model_validate_json(row[0]) if row else None
            )
            if existing is None or existing.content_sha256 != result.content_sha256:
                raise ValueError("Run attempt Result content conflict")
            return existing

    def get(self, result_id: str) -> ResearchResult | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM research_results WHERE result_id=?",
            (result_id,),
        ).fetchone()
        return ResearchResult.model_validate_json(row[0]) if row else None

    def get_by_run_attempt(
        self, run_id: str, run_attempt: int
    ) -> ResearchResult | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM research_results WHERE run_id=? "
            "AND run_attempt=?",
            (run_id, run_attempt),
        ).fetchone()
        return ResearchResult.model_validate_json(row[0]) if row else None

    def latest_for_run(self, run_id: str) -> ResearchResult | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM research_results WHERE run_id=? "
            "ORDER BY run_attempt DESC, revision DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return ResearchResult.model_validate_json(row[0]) if row else None

    def latest_before_attempt(
        self, run_id: str, run_attempt: int
    ) -> ResearchResult | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM research_results WHERE run_id=? "
            "AND run_attempt<? ORDER BY run_attempt DESC, revision DESC LIMIT 1",
            (run_id, run_attempt),
        ).fetchone()
        return ResearchResult.model_validate_json(row[0]) if row else None

    def list_for_conversation(
        self, conversation_id: str, *, limit: int = 100
    ) -> tuple[ResearchResult, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM research_results WHERE conversation_id=? "
            "ORDER BY created_at, result_id LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return tuple(ResearchResult.model_validate_json(row[0]) for row in rows)


__all__ = ["SQLiteResearchResultStore"]
