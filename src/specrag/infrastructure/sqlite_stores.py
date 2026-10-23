"""SQLite adapters for runs, events, SpecFacts, and Core/Topic Memory."""

from __future__ import annotations

import json
import hashlib
import sqlite3

from specrag.models import (
    MemoryEntry,
    MemoryMaintenanceState,
    MemoryProposal,
    MemoryTopic,
    ResearchRun,
    ResearchOutcome,
    RunEvent,
    SpecFact,
    SpecFactProposal,
)

from .database import SQLiteDatabase


def _json(model) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteRunStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def put(self, run: ResearchRun) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO runs(run_id, thread_id, workspace_id, owner_user_id, status, "
                "cancel_requested_at, payload_json, conversation_id, origin_message_id, "
                "origin_turn_id, submitted_user_sequence, submission_operation_key, "
                "run_attempt, input_snapshot_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
                "cancel_requested_at=excluded.cancel_requested_at, "
                "conversation_id=excluded.conversation_id, "
                "origin_message_id=excluded.origin_message_id, "
                "origin_turn_id=excluded.origin_turn_id, "
                "submitted_user_sequence=excluded.submitted_user_sequence, "
                "submission_operation_key=excluded.submission_operation_key, "
                "run_attempt=excluded.run_attempt, "
                "input_snapshot_json=excluded.input_snapshot_json, "
                "payload_json=excluded.payload_json",
                (run.run_id, run.thread_id, run.workspace_id, run.owner_user_id,
                 run.status,
                 run.cancel_requested_at.isoformat() if run.cancel_requested_at else None,
                 _json(run), run.conversation_id, run.origin_message_id,
                 run.origin_turn_id, run.submitted_user_sequence,
                 run.submission_operation_key, run.run_attempt,
                 _json(run.input_snapshot) if run.input_snapshot else None),
            )

    def get(self, run_id: str) -> ResearchRun | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return ResearchRun.model_validate_json(row[0]) if row else None

    def get_by_submission(
        self, conversation_id: str, operation_key: str,
    ) -> ResearchRun | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM runs WHERE conversation_id=? "
            "AND submission_operation_key=?",
            (conversation_id, operation_key),
        ).fetchone()
        return ResearchRun.model_validate_json(row[0]) if row else None

    def list_runs(
        self, workspace_id: str, *, owner_user_id: str | None, limit: int,
    ) -> tuple[ResearchRun, ...]:
        if owner_user_id is None:
            rows = self.database.connection.execute(
                "SELECT payload_json FROM runs WHERE workspace_id = ? "
                "ORDER BY rowid DESC LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
        else:
            rows = self.database.connection.execute(
                "SELECT payload_json FROM runs WHERE workspace_id = ? "
                "AND owner_user_id = ? ORDER BY rowid DESC LIMIT ?",
                (workspace_id, owner_user_id, limit),
            ).fetchall()
        return tuple(ResearchRun.model_validate_json(row[0]) for row in rows)

    def list_active_runs(self) -> tuple[ResearchRun, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM runs WHERE status IN ('queued', 'running') "
            "ORDER BY rowid"
        ).fetchall()
        return tuple(ResearchRun.model_validate_json(row[0]) for row in rows)

    def is_cancel_requested(self, run_id: str) -> bool:
        row = self.database.connection.execute(
            "SELECT cancel_requested_at FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return bool(row and row[0])

    def append_event(self, event: RunEvent) -> RunEvent:
        try:
            with self.database.connection:
                self.database.connection.execute(
                    "INSERT INTO run_events(event_id, run_id, workspace_id, sequence, "
                    "operation_key, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (event.event_id, event.run_id, event.workspace_id, event.sequence,
                     event.operation_key, _json(event)),
                )
            return event
        except sqlite3.IntegrityError:
            if event.operation_key is None:
                raise
            row = self.database.connection.execute(
                "SELECT payload_json FROM run_events WHERE run_id = ? AND operation_key = ?",
                (event.run_id, event.operation_key),
            ).fetchone()
            if row is None:
                raise
            return RunEvent.model_validate_json(row[0])

    def finalize(
        self,
        run: ResearchRun,
        event: RunEvent,
        outcome: ResearchOutcome | None,
    ) -> RunEvent:
        """Atomically publish one terminal attempt and its completion outbox."""

        if run.status not in {"completed", "failed", "cancelled", "timed_out"}:
            raise ValueError("only a terminal Run can be finalized")
        if event.event_type != "run_stopped" or event.run_id != run.run_id:
            raise ValueError("terminal Run requires its run_stopped event")
        if outcome is not None and (
            outcome.run_id != run.run_id or outcome.run_attempt != run.run_attempt
        ):
            raise ValueError("Research Outcome does not match the Run attempt")
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            updated = connection.execute(
                "UPDATE runs SET status=?, cancel_requested_at=?, "
                "conversation_id=?, origin_message_id=?, origin_turn_id=?, "
                "submitted_user_sequence=?, submission_operation_key=?, run_attempt=?, "
                "input_snapshot_json=?, payload_json=? WHERE run_id=?",
                (
                    run.status,
                    run.cancel_requested_at.isoformat()
                    if run.cancel_requested_at else None,
                    run.conversation_id,
                    run.origin_message_id,
                    run.origin_turn_id,
                    run.submitted_user_sequence,
                    run.submission_operation_key,
                    run.run_attempt,
                    _json(run.input_snapshot) if run.input_snapshot else None,
                    _json(run),
                    run.run_id,
                ),
            )
            if updated.rowcount != 1:
                raise KeyError(run.run_id)
            try:
                connection.execute(
                    "INSERT INTO run_events(event_id, run_id, workspace_id, sequence, "
                    "operation_key, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.run_id,
                        event.workspace_id,
                        event.sequence,
                        event.operation_key,
                        _json(event),
                    ),
                )
                stored_event = event
            except sqlite3.IntegrityError:
                row = connection.execute(
                    "SELECT payload_json FROM run_events WHERE run_id=? "
                    "AND operation_key=?",
                    (event.run_id, event.operation_key),
                ).fetchone()
                if row is None:
                    raise
                stored_event = RunEvent.model_validate_json(row[0])
            if outcome is not None:
                payload = _json(outcome)
                digest = _sha256(payload)
                connection.execute(
                    "INSERT OR IGNORE INTO run_outcomes(run_id, run_attempt, "
                    "content_sha256, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        outcome.run_id,
                        outcome.run_attempt,
                        digest,
                        payload,
                        outcome.created_at.isoformat(),
                    ),
                )
                existing = connection.execute(
                    "SELECT content_sha256 FROM run_outcomes WHERE run_id=? "
                    "AND run_attempt=?",
                    (outcome.run_id, outcome.run_attempt),
                ).fetchone()
                if existing is None or existing[0] != digest:
                    raise ValueError("Run attempt Outcome content conflict")
            now = event.occurred_at.isoformat()
            connection.execute(
                "INSERT OR IGNORE INTO run_completion_jobs(run_id, run_attempt, "
                "state, available_at, lease_owner, lease_expires_at, attempts, "
                "created_at, updated_at) SELECT ?, ?, 'queued', ?, NULL, NULL, 0, ?, ? "
                "WHERE NOT EXISTS (SELECT 1 FROM run_completion_receipts "
                "WHERE run_id=? AND run_attempt=?)",
                (
                    run.run_id,
                    run.run_attempt,
                    now,
                    now,
                    now,
                    run.run_id,
                    run.run_attempt,
                ),
            )
            connection.commit()
            return stored_event
        except BaseException:
            connection.rollback()
            raise

    def get_outcome(
        self, run_id: str, run_attempt: int,
    ) -> ResearchOutcome | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM run_outcomes WHERE run_id=? AND run_attempt=?",
            (run_id, run_attempt),
        ).fetchone()
        return ResearchOutcome.model_validate_json(row[0]) if row else None

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM run_events WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        return tuple(RunEvent.model_validate_json(row[0]) for row in rows)


class SQLiteSpecFactStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def all_proposals(self) -> tuple[SpecFactProposal, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM specfact_proposals ORDER BY proposal_id"
        ).fetchall()
        return tuple(SpecFactProposal.model_validate_json(row[0]) for row in rows)

    def put_proposal(self, proposal: SpecFactProposal) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO specfact_proposals(proposal_id, operation_key, workspace_id, status, payload_json) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(proposal_id) DO UPDATE SET "
                "status=excluded.status, payload_json=excluded.payload_json",
                (proposal.proposal_id, proposal.operation_key, proposal.workspace_id,
                 proposal.status, _json(proposal)),
            )

    def all_facts(self) -> tuple[SpecFact, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM specfacts ORDER BY fact_id"
        ).fetchall()
        return tuple(SpecFact.model_validate_json(row[0]) for row in rows)

    def put_fact(self, fact: SpecFact) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO specfacts(fact_id, proposal_id, workspace_id, status, payload_json) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET "
                "status=excluded.status, payload_json=excluded.payload_json",
                (fact.fact_id, fact.proposal_id, fact.workspace_id, fact.status, _json(fact)),
            )


class SQLiteMemoryStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def all_entries(self) -> tuple[MemoryEntry, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM memory_entries ORDER BY memory_id"
        ).fetchall()
        return tuple(MemoryEntry.model_validate_json(row[0]) for row in rows)

    def put_entry(self, item: MemoryEntry) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO memory_entries(memory_id, scope, owner_user_id, workspace_id, "
                "topic_id, status, operation_key, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET status=excluded.status, "
                "operation_key=excluded.operation_key, payload_json=excluded.payload_json",
                (
                    item.memory_id,
                    item.scope,
                    item.owner_user_id,
                    item.workspace_id,
                    item.topic_id,
                    item.status,
                    item.operation_key,
                    _json(item),
                ),
            )

    def all_topics(self) -> tuple[MemoryTopic, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM memory_topics ORDER BY topic_id"
        ).fetchall()
        return tuple(MemoryTopic.model_validate_json(row[0]) for row in rows)

    def put_topic(self, topic: MemoryTopic) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO memory_topics(topic_id, workspace_id, slug, status, pinned, "
                "payload_json) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(topic_id) DO UPDATE SET "
                "status=excluded.status, pinned=excluded.pinned, payload_json=excluded.payload_json",
                (
                    topic.topic_id,
                    topic.workspace_id,
                    topic.slug,
                    topic.status,
                    int(topic.pinned),
                    _json(topic),
                ),
            )

    def all_proposals(self) -> tuple[MemoryProposal, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM memory_proposals ORDER BY proposal_id"
        ).fetchall()
        return tuple(MemoryProposal.model_validate_json(row[0]) for row in rows)

    def put_proposal(self, proposal: MemoryProposal) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO memory_proposals(proposal_id, operation_key, workspace_id, "
                "owner_user_id, status, payload_json) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(proposal_id) DO UPDATE SET status=excluded.status, "
                "payload_json=excluded.payload_json",
                (
                    proposal.proposal_id,
                    proposal.operation_key,
                    proposal.workspace_id,
                    proposal.owner_user_id,
                    proposal.status,
                    _json(proposal),
                ),
            )

    def get_maintenance(self, maintenance_id: str) -> MemoryMaintenanceState | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM memory_maintenance WHERE maintenance_id = ?",
            (maintenance_id,),
        ).fetchone()
        return MemoryMaintenanceState.model_validate_json(row[0]) if row else None

    def put_maintenance(self, state: MemoryMaintenanceState) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO memory_maintenance(maintenance_id, workspace_id, owner_user_id, "
                "payload_json) VALUES (?, ?, ?, ?) ON CONFLICT(maintenance_id) DO UPDATE SET "
                "payload_json=excluded.payload_json",
                (
                    state.maintenance_id,
                    state.workspace_id,
                    state.owner_user_id,
                    _json(state),
                ),
            )


__all__ = ["SQLiteMemoryStore", "SQLiteRunStore", "SQLiteSpecFactStore"]
