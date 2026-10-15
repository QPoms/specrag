"""SQLite adapters for runs, events, SpecFacts, and Core/Topic Memory."""

from __future__ import annotations

import json
import sqlite3

from specrag.models import (
    MemoryEntry,
    MemoryMaintenanceState,
    MemoryProposal,
    MemoryTopic,
    ResearchRun,
    RunEvent,
    SpecFact,
    SpecFactProposal,
)

from .database import SQLiteDatabase


def _json(model) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


class SQLiteRunStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def put(self, run: ResearchRun) -> None:
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO runs(run_id, thread_id, workspace_id, owner_user_id, status, "
                "cancel_requested_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
                "cancel_requested_at=excluded.cancel_requested_at, payload_json=excluded.payload_json",
                (run.run_id, run.thread_id, run.workspace_id, run.owner_user_id,
                 run.status,
                 run.cancel_requested_at.isoformat() if run.cancel_requested_at else None,
                 _json(run)),
            )

    def get(self, run_id: str) -> ResearchRun | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return ResearchRun.model_validate_json(row[0]) if row else None

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
