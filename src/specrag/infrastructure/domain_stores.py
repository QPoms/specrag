"""Small in-memory adapters used by tests and local composition."""

from __future__ import annotations

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


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, ResearchRun] = {}
        self._events: dict[str, list[RunEvent]] = {}

    def put(self, run: ResearchRun) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> ResearchRun | None:
        return self._runs.get(run_id)

    def is_cancel_requested(self, run_id: str) -> bool:
        run = self.get(run_id)
        return bool(run and run.cancel_requested_at)

    def append_event(self, event: RunEvent) -> RunEvent:
        events = self._events.setdefault(event.run_id, [])
        if event.operation_key is not None:
            existing = next((
                item for item in events if item.operation_key == event.operation_key
            ), None)
            if existing is not None:
                return existing
        events.append(event)
        return event

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        return tuple(self._events.get(run_id, ()))


class InMemorySpecFactStore:
    def __init__(self) -> None:
        self._proposals: dict[str, SpecFactProposal] = {}
        self._facts: dict[str, SpecFact] = {}

    def all_proposals(self) -> tuple[SpecFactProposal, ...]:
        return tuple(self._proposals[key] for key in sorted(self._proposals))

    def put_proposal(self, proposal: SpecFactProposal) -> None:
        self._proposals[proposal.proposal_id] = proposal

    def all_facts(self) -> tuple[SpecFact, ...]:
        return tuple(self._facts[key] for key in sorted(self._facts))

    def put_fact(self, fact: SpecFact) -> None:
        self._facts[fact.fact_id] = fact


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._topics: dict[str, MemoryTopic] = {}
        self._proposals: dict[str, MemoryProposal] = {}
        self._maintenance: dict[str, MemoryMaintenanceState] = {}

    def all_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def put_entry(self, item: MemoryEntry) -> None:
        self._entries[item.memory_id] = item

    def all_topics(self) -> tuple[MemoryTopic, ...]:
        return tuple(self._topics[key] for key in sorted(self._topics))

    def put_topic(self, topic: MemoryTopic) -> None:
        self._topics[topic.topic_id] = topic

    def all_proposals(self) -> tuple[MemoryProposal, ...]:
        return tuple(self._proposals[key] for key in sorted(self._proposals))

    def put_proposal(self, proposal: MemoryProposal) -> None:
        self._proposals[proposal.proposal_id] = proposal

    def get_maintenance(self, maintenance_id: str) -> MemoryMaintenanceState | None:
        return self._maintenance.get(maintenance_id)

    def put_maintenance(self, state: MemoryMaintenanceState) -> None:
        self._maintenance[state.maintenance_id] = state


__all__ = ["InMemoryMemoryStore", "InMemoryRunStore", "InMemorySpecFactStore"]
