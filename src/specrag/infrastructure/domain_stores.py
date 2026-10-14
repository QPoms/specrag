"""Small in-memory adapters used by tests and local composition."""

from __future__ import annotations

from specrag.models import ResearchRun, RunEvent, SpecFact, SpecFactProposal, UserMemoryItem


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


class InMemoryUserMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, UserMemoryItem] = {}

    def all(self) -> tuple[UserMemoryItem, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def put(self, item: UserMemoryItem) -> None:
        self._records[item.memory_id] = item


__all__ = ["InMemoryRunStore", "InMemorySpecFactStore", "InMemoryUserMemoryStore"]
