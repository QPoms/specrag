"""Stable contracts shared by the durable Run harness and its adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from specrag.access import AccessPolicy
from specrag.models import (
    ActorContext,
    CorpusVersionRef,
    ResearchOutcome,
    ResearchRun,
    RunEvent,
    SpecificationScope,
)

from .application import ResearchApplication


class RunStore(Protocol):
    def put(self, run: ResearchRun) -> None: ...
    def get(self, run_id: str) -> ResearchRun | None: ...
    def get_by_submission(
        self, conversation_id: str, operation_key: str,
    ) -> ResearchRun | None: ...
    def list_runs(
        self, workspace_id: str, *, owner_user_id: str | None, limit: int,
    ) -> tuple[ResearchRun, ...]: ...
    def list_active_runs(self) -> tuple[ResearchRun, ...]: ...
    def is_cancel_requested(self, run_id: str) -> bool: ...
    def append_event(self, event: RunEvent) -> RunEvent: ...
    def events(self, run_id: str) -> tuple[RunEvent, ...]: ...
    def finalize(
        self, run: ResearchRun, event: RunEvent, outcome: ResearchOutcome | None,
    ) -> RunEvent: ...
    def get_outcome(
        self, run_id: str, run_attempt: int,
    ) -> ResearchOutcome | None: ...


ApplicationFactory = Callable[[int, Callable[[str], str | None],
    Callable[..., None], tuple[dict[str, object], ...]], ResearchApplication]
ContextSnapshot = Callable[[ActorContext, str, int], tuple[
    tuple[str, ...], tuple[str, ...], tuple[dict[str, object], ...]]]
ContextLoader = Callable[[ActorContext, ResearchRun], tuple[dict[str, object], ...]]
ProposalWriter = Callable[[ActorContext, ResearchRun, Any], tuple[str, ...]]
CorpusSnapshot = Callable[[SpecificationScope], tuple[CorpusVersionRef, ...]]
CorpusBindingWriter = Callable[[ResearchRun], None]
ContextWriter = Callable[[ActorContext, ResearchRun], None]
ResultLookup = Callable[[str], str | None]
CheckpointReader = Callable[[str], dict[str, Any] | None]


@dataclass(frozen=True)
class RunInspection:
    run: ResearchRun
    events: tuple[RunEvent, ...]
    checkpoint: dict[str, Any] | None
    stage_latency_ms: dict[str, float]
    input_tokens: int | None
    output_tokens: int | None
    result_id: str | None = None


__all__ = [
    "AccessPolicy",
    "ApplicationFactory",
    "ContextLoader",
    "ContextSnapshot",
    "ContextWriter",
    "CorpusBindingWriter",
    "CorpusSnapshot",
    "CheckpointReader",
    "ProposalWriter",
    "ResultLookup",
    "RunInspection",
    "RunStore",
]
