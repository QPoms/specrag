"""Low-frequency Memory consolidation and capacity maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from specrag.access import AccessPolicy
from specrag.models import (
    ActorContext,
    MemoryEntry,
    MemoryMaintenanceState,
    MemoryMutationDraft,
    MemoryProposal,
)

from .service import MemoryService, _key
from .proposals import MemoryProposalService


class MemoryConsolidator(Protocol):
    def consolidate(
        self, entries: tuple[MemoryEntry, ...]
    ) -> tuple[MemoryMutationDraft, ...]: ...


@dataclass(frozen=True)
class MaintenanceEligibility:
    eligible: bool
    elapsed_seconds: int
    changes_since_consolidation: int


class MemoryMaintenanceService:
    def __init__(
        self,
        memory: MemoryService,
        proposals: MemoryProposalService,
        *,
        minimum_interval: timedelta = timedelta(hours=24),
        minimum_changes: int = 10,
        topic_capacity: int = 200,
        policy: AccessPolicy | None = None,
    ) -> None:
        if minimum_interval.total_seconds() < 0 or minimum_changes < 1:
            raise ValueError("invalid consolidation thresholds")
        if topic_capacity < 1:
            raise ValueError("Topic capacity must be positive")
        self.memory = memory
        self.proposals = proposals
        self.minimum_interval = minimum_interval
        self.minimum_changes = minimum_changes
        self.topic_capacity = topic_capacity
        self.policy = policy or AccessPolicy()

    def eligibility(
        self,
        context: ActorContext,
        *,
        scope: str,
        now: datetime | None = None,
    ) -> MaintenanceEligibility:
        owner = context.user_id if scope == "user" else None
        self._authorize(context, scope, owner)
        current = now or datetime.now(timezone.utc)
        state = self._state(scope, context.workspace_id, owner)
        active = self._active(scope, context.workspace_id, owner)
        baseline = state.last_consolidated_at
        if baseline is None and active:
            baseline = min(item.created_at for item in active)
        elapsed = current - baseline if baseline is not None else timedelta(0)
        return MaintenanceEligibility(
            eligible=(
                elapsed >= self.minimum_interval
                and state.changes_since_consolidation >= self.minimum_changes
            ),
            elapsed_seconds=max(0, int(elapsed.total_seconds())),
            changes_since_consolidation=state.changes_since_consolidation,
        )

    def propose_consolidation(
        self,
        context: ActorContext,
        *,
        scope: str,
        consolidator: MemoryConsolidator,
        now: datetime | None = None,
    ) -> tuple[MemoryProposal, ...]:
        check = self.eligibility(context, scope=scope, now=now)
        if not check.eligible:
            raise ValueError("Memory consolidation thresholds are not satisfied")
        owner = context.user_id if scope == "user" else None
        entries = self._active(scope, context.workspace_id, owner)
        drafts = consolidator.consolidate(entries)
        proposals: list[MemoryProposal] = []
        for index, draft in enumerate(drafts):
            if draft.operation != "merge" or draft.scope != scope:
                raise ValueError("consolidation may only propose same-scope merges")
            proposals.append(
                self.proposals.propose(
                    context,
                    draft,
                    owner_user_id=owner,
                    now=now,
                    operation_key=_key(
                        "memory:consolidate:",
                        scope,
                        context.workspace_id,
                        owner or "workspace",
                        str(index),
                        draft.model_dump_json(),
                    ),
                )
            )
        return tuple(proposals)

    def propose_pruning(
        self,
        context: ActorContext,
        *,
        now: datetime | None = None,
    ) -> tuple[MemoryProposal, ...]:
        self.policy.require(
            context, "memory.project.manage", workspace_id=context.workspace_id
        )
        if self.eligibility(context, scope="workspace", now=now).eligible:
            raise ValueError("consolidation is due before capacity pruning")
        topics = {topic.topic_id: topic for topic in self.memory.list_topics(context)}
        candidates = tuple(
            sorted(
                (
                    item
                    for item in self._active("workspace", context.workspace_id, None)
                    if item.layer == "topic"
                    and item.topic_id in topics
                    and not topics[item.topic_id].pinned
                ),
                key=lambda item: (item.updated_at, item.memory_id),
            )
        )
        excess = len(candidates) - self.topic_capacity
        if excess <= 0:
            return ()
        draft = MemoryMutationDraft(
            operation="archive",
            scope="workspace",
            target_memory_ids=tuple(item.memory_id for item in candidates[:excess]),
            source_refs=("capacity-maintenance",),
        )
        return (
            self.proposals.propose(
                context,
                draft,
                now=now,
                operation_key=_key(
                    "memory:prune:",
                    context.workspace_id,
                    *draft.target_memory_ids,
                ),
            ),
        )

    def propose_restore(
        self,
        context: ActorContext,
        *memory_ids: str,
        now: datetime | None = None,
    ) -> MemoryProposal:
        self.policy.require(
            context, "memory.project.manage", workspace_id=context.workspace_id
        )
        if not memory_ids:
            raise ValueError("restore requires at least one Memory ID")
        draft = MemoryMutationDraft(
            operation="restore",
            scope="workspace",
            target_memory_ids=tuple(memory_ids),
            source_refs=("capacity-maintenance-restore",),
        )
        return self.proposals.propose(
            context,
            draft,
            now=now,
            operation_key=_key(
                "memory:restore:", context.workspace_id, *draft.target_memory_ids
            ),
        )

    def mark_consolidated(
        self,
        context: ActorContext,
        proposal_id: str,
        *,
        now: datetime | None = None,
    ) -> MemoryMaintenanceState:
        proposal = next(
            (
                item
                for item in self.memory.store.all_proposals()
                if item.proposal_id == proposal_id
            ),
            None,
        )
        if proposal is None or proposal.status != "approved":
            raise ValueError("approved consolidation proposal is required")
        if proposal.draft.operation != "merge":
            raise ValueError("proposal is not a consolidation merge")
        owner = proposal.owner_user_id
        self._authorize(context, proposal.draft.scope, owner)
        state = self._state(proposal.draft.scope, proposal.workspace_id, owner)
        updated = state.model_copy(
            update={
                "last_consolidated_at": now or datetime.now(timezone.utc),
                "changes_since_consolidation": 0,
            }
        )
        self.memory.store.put_maintenance(updated)
        return updated

    def _authorize(
        self, context: ActorContext, scope: str, owner_user_id: str | None
    ) -> None:
        if scope == "user":
            self.policy.require(
                context,
                "memory.user.manage",
                workspace_id=context.workspace_id,
                owner_user_id=owner_user_id,
            )
        elif scope == "workspace":
            self.policy.require(
                context, "memory.project.manage", workspace_id=context.workspace_id
            )
        else:
            raise ValueError("Memory scope must be user or workspace")

    def _state(
        self, scope: str, workspace_id: str, owner_user_id: str | None
    ) -> MemoryMaintenanceState:
        maintenance_id = _key(
            "MM-", scope, workspace_id, owner_user_id or "workspace"
        )
        return self.memory.store.get_maintenance(maintenance_id) or MemoryMaintenanceState(
            maintenance_id=maintenance_id,
            scope=scope,
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
        )

    def _active(
        self, scope: str, workspace_id: str, owner_user_id: str | None
    ) -> tuple[MemoryEntry, ...]:
        current = datetime.now(timezone.utc)
        return tuple(
            item
            for item in self.memory.store.all_entries()
            if item.scope == scope
            and item.workspace_id == workspace_id
            and item.owner_user_id == owner_user_id
            and item.status == "active"
            and (item.expires_at is None or item.expires_at > current)
        )


__all__ = [
    "MaintenanceEligibility", "MemoryConsolidator", "MemoryMaintenanceService",
]
