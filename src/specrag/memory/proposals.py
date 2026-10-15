"""Reviewable Memory mutation proposals and application."""

from __future__ import annotations

from datetime import datetime, timezone

from specrag.access import AccessPolicy, AuthorizationError
from specrag.models import ActorContext, MemoryMutationDraft, MemoryProposal

from .service import MemoryService, _key


class MemoryProposalService:
    def __init__(
        self, memory: MemoryService, policy: AccessPolicy | None = None
    ) -> None:
        self.memory = memory
        self.policy = policy or AccessPolicy()

    def propose(
        self,
        context: ActorContext,
        draft: MemoryMutationDraft,
        *,
        owner_user_id: str | None = None,
        proposed_by: str | None = None,
        now: datetime | None = None,
        operation_key: str | None = None,
    ) -> MemoryProposal:
        owner = owner_user_id or (context.user_id if draft.scope == "user" else None)
        if draft.scope == "user":
            self.policy.require(
                context,
                "memory.user.manage",
                workspace_id=context.workspace_id,
                owner_user_id=owner,
            )
        else:
            self.policy.require(
                context, "memory.project.propose", workspace_id=context.workspace_id
            )
        self._validate_targets(context.workspace_id, owner, draft)
        timestamp = now or datetime.now(timezone.utc)
        material = draft.model_dump_json()
        op_key = operation_key or _key(
            "memory:proposal:", context.workspace_id, owner or "workspace", material
        )
        existing = next(
            (
                item
                for item in self.memory.store.all_proposals()
                if item.operation_key == op_key
            ),
            None,
        )
        if existing is not None:
            return existing
        proposal = MemoryProposal(
            proposal_id=_key("MP-", op_key),
            operation_key=op_key,
            workspace_id=context.workspace_id,
            owner_user_id=owner,
            draft=draft,
            proposed_by=proposed_by or context.user_id,
            proposed_at=timestamp,
        )
        self.memory.store.put_proposal(proposal)
        return proposal

    def review(
        self,
        context: ActorContext,
        proposal_id: str,
        *,
        approve: bool,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> MemoryProposal:
        proposal = next(
            (
                item
                for item in self.memory.store.all_proposals()
                if item.proposal_id == proposal_id
            ),
            None,
        )
        if proposal is None:
            raise KeyError(f"unknown Memory proposal: {proposal_id}")
        if proposal.draft.scope == "user":
            self.policy.require(
                context,
                "memory.user.manage",
                workspace_id=proposal.workspace_id,
                owner_user_id=proposal.owner_user_id,
            )
        else:
            self.policy.require(
                context, "memory.project.review", workspace_id=proposal.workspace_id
            )
        if proposal.status != "pending":
            return proposal
        if not approve and not (reason and reason.strip()):
            raise ValueError("rejection requires a reason")
        timestamp = now or datetime.now(timezone.utc)
        applied_ids: tuple[str, ...] = ()
        applied_topic_id: str | None = None
        if approve:
            applied_ids, applied_topic_id = self._apply(context, proposal, timestamp)
        reviewed = proposal.model_copy(
            update={
                "status": "approved" if approve else "rejected",
                "reviewer_user_id": context.user_id,
                "reviewed_at": timestamp,
                "review_reason": reason.strip() if reason else None,
                "applied_memory_ids": applied_ids,
                "applied_topic_id": applied_topic_id,
            }
        )
        self.memory.store.put_proposal(reviewed)
        return reviewed

    def list(self, context: ActorContext) -> tuple[MemoryProposal, ...]:
        records: list[MemoryProposal] = []
        for item in self.memory.store.all_proposals():
            if item.workspace_id != context.workspace_id:
                continue
            if item.draft.scope == "user":
                if item.owner_user_id != context.user_id:
                    continue
                self.policy.require(
                    context,
                    "memory.user.manage",
                    workspace_id=item.workspace_id,
                    owner_user_id=item.owner_user_id,
                )
            else:
                self.policy.require(
                    context, "memory.project.read", workspace_id=item.workspace_id
                )
            records.append(item)
        return tuple(sorted(records, key=lambda item: item.proposed_at))

    def _apply(
        self,
        context: ActorContext,
        proposal: MemoryProposal,
        timestamp: datetime,
    ) -> tuple[tuple[str, ...], str | None]:
        draft = proposal.draft
        if draft.operation == "create_topic":
            topic = self.memory.create_topic(
                context,
                slug=draft.topic_slug or "",
                description=draft.topic_description or "",
                pinned=draft.topic_pinned,
                now=timestamp,
                operation_key=f"{proposal.operation_key}:apply",
            )
            return (), topic.topic_id
        targets = tuple(self.memory._entry(item) for item in draft.target_memory_ids)
        if draft.operation in {"retract", "archive", "restore"}:
            status = {
                "retract": "retracted",
                "archive": "archived",
                "restore": "active",
            }[draft.operation]
            for item in targets:
                self.memory.store.put_entry(
                    item.model_copy(update={"status": status, "updated_at": timestamp})
                )
            self.memory._record_change(
                draft.scope, proposal.workspace_id, proposal.owner_user_id
            )
            if draft.operation == "archive":
                self._record_pruned(proposal, timestamp)
            return tuple(item.memory_id for item in targets), None
        if draft.operation == "add":
            item = self.memory._write_entry(
                scope=draft.scope,
                layer="topic" if draft.topic_id else "core",
                context=context,
                owner_user_id=proposal.owner_user_id,
                topic_id=draft.topic_id,
                kind=draft.kind or "preference",
                key=draft.key or "memory",
                content=draft.content or "",
                expires_at=None,
                origin="extractor",
                source_refs=draft.source_refs,
                operation_key=f"{proposal.operation_key}:apply",
                now=timestamp,
            )
            return (item.memory_id,), None
        if draft.operation == "supersede":
            old = targets[0]
            item = self.memory._write_entry(
                scope=old.scope,
                layer=old.layer,
                context=context,
                owner_user_id=old.owner_user_id,
                topic_id=old.topic_id,
                kind=draft.kind or old.kind,
                key=draft.key or old.key,
                content=draft.content or "",
                expires_at=old.expires_at,
                origin="extractor",
                source_refs=tuple(sorted(set((*old.source_refs, *draft.source_refs)))),
                operation_key=f"{proposal.operation_key}:apply",
                now=timestamp,
                supersedes=old,
            )
            return (item.memory_id,), None
        if draft.operation == "merge":
            first = targets[0]
            refs = set(draft.source_refs)
            for target in targets:
                refs.update(target.source_refs)
            item = self.memory._write_entry(
                scope=first.scope,
                layer=first.layer,
                context=context,
                owner_user_id=first.owner_user_id,
                topic_id=first.topic_id,
                kind=draft.kind or first.kind,
                key=draft.key or f"consolidated-{proposal.proposal_id[-8:].casefold()}",
                content=draft.content or "",
                expires_at=None,
                origin="consolidation",
                source_refs=tuple(sorted(refs)),
                operation_key=f"{proposal.operation_key}:apply",
                now=timestamp,
                merged_from_ids=tuple(item.memory_id for item in targets),
            )
            for target in targets:
                if target.memory_id != item.memory_id:
                    self.memory.store.put_entry(
                        target.model_copy(
                            update={"status": "superseded", "updated_at": timestamp}
                        )
                    )
            self._record_consolidated(proposal, timestamp)
            return (item.memory_id,), None
        raise ValueError(f"unsupported Memory operation: {draft.operation}")

    def _validate_targets(
        self,
        workspace_id: str,
        owner_user_id: str | None,
        draft: MemoryMutationDraft,
    ) -> None:
        targets = tuple(self.memory._entry(item) for item in draft.target_memory_ids)
        for item in targets:
            if item.workspace_id != workspace_id or item.scope != draft.scope:
                raise AuthorizationError("access denied")
            if draft.scope == "user" and item.owner_user_id != owner_user_id:
                raise AuthorizationError("access denied")
            expected_status = "archived" if draft.operation == "restore" else "active"
            if item.status != expected_status:
                raise ValueError(
                    "restore requires archived revisions; other proposals require active revisions"
                )
        if draft.topic_id is not None:
            topic = self.memory._topic(draft.topic_id)
            if topic.workspace_id != workspace_id or topic.status != "active":
                raise AuthorizationError("access denied")
        if draft.operation == "merge" and targets:
            collections = {
                (item.scope, item.owner_user_id, item.layer, item.topic_id, item.kind)
                for item in targets
            }
            if len(collections) != 1:
                raise ValueError("merge cannot cross a Memory scope, Topic, or kind")
            source_refs = {ref for item in targets for ref in item.source_refs}
            if not set(draft.source_refs).issubset(source_refs):
                raise ValueError("merge cannot introduce a new source reference")
        if draft.operation == "archive" and any(item.layer != "topic" for item in targets):
            raise ValueError("capacity pruning can archive only Topic Memory")
        if draft.operation == "archive" and any(
            self.memory._topic(item.topic_id or "").pinned for item in targets
        ):
            raise ValueError("pinned Topic Memory cannot be archived")
        if draft.operation == "restore" and any(item.layer != "topic" for item in targets):
            raise ValueError("only archived Topic Memory can be restored")

    def _record_consolidated(
        self, proposal: MemoryProposal, timestamp: datetime
    ) -> None:
        maintenance_id = _key(
            "MM-",
            proposal.draft.scope,
            proposal.workspace_id,
            proposal.owner_user_id or "workspace",
        )
        state = self.memory.store.get_maintenance(maintenance_id)
        if state is not None:
            self.memory.store.put_maintenance(
                state.model_copy(
                    update={
                        "last_consolidated_at": timestamp,
                        "changes_since_consolidation": 0,
                    }
                )
            )

    def _record_pruned(
        self, proposal: MemoryProposal, timestamp: datetime
    ) -> None:
        maintenance_id = _key(
            "MM-", "workspace", proposal.workspace_id, "workspace"
        )
        state = self.memory.store.get_maintenance(maintenance_id)
        if state is not None:
            self.memory.store.put_maintenance(
                state.model_copy(update={"last_pruned_at": timestamp})
            )


__all__ = ["MemoryProposalService"]
