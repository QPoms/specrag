"""Authorized read and lifecycle access projections for Research Runs."""

from __future__ import annotations

from typing import Protocol

from specrag.access import AccessPolicy
from specrag.models import ActorContext, ResearchRun, RunEvent


class RunAccessStore(Protocol):
    def get(self, run_id: str) -> ResearchRun | None: ...
    def list_runs(
        self, workspace_id: str, *, owner_user_id: str | None, limit: int
    ) -> tuple[ResearchRun, ...]: ...
    def events(self, run_id: str) -> tuple[RunEvent, ...]: ...


class RunAccessMixin:
    """Separate private Run content from operator-safe lifecycle access."""

    store: RunAccessStore
    policy: AccessPolicy

    def status(self, context: ActorContext, run_id: str) -> ResearchRun:
        return self._authorized_run(context, run_id, "run.content.read")

    def metadata_status(self, context: ActorContext, run_id: str) -> ResearchRun:
        return self._authorized_run(context, run_id, "run.metadata.read")

    def content_status(self, context: ActorContext, run_id: str) -> ResearchRun:
        return self._authorized_run(context, run_id, "run.content.read")

    def list_runs(
        self, context: ActorContext, *, limit: int = 50
    ) -> tuple[ResearchRun, ...]:
        self.policy.require(
            context, "run.content.read", workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        return self.store.list_runs(
            context.workspace_id, owner_user_id=context.user_id, limit=limit
        )

    def list_run_metadata(
        self, context: ActorContext, *, limit: int = 50
    ) -> tuple[ResearchRun, ...]:
        self.policy.require(
            context, "run.metadata.read", workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        return self.store.list_runs(
            context.workspace_id,
            owner_user_id=None if context.role == "admin" else context.user_id,
            limit=limit,
        )

    def events(self, context: ActorContext, run_id: str) -> tuple[RunEvent, ...]:
        self._authorized_run(context, run_id, "run.content.read")
        return self.store.events(run_id)

    def _authorized_run(
        self, context: ActorContext, run_id: str, permission: str
    ) -> ResearchRun:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        self.policy.require(
            context, permission, workspace_id=run.workspace_id,
            owner_user_id=run.owner_user_id,
        )
        return run


__all__ = ["RunAccessMixin", "RunAccessStore"]
