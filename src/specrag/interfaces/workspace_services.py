"""Composition helpers for one authorized workspace command."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from specrag.access import LocalIdentityService
from specrag.config import REPO_ROOT
from specrag.infrastructure import (
    SQLiteDatabase,
    SQLiteIdentityStore,
    SQLiteRunStore,
    SQLiteSpecFactStore,
    SQLiteMemoryStore,
)
from specrag.memory import (
    MemoryExtractionService,
    MemoryMaintenanceService,
    MemoryProposalService,
    MemoryRecallService,
    MemoryService,
)
from specrag.models import ActorContext, ResearchRun
from specrag.orchestration import RunHarness, SQLiteCheckpointManager
from specrag.orchestration.application import build_research_application
from specrag.specfacts import ClaimSpecFactDistiller, SpecFactService


@dataclass
class WorkspaceServices:
    database: SQLiteDatabase
    checkpoint: SQLiteCheckpointManager
    identity: LocalIdentityService
    memory: MemoryService
    memory_proposals: MemoryProposalService
    memory_recall: MemoryRecallService
    memory_extraction: MemoryExtractionService
    memory_maintenance: MemoryMaintenanceService
    specfacts: SpecFactService
    runs: SQLiteRunStore
    harness: RunHarness

    @classmethod
    def create(cls) -> "WorkspaceServices":
        database = SQLiteDatabase()
        checkpoint = SQLiteCheckpointManager(database.path)
        identity = LocalIdentityService(SQLiteIdentityStore(database))
        memory = MemoryService(SQLiteMemoryStore(database))
        memory_proposals = MemoryProposalService(memory)
        memory_recall = MemoryRecallService(memory)
        memory_extraction = MemoryExtractionService(memory, memory_proposals)
        memory_maintenance = MemoryMaintenanceService(memory, memory_proposals)
        specfacts = SpecFactService(SQLiteSpecFactStore(database))
        runs = SQLiteRunStore(database)
        def snapshot(
            context: ActorContext, question: str, release: int,
        ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[dict[str, object], ...]]:
            def source_is_valid(source) -> bool:
                relative = Path(source.source_path)
                path = (REPO_ROOT / relative).resolve()
                try:
                    path.relative_to((REPO_ROOT / "data/3gpp").resolve())
                except ValueError:
                    return False
                if not path.is_file():
                    return False
                if hashlib.sha256(path.read_bytes()).hexdigest().upper() != source.document_sha256:
                    return False
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                return 1 <= source.line_start <= source.line_end <= line_count

            facts = specfacts.search(
                context, question, release=release, source_is_valid=source_is_valid
            )
            recalled = memory_recall.recall(context, question)
            fact_context = tuple({
                "kind": "approved_specfact_hint",
                "fact_id": item.fact.fact_id,
                "assertion": item.fact.assertion,
                "source_paths": [source.source_path for source in item.fact.sources],
                "instruction": "Use only as a retrieval hint; re-open and validate source text.",
            } for item in facts)
            memory_context = memory.as_research_context(recalled.entries)
            return (
                tuple(item.fact.fact_id for item in facts),
                tuple(item.memory_id for item in recalled.entries),
                (*fact_context, *memory_context),
            )

        def load_snapshot(
            context: ActorContext, run: ResearchRun,
        ) -> tuple[dict[str, object], ...]:
            if context.workspace_id != run.workspace_id:
                raise PermissionError("access denied")
            facts_by_id = {
                item.fact_id: item for item in specfacts.store.all_facts()
                if item.workspace_id == run.workspace_id
            }
            memories = memory.snapshot_for_run(context, run)
            memory_context = memory.as_research_context(memories)
            fact_context = tuple({
                "kind": "approved_specfact_hint",
                "fact_id": item.fact_id,
                "assertion": item.assertion,
                "source_paths": [source.source_path for source in item.sources],
                "instruction": "Use only as a retrieval hint; re-open and validate source text.",
            } for fact_id in run.specfact_ids if (item := facts_by_id.get(fact_id)))
            return (*fact_context, *memory_context)

        def factory(release, stop_requested, event_sink, research_context):
            return build_research_application(
                release=release,
                checkpointer=checkpoint.saver,
                stop_requested=stop_requested,
                event_sink=event_sink,
                research_context=research_context,
            )

        def propose(context: ActorContext, run: ResearchRun, state) -> tuple[str, ...]:
            synthesis = state.get("synthesis")
            if synthesis is None:
                return ()
            proposals = specfacts.propose_from_synthesis(
                context,
                run_id=run.run_id,
                synthesis=synthesis,
                tasks=state["research_tasks"],
                candidates=state["candidate_evidence"],
                validated=state["validated_evidence"],
                sufficiency=state["sufficiency_assessments"],
                distiller=ClaimSpecFactDistiller(),
            )
            return tuple(item.proposal_id for item in proposals)

        harness = RunHarness(
            runs,
            factory,
            context_snapshot=snapshot,
            context_loader=load_snapshot,
            proposal_writer=propose,
        )
        return cls(
            database,
            checkpoint,
            identity,
            memory,
            memory_proposals,
            memory_recall,
            memory_extraction,
            memory_maintenance,
            specfacts,
            runs,
            harness,
        )

    def close(self) -> None:
        self.checkpoint.close()
        self.database.close()


__all__ = ["WorkspaceServices"]
