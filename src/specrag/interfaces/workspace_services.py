"""Composition helpers for one authorized workspace command."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from specrag.access import LocalIdentityService
from specrag.config import REPO_ROOT
from specrag.infrastructure import (
    DEFAULT_DATABASE_PATH,
    SQLiteCorpusStore,
    SQLiteDatabase,
    SQLiteIdentityStore,
    SQLiteRunStore,
    SQLiteRunQueue,
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
from specrag.models import (
    ActorContext,
    CorpusBuildResult,
    CorpusVersionRef,
    ResearchRun,
)
from specrag.orchestration import RunHarness, SQLiteCheckpointManager
from specrag.orchestration.application import build_research_application
from specrag.specfacts import ClaimSpecFactDistiller, SpecFactService
from specrag.retrieval.corpus import discover_source_documents
from specrag.retrieval.corpus_sync import CorpusSyncService, active_snapshot
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.models import CorpusScope, SpecificationVersion
from specrag.config import load_config


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
    corpus_store: SQLiteCorpusStore
    corpus_sync: CorpusSyncService
    runs: SQLiteRunStore
    run_queue: SQLiteRunQueue
    harness: RunHarness

    @classmethod
    def create(
        cls, database_path: Path = DEFAULT_DATABASE_PATH,
    ) -> "WorkspaceServices":
        database = SQLiteDatabase(database_path)
        checkpoint = SQLiteCheckpointManager(database.path)
        identity = LocalIdentityService(SQLiteIdentityStore(database))
        memory = MemoryService(SQLiteMemoryStore(database))
        memory_proposals = MemoryProposalService(memory)
        memory_recall = MemoryRecallService(memory)
        memory_extraction = MemoryExtractionService(memory, memory_proposals)
        memory_maintenance = MemoryMaintenanceService(memory, memory_proposals)
        specfacts = SpecFactService(SQLiteSpecFactStore(database))
        corpus_store = SQLiteCorpusStore(database)
        runs = SQLiteRunStore(database)
        run_queue = SQLiteRunQueue(database)

        def build_document(document) -> CorpusBuildResult:
            assert document.specification is not None and document.version is not None
            cfg = load_config().retrieval
            scope = CorpusScope(
                version_policy="exact_version",
                versions=(SpecificationVersion(
                    specification=document.specification,
                    version=document.version,
                ),),
            )
            retriever = HybridMarkdownRetriever(cfg, corpus_scope=scope)
            if retriever.collection is None:
                raise RuntimeError(
                    retriever.vector_error or "vector index is unavailable"
                )
            stored = retriever.collection.get(
                ids=[item.chunk_id for item in retriever.chunks],
                include=[],
            )
            vector_count = len(stored.get("ids", ()))
            return CorpusBuildResult(
                index_fingerprint=retriever.fingerprint,
                chunk_count=len(retriever.chunks),
                lexical_record_count=len(retriever.bm25.chunks),
                vector_record_count=vector_count,
            )

        corpus_sync = CorpusSyncService(corpus_store, build_document)
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

        def corpus_snapshot(release: int) -> tuple[CorpusVersionRef, ...]:
            active = active_snapshot(corpus_store, release)
            cfg = load_config().retrieval.model_copy(update={
                "version_policy": "highest_version_within_release",
                "target_release": release,
            })
            selected = tuple(
                CorpusVersionRef(
                    specification=document.specification,
                    version=document.version,
                    release=document.release,
                )
                for document in discover_source_documents(cfg)
                if document.specification is not None and document.version is not None
            )
            if (
                active
                and {item.specification for item in active}
                == {item.specification for item in selected}
            ):
                return active
            return selected

        def bind_corpus(run: ResearchRun) -> None:
            version_ids = tuple(
                record.version_id
                for item in run.corpus_versions
                if (
                    record := corpus_store.find(
                        int(item.release), item.specification, item.version
                    )
                ) is not None
            )
            if version_ids:
                corpus_store.bind_run(run.run_id, version_ids)

        def factory(release, stop_requested, event_sink, research_context):
            run_configuration = next((
                item for item in research_context
                if item.get("kind") == "run_configuration"
            ), {})
            raw_versions = run_configuration.get("corpus_versions", [])
            corpus_scope = None
            if isinstance(raw_versions, list) and raw_versions:
                versions = tuple(
                    SpecificationVersion.model_validate(item)
                    for item in raw_versions
                )
                corpus_scope = CorpusScope(
                    version_policy=(
                        "exact_version" if len(versions) == 1 else "mixed_versions"
                    ),
                    versions=versions,
                )
            return build_research_application(
                release=release,
                checkpointer=checkpoint.saver,
                stop_requested=stop_requested,
                event_sink=event_sink,
                research_context=research_context,
                corpus_scope=corpus_scope,
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
            corpus_snapshot=corpus_snapshot,
            corpus_binding_writer=bind_corpus,
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
            corpus_store,
            corpus_sync,
            runs,
            run_queue,
            harness,
        )

    def close(self) -> None:
        self.checkpoint.close()
        self.database.close()


__all__ = ["WorkspaceServices"]
