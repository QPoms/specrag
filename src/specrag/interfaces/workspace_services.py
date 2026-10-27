"""Composition helpers for one authorized workspace command."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from collections.abc import Callable

from specrag.access import LocalIdentityService
from specrag.context import ContextHarness
from specrag.config import REPO_ROOT
from specrag.infrastructure import (
    DEFAULT_DATABASE_PATH,
    SQLiteContextStore,
    SQLiteConversationLifecycleStore,
    SQLiteConversationAgentStore,
    SQLiteConversationStore,
    SQLiteConversationScopeStore,
    SQLiteConversationTurnQueue,
    SQLiteCorpusStore,
    SQLiteDatabase,
    SQLiteIdentityStore,
    SQLiteRunStore,
    SQLiteRunCompletionQueue,
    SQLiteResearchResultStore,
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
    SpecificationScope,
)
from specrag.orchestration import (
    ConversationService,
    ResearchResultService,
    RunHarness,
    SQLiteCheckpointManager,
)
from specrag.orchestration.application import build_research_application
from specrag.infrastructure.llm_budget import build_token_budget_manager
from specrag.specfacts import SpecFactService
from specrag.retrieval.corpus import discover_source_documents
from specrag.retrieval.corpus_sync import CorpusSyncService, active_snapshot
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.pool import RetrieverPool
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
    context_harness: ContextHarness
    conversation_store: SQLiteConversationStore
    conversation_scope_store: SQLiteConversationScopeStore
    conversation_turn_queue: SQLiteConversationTurnQueue
    conversation_lifecycle_store: SQLiteConversationLifecycleStore
    conversation_agent_store: SQLiteConversationAgentStore
    completion_queue: SQLiteRunCompletionQueue
    conversations: ConversationService
    result_store: SQLiteResearchResultStore
    research_results: ResearchResultService
    harness: RunHarness
    retriever_pool: RetrieverPool

    def prewarm_research_runtime(
        self,
        release: int = 17,
        *,
        stage_observer: Callable[[str, float], None] | None = None,
    ) -> HybridMarkdownRetriever:
        """Open one validated immutable corpus runtime before serving tools."""

        def measured(name: str, operation):
            started = perf_counter()
            result = operation()
            if stage_observer is not None:
                stage_observer(name, (perf_counter() - started) * 1000)
            return result

        if self.harness.corpus_snapshot is None:
            raise RuntimeError("corpus snapshot service is unavailable")
        versions = measured(
            "corpus_snapshot",
            lambda: self.harness.corpus_snapshot(SpecificationScope(
                policy="highest_version_within_release",
                release=release,
            )),
        )
        local = measured(
            "local_snapshot_discovery",
            lambda: discover_source_documents(
                load_config().retrieval,
                CorpusScope(
                    version_policy="highest_version_within_release",
                    release=release,
                ),
            ),
        )

        def validate_snapshot() -> tuple[SpecificationVersion, ...]:
            active_identity = {
                (item.specification, item.version) for item in versions
            }
            local_identity = {
                (item.specification, item.version)
                for item in local
                if item.specification is not None and item.version is not None
            }
            if active_identity != local_identity:
                raise RuntimeError(
                    f"Release {release} active corpus does not match the local "
                    "source snapshot"
                )
            return tuple(
                SpecificationVersion(
                    specification=item.specification,
                    version=item.version,
                    release=item.release,
                )
                for item in versions
            )

        bound = measured("snapshot_consistency", validate_snapshot)
        return measured(
            "retriever_construction",
            lambda: self.retriever_pool.get(CorpusScope(
                version_policy=(
                    "exact_version" if len(bound) == 1 else "mixed_versions"
                ),
                versions=bound,
            )),
        )

    @classmethod
    def create(
        cls, database_path: Path = DEFAULT_DATABASE_PATH,
    ) -> "WorkspaceServices":
        database = SQLiteDatabase(database_path)
        app_config = load_config()
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
        context_harness = ContextHarness(
            SQLiteContextStore(database),
            build_token_budget_manager(app_config),
            tool_result_soft_limit_tokens=(
                app_config.llm.tool_result_soft_limit_tokens
            ),
            recent_history_tokens=(
                app_config.llm.compaction_recent_history_tokens
            ),
        )
        conversation_store = SQLiteConversationStore(database)
        conversation_scope_store = SQLiteConversationScopeStore(database)
        conversation_turn_queue = SQLiteConversationTurnQueue(database)
        conversation_lifecycle_store = SQLiteConversationLifecycleStore(database)
        conversation_agent_store = SQLiteConversationAgentStore(database)
        completion_queue = SQLiteRunCompletionQueue(database)
        conversations = ConversationService(
            conversation_store,
            context_harness,
        )
        result_store = SQLiteResearchResultStore(database)
        research_results = ResearchResultService(
            runs,
            result_store,
            conversation_store,
        )

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
        retriever_pool = RetrieverPool(app_config.retrieval)
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

        def corpus_snapshot(
            requested: SpecificationScope,
        ) -> tuple[CorpusVersionRef, ...]:
            supported_release = 17
            requested_releases = {
                int(item.release) for item in requested.versions
                if item.release is not None
            }
            if requested.release is not None:
                requested_releases.add(requested.release)
            if (
                requested.policy == "latest_per_spec"
                or requested_releases != {supported_release}
            ):
                raise ValueError(
                    "only Release 17 research is currently supported"
                )
            if requested.policy == "highest_version_within_release":
                assert requested.release is not None
                selected = active_snapshot(corpus_store, requested.release)
                if not selected:
                    raise ValueError(
                        f"Release {requested.release} corpus is not active; run "
                        f"`specrag corpus sync --release {requested.release} --apply`"
                    )
            else:
                selected = tuple(CorpusVersionRef(
                    specification=item.specification,
                    version=item.version,
                    release=item.release,
                ) for item in requested.versions)
            for item in selected:
                record = corpus_store.find(
                    int(item.release), item.specification, item.version
                )
                if record is None or record.status not in {
                    "ready", "active", "draining"
                }:
                    raise ValueError(
                        f"{item.specification} v{item.version} index is not ready"
                    )
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
                context_harness=context_harness,
                retriever=retriever_pool.get(
                    corpus_scope or CorpusScope(
                        version_policy="highest_version_within_release",
                        release=release,
                    )
                ),
            )

        harness = RunHarness(
            runs,
            factory,
            context_snapshot=snapshot,
            context_loader=load_snapshot,
            corpus_snapshot=corpus_snapshot,
            corpus_binding_writer=bind_corpus,
            context_writer=lambda actor, run: context_harness.create_context(
                run.run_id,
                workspace_id=actor.workspace_id,
                owner_user_id=actor.user_id,
                run_id=run.run_id,
            ),
            result_lookup=lambda run_id: (
                result.result_id
                if (result := result_store.latest_for_run(run_id)) is not None
                else None
            ),
            checkpoint_reader=lambda thread_id: (
                dict(checkpoint_tuple.checkpoint.get("channel_values", {}))
                if (
                    checkpoint_tuple := checkpoint.saver.get_tuple({
                        "configurable": {"thread_id": thread_id}
                    })
                ) is not None
                else None
            ),
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
            context_harness,
            conversation_store,
            conversation_scope_store,
            conversation_turn_queue,
            conversation_lifecycle_store,
            conversation_agent_store,
            completion_queue,
            conversations,
            result_store,
            research_results,
            harness,
            retriever_pool,
        )

    def close(self) -> None:
        self.checkpoint.close()
        self.database.close()


__all__ = ["WorkspaceServices"]
