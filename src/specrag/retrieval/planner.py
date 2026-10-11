"""Protocol-agnostic, auditable planning for Discovery retrieval.

The Discovery question and its frozen angle are the semantic authority. The
planner may select and normalize spans present in the question, but it must not
add an unstated message, IE, interface, state, or other protocol fact.
Candidate specifications are resolved separately by corpus-level document
routing. Cross-specification expansion is permitted only when a first-hop
passage contains an explicit TS reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Mapping

from ..config import AppConfig, load_config
from .hybrid import (
    CorpusScope,
    CorpusVersionPolicy,
    HybridMarkdownRetriever,
    MarkdownChunk,
    effective_corpus_scope,
)
from .index import searchable_text
from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .reranker import CrossEncoderReranker
from .execution import (
    exact_match_priority as _exact_match_priority,
    execute_retrieval_task as _execute_task,
    fold_protocol_text as _fold_protocol_text,
    specification_references as _references,
)
from .ranking import ordered_unique as _ordered_unique
from .answer_synthesis import summarize_expanded_discovery_evidence
from .curation import curate_discovery_evidence
from .evidence_selection import select_expanded_discovery_evidence
from .expansion import (
    _grounded_followup_plan,
    _merge_stage_tasks,
    _second_stage_item,
    execute_expanded_discovery_retrieval,
)
from .relation_evidence import (
    _RELATION_MESSAGE,
    _build_grounded_relation_evidence,
    _useful_relation_term,
)
from .selected_answer import summarize_selected_discovery_evidence
from .normalization import (
    protocol_identifier_in_text,
    protocol_identifier_variants,
)
from .plan_models import (
    AnchorType,
    DiscoveryRetrievalPlanV2,
    DiscoveryRetrievalPlanV3,
    EvidenceTermProposal,
    ExplicitFacet,
    GroundedTerm,
    PLANNER_PROFILE,
    PlannedDiscoveryRetrieval,
    QueryAnchor,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    RetrievalSubquery,
    RoutingPolicy,
    SourceSpan,
    SpecificationRoute,
)
from .execution_models import (
    CuratedEvidenceItem,
    CuratedEvidenceTask,
    CuratedRetrievalExecution,
    EvidenceRelationClass,
    EvidenceExpansionBatch,
    EvidenceGroundedFollowup,
    ExpandedRetrievalExecution,
    FollowupProbeProposal,
    GroundedRelationEvidence,
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
    QuestionExpansionProposal,
    ReleaseComparisonExecution,
)
from .answer_models import (
    AgentEvidenceGroupProposal,
    AgentHypothesisEvidenceAssessmentProposal,
    AgentIndependentEvidenceGroupProposal,
    AgentQuestionAnswerProposal,
    AgentQuestionEvidenceSelectionProposal,
    AgentRepairableQuestionAnswerProposal,
    DiscoveryEvidenceSelection,
    DualStageRetrievalExecution,
    EvidenceAnswerSummary,
    EvidenceBackedAnswer,
    EvidenceGroundedExpansionPlan,
    EvidenceRepairQuery,
    EvidenceRepairSubQuery,
    EvidenceSourceStage,
    HypothesisEvidenceAssessment,
    HypothesisSupportStatus,
    QuestionAnswerSummary,
    QuestionEvidenceSelection,
    SelectedEvidenceGroup,
    SelectedSpecificationEvidence,
)
from .normalization import GENERIC_FACET_TERMS
from .routing import resolve_discovery_retrieval_plan

def execute_curated_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    app_config: AppConfig | None = None,
    first_stage_hits: int = 12,
    relation_candidates_per_question: int = 64,
    final_evidence_hits: int = 20,
    retriever: HybridMarkdownRetriever | None = None,
) -> CuratedRetrievalExecution:
    """Retrieve once, expand corpus-grounded relations, and curate evidence."""

    if first_stage_hits < 1:
        raise ValueError("first_stage_hits must be positive")
    if relation_candidates_per_question < 1:
        raise ValueError("relation_candidates_per_question must be positive")
    if final_evidence_hits < 1:
        raise ValueError("final_evidence_hits must be positive")
    cfg = (app_config or load_config()).retrieval
    active_scope = effective_corpus_scope(cfg)
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        active_scope = effective_corpus_scope(cfg, active_retriever)
        resolved = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        unavailable = PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=active_scope.version_policy,
            corpus_release=active_scope.release,
            corpus_versions=active_scope.versions,
            errors=(error,),
        )
        return CuratedRetrievalExecution(
            status="unavailable",
            first_stage=unavailable,
            errors=(error,),
        )

    # Explicit-reference and relation discovery remain evidence channels, but
    # there is no Agent-authored query and no second hybrid retrieval pass.
    first_items = tuple(
        item.model_copy(update={
            "reference_expansion": ReferenceExpansionPolicy(enabled=False),
        })
        for item in resolved.retrievals
    )
    resolved_first = resolved.model_copy(update={"retrievals": first_items})
    first_tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=first_stage_hits,
            retrieval_stage=1,
        )
        for item in first_items
    )
    first_errors = tuple(error for task in first_tasks for error in task.errors)
    first_status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if first_tasks and all(task.status == "unavailable" for task in first_tasks)
        else "partial"
        if any(task.status in {"partial", "unavailable"} for task in first_tasks)
        else "completed"
    )
    first_execution = PlannedRetrievalExecution(
        status=first_status,
        resolved_plan=resolved_first,
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=active_scope.version_policy,
        corpus_release=active_scope.release,
        corpus_versions=active_scope.versions,
        tasks=first_tasks,
        errors=first_errors,
    )
    relation_candidates = _build_grounded_relation_evidence(
        active_retriever,
        resolved_first,
        max_per_question=relation_candidates_per_question,
    )
    curated_tasks = curate_discovery_evidence(
        resolved_first,
        first_tasks,
        relation_candidates,
        final_evidence_hits=final_evidence_hits,
    )
    status: Literal["completed", "partial", "unavailable"] = (
        "partial" if first_errors else "completed"
    )
    return CuratedRetrievalExecution(
        status=status,
        resolved_plan=resolved_first,
        first_stage=first_execution,
        relation_candidates=relation_candidates,
        curated_tasks=curated_tasks,
        errors=first_errors,
    )


def execute_dual_stage_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    expansion_agent: CodexStructuredAgent,
    app_config: AppConfig | None = None,
    first_stage_hits: int = 12,
    second_stage_hits: int = 12,
    retriever: HybridMarkdownRetriever | None = None,
) -> DualStageRetrievalExecution:
    """Retrieve, derive grounded follow-up probes with Codex, then retrieve again."""

    if first_stage_hits < 1 or second_stage_hits < 1:
        raise ValueError("stage hit limits must be positive")
    cfg = (app_config or load_config()).retrieval
    active_scope = effective_corpus_scope(cfg)
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        active_scope = effective_corpus_scope(cfg, active_retriever)
        resolved = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        unavailable = PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=active_scope.version_policy,
            corpus_release=active_scope.release,
            corpus_versions=active_scope.versions,
            errors=(error,),
        )
        return DualStageRetrievalExecution(
            status="unavailable",
            first_stage=unavailable,
            errors=(error,),
        )

    first_items = tuple(
        item.model_copy(update={
            "reference_expansion": ReferenceExpansionPolicy(enabled=False),
        })
        for item in resolved.retrievals
    )
    resolved_first = resolved.model_copy(update={"retrievals": first_items})
    first_tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=first_stage_hits,
            retrieval_stage=1,
        )
        for item in first_items
    )
    first_execution = PlannedRetrievalExecution(
        status=(
            "unavailable"
            if first_tasks and all(task.status == "unavailable" for task in first_tasks)
            else "partial"
            if any(task.status in {"partial", "unavailable"} for task in first_tasks)
            else "completed"
        ),
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=active_scope.version_policy,
        corpus_release=active_scope.release,
        corpus_versions=active_scope.versions,
        tasks=first_tasks,
        errors=tuple(error for task in first_tasks for error in task.errors),
    )
    relation_evidence = _build_grounded_relation_evidence(
        active_retriever,
        resolved_first,
    )
    try:
        expansion = _grounded_followup_plan(
            resolved_plan=resolved_first,
            first_stage_tasks=first_tasks,
            expansion_agent=expansion_agent,
            available_specifications=frozenset(
                document.specification
                for document in active_retriever.documents
                if document.specification is not None
            ),
            relation_evidence=relation_evidence,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return DualStageRetrievalExecution(
            status="partial" if first_tasks else "unavailable",
            resolved_first_stage_plan=resolved_first,
            relation_evidence=relation_evidence,
            first_stage=first_execution,
            merged_tasks=first_tasks,
            errors=(*first_execution.errors, error),
        )

    first_by_query = {item.query_id: item for item in first_items}
    second_tasks: list[PlannedRetrievalTaskResult] = []
    for followup in expansion.retrievals:
        if not followup.retrieval_probes:
            second_tasks.append(PlannedRetrievalTaskResult(
                retrieval_id=followup.retrieval_id,
                query_id=followup.query_id,
                planner_action="planned",
                status="no_hits",
            ))
            continue
        second_tasks.append(_execute_task(
            active_retriever,
            _second_stage_item(first_by_query[followup.query_id], followup),
            final_hits=second_stage_hits,
            retrieval_stage=2,
        ))
    second_task_tuple = tuple(second_tasks)
    second_execution = PlannedRetrievalExecution(
        status=(
            "unavailable"
            if second_task_tuple and all(
                task.status == "unavailable" for task in second_task_tuple
            )
            else "partial"
            if any(
                task.status in {"partial", "unavailable"}
                for task in second_task_tuple
            )
            else "completed"
        ),
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=active_scope.version_policy,
        corpus_release=active_scope.release,
        corpus_versions=active_scope.versions,
        tasks=second_task_tuple,
        errors=tuple(
            error for task in second_task_tuple for error in task.errors
        ),
    )
    merged = _merge_stage_tasks(first_tasks, second_task_tuple)
    all_errors = (*first_execution.errors, *second_execution.errors)
    status: Literal["completed", "partial", "unavailable"] = (
        "partial" if all_errors else "completed"
    )
    return DualStageRetrievalExecution(
        status=status,
        resolved_first_stage_plan=resolved_first,
        relation_evidence=relation_evidence,
        expansion_plan=expansion,
        first_stage=first_execution,
        second_stage=second_execution,
        merged_tasks=merged,
        errors=all_errors,
    )


def execute_planned_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    app_config: AppConfig | None = None,
    corpus_scope: CorpusScope | None = None,
    final_hits: int = 8,
    retriever: HybridMarkdownRetriever | None = None,
    reserve_probe_candidates: bool = False,
    ranking_mode: Literal[
        "full",
        "rrf",
        "rrf_anchor",
        "rrf_heading",
        "rrf_noise",
        "rrf_anchor_heading",
        "rrf_anchor_single",
        "rrf_anchor_full_coverage",
        "rrf_anchor_single_full_coverage",
        "rrf_anchor_single_if_full",
    ] = "full",
    anchor_weight: float = 0.004,
    single_channel_max_rank: int = 3,
    single_channel_weight: float = 1.0,
    full_coverage_bonus: float = 0.004,
    reranker: CrossEncoderReranker | None = None,
) -> PlannedRetrievalExecution:
    """Resolve routes, retrieve passages, and follow explicit TS references.

    ``reserve_probe_candidates`` preserves the historical channel/probe
    reservation behavior by default. Evaluation experiments may disable it
    to rank every deduplicated candidate only by the deterministic score.
    """

    if final_hits < 1:
        raise ValueError("final_hits must be positive")
    cfg = (app_config or load_config()).retrieval
    active_scope = corpus_scope or effective_corpus_scope(cfg)
    try:
        active_retriever = retriever or HybridMarkdownRetriever(
            cfg,
            corpus_scope=active_scope,
        )
        retriever_scope = effective_corpus_scope(cfg, active_retriever)
        if corpus_scope is not None and retriever_scope != corpus_scope:
            raise ValueError("retriever scope does not match requested corpus scope")
        active_scope = retriever_scope
        resolved_plan = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            reranker=(reranker.metadata() if reranker is not None else None),
            corpus_version_policy=active_scope.version_policy,
            corpus_release=active_scope.release,
            corpus_versions=active_scope.versions,
            tasks=tuple(
                PlannedRetrievalTaskResult(
                    retrieval_id=item.retrieval_id,
                    query_id=item.query_id,
                    planner_action=item.planner_action,
                    status="unavailable",
                    errors=(error,),
                )
                for item in plan.retrievals
            ),
            errors=(error,),
        )

    tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=final_hits,
            reserve_probe_candidates=reserve_probe_candidates,
            ranking_mode=ranking_mode,
            anchor_weight=anchor_weight,
            single_channel_max_rank=single_channel_max_rank,
            single_channel_weight=single_channel_weight,
            full_coverage_bonus=full_coverage_bonus,
            reranker=reranker,
        )
        for item in resolved_plan.retrievals
    )
    if tasks and all(item.status == "unavailable" for item in tasks):
        status: Literal["completed", "partial", "unavailable"] = "unavailable"
    elif any(item.status in {"partial", "unavailable"} for item in tasks):
        status = "partial"
    else:
        status = "completed"
    return PlannedRetrievalExecution(
        status=status,
        resolved_plan=resolved_plan,
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        reranker=(reranker.metadata() if reranker is not None else None),
        corpus_version_policy=active_scope.version_policy,
        corpus_release=active_scope.release,
        corpus_versions=active_scope.versions,
        tasks=tasks,
        errors=tuple(error for task in tasks for error in task.errors),
    )


def execute_release_comparison(
    plan: DiscoveryRetrievalPlanV3,
    *,
    base_release: int,
    target_release: int,
    app_config: AppConfig | None = None,
    final_hits: int = 8,
    reranker: CrossEncoderReranker | None = None,
) -> ReleaseComparisonExecution:
    """Run one plan against two isolated Release snapshots for later analysis."""

    if base_release == target_release:
        raise ValueError("release comparison requires two different Releases")
    config = app_config or load_config()
    scopes = (
        CorpusScope(
            version_policy="highest_version_within_release",
            release=base_release,
        ),
        CorpusScope(
            version_policy="highest_version_within_release",
            release=target_release,
        ),
    )
    base_execution, target_execution = (
        execute_planned_discovery_retrieval(
            plan,
            app_config=config,
            corpus_scope=scope,
            final_hits=final_hits,
            reranker=reranker,
        )
        for scope in scopes
    )
    executions = (base_execution, target_execution)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if all(execution.status == "unavailable" for execution in executions)
        else "partial"
        if any(execution.status != "completed" for execution in executions)
        else "completed"
    )
    errors = tuple(
        f"{label}: {error}"
        for label, execution in zip(("base", "target"), executions, strict=True)
        for error in execution.errors
    )
    return ReleaseComparisonExecution(
        status=status,
        base_release=base_release,
        target_release=target_release,
        base_execution=base_execution,
        target_execution=target_execution,
        errors=errors,
    )


__all__ = [
    "DualStageRetrievalExecution",
    "DiscoveryRetrievalPlanV2",
    "DiscoveryRetrievalPlanV3",
    "EvidenceExpansionBatch",
    "EvidenceGroundedExpansionPlan",
    "EvidenceGroundedFollowup",
    "EvidenceTermProposal",
    "ExplicitFacet",
    "FollowupProbeProposal",
    "PLANNER_PROFILE",
    "PlannedDiscoveryRetrieval",
    "PlannedRetrievalExecution",
    "PlannedRetrievalHit",
    "PlannedRetrievalTaskResult",
    "QuestionExpansionProposal",
    "QueryAnchor",
    "ReferenceExpansionPolicy",
    "ReleaseComparisonExecution",
    "RetrievalProbe",
    "RetrievalSubquery",
    "RoutingPolicy",
    "SourceSpan",
    "SpecificationRoute",
    "execute_dual_stage_discovery_retrieval",
    "execute_planned_discovery_retrieval",
    "execute_release_comparison",
    "resolve_discovery_retrieval_plan",
]
