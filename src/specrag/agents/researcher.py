"""Bounded Researcher loop over trusted local retrieval capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from specrag.config import AppConfig
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.models import ResearchTask
from specrag.retrieval.hybrid import (
    HybridMarkdownRetriever,
    effective_corpus_scope,
)
from specrag.retrieval.planner import (
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
)
from specrag.tools import (
    RetrievalToolRequest,
    RetrievalToolResult,
    ToolEvidence,
    ToolRegistry,
    build_retrieval_tool_registry,
)

from .models import (
    AgentActionOutcome,
    AgentRetrievalAction,
    AgentRetrievalDecision,
    AgentRetrievalTurn,
    AgenticRetrievalExecution,
)
from .policy import (
    action_signature,
    validate_action,
)
from .state import EvidenceLedger

@dataclass
class RegistryResearcherService:
    """Framework-independent application service for one Researcher worker."""

    controller_agent: StructuredResearcherController
    retriever: HybridMarkdownRetriever
    app_config: AppConfig
    tool_registry: ToolRegistry | None = None
    memory_reader: Any | None = None
    max_rounds: int = 4
    max_search_actions: int = 6
    max_expansion_actions: int = 3
    hits_per_search: int = 8
    max_evidence_per_question: int = 28

    def run(
        self,
        tasks: tuple[ResearchTask, ...],
        *,
        shared_context: Mapping[str, Any] | None = None,
        max_tool_calls: int | None = None,
    ) -> tuple[AgenticRetrievalExecution, dict[str, str]]:
        if not tasks:
            raise ValueError("Researcher requires at least one research task")
        memory_context: list[dict[str, object]] = []
        if self.memory_reader is not None:
            for task in tasks:
                memory_context.extend(
                    item.knowledge.model_dump(mode="json")
                    for item in self.memory_reader.search(task.question, limit=4)
                )
        execution = run_agentic_retrieval_loop(
            tasks,
            controller_agent=self.controller_agent,
            retriever=self.retriever,
            app_config=self.app_config,
            tool_registry=self.tool_registry,
            shared_context={
                **dict(shared_context or {}),
                "validated_memory": memory_context,
            },
            max_rounds=self.max_rounds,
            max_search_actions=self.max_search_actions,
            max_expansion_actions=self.max_expansion_actions,
            max_total_actions=max_tool_calls,
            hits_per_search=self.hits_per_search,
            max_evidence_per_question=self.max_evidence_per_question,
        )
        return execution, {task.task_id: task.task_id for task in tasks}


def _tool_request(
    action: AgentRetrievalAction,
    *,
    round_number: int,
    ledger: EvidenceLedger,
    result_limit: int,
) -> RetrievalToolRequest:
    seeds = tuple(
        hit.chunk_id
        for evidence_id in action.seed_evidence_ids
        if (hit := ledger.find(action.query_id, evidence_id)) is not None
    )
    if action.tool in {"keyword_search", "hybrid_search"}:
        query = action.lexical_query
    elif action.tool in {"semantic_search", "rerank"}:
        query = action.semantic_query
    else:
        query = ""
    semantic_query = (
        action.semantic_query
        if action.tool == "hybrid_search"
        else None
    )
    return RetrievalToolRequest(
        request_id=f"{action.query_id}:agent:r{round_number}:{action.action_id}",
        query_id=action.query_id,
        query=query,
        semantic_query=semantic_query,
        target_specifications=action.target_specifications,
        clause=action.clause or None,
        seed_chunk_ids=seeds,
        limit=min(action.max_results, result_limit),
    )


def _evidence_score(evidence: ToolEvidence) -> float:
    if evidence.score is None:
        return 1.0 / evidence.rank
    if evidence.score_kind == "semantic_distance":
        return 1.0 / (1.0 + max(evidence.score, 0.0))
    return evidence.score


def _planned_hits_from_tool(
    result: RetrievalToolResult,
    *,
    task: ResearchTask,
) -> tuple[PlannedRetrievalHit, ...]:
    hits: list[PlannedRetrievalHit] = []
    for evidence in result.evidence:
        score = _evidence_score(evidence)
        hits.append(PlannedRetrievalHit(
            hit_id=evidence.evidence_id,
            chunk_id=evidence.chunk_id,
            collection_rank=evidence.rank,
            specification=evidence.specification,
            release=evidence.release,
            version=evidence.version,
            source_path=evidence.source_path,
            document_sha256=evidence.document_sha256,
            heading_path=evidence.heading_path,
            line_start=evidence.line_start,
            line_end=evidence.line_end,
            excerpt=evidence.excerpt,
            evidence_roles=(task.objective,),
            probe_ids=(result.request_id,),
            retrieval_sources=(f"tool:{result.tool_name}",),
            retrieval_hop=(
                2 if result.tool_name == "expand_same_section" else 1
            ),
            semantic_rank=(
                evidence.rank
                if result.tool_name in {"semantic_search", "hybrid_search"}
                else None
            ),
            lexical_rank=(
                evidence.rank
                if result.tool_name in {"keyword_search", "hybrid_search"}
                else None
            ),
            rrf_score=(
                max(evidence.score or 0.0, 0.0)
                if evidence.score_kind == "rrf"
                else 0.0
            ),
            route_score=0.0,
            anchor_score=0.0,
            heading_score=0.0,
            agreement_score=0.0,
            noise_penalty=0.0,
            reranker_score=(
                evidence.score
                if evidence.score_kind == "cross_encoder"
                else None
            ),
            final_score=score,
        ))
    return tuple(hits)


def _evidence_execution(
    *,
    research_tasks: tuple[ResearchTask, ...],
    ledger: EvidenceLedger,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
) -> PlannedRetrievalExecution:
    active_scope = effective_corpus_scope(app_config.retrieval, retriever)
    tasks: list[PlannedRetrievalTaskResult] = []
    for task in research_tasks:
        hits = ledger.hits(task.task_id)
        tasks.append(PlannedRetrievalTaskResult(
            retrieval_id=f"RET-{task.task_id}",
            query_id=task.task_id,
            planner_action="planned",
            status="completed" if hits else "no_hits",
            hits=hits,
        ))
    has_hits = any(task.hits for task in tasks)
    return PlannedRetrievalExecution(
        status="completed" if has_hits else "unavailable",
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=app_config.retrieval.embedding_model,
        corpus_version_policy=active_scope.version_policy,
        corpus_release=active_scope.release,
        corpus_versions=active_scope.versions,
        tasks=tuple(tasks),
    )


def run_agentic_retrieval_loop(
    research_tasks: tuple[ResearchTask, ...],
    *,
    controller_agent: StructuredResearcherController,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    tool_registry: ToolRegistry | None = None,
    shared_context: Mapping[str, Any] | None = None,
    max_rounds: int = 4,
    max_search_actions: int = 6,
    max_expansion_actions: int = 3,
    max_total_actions: int | None = None,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
) -> AgenticRetrievalExecution:
    """Let an agent choose bounded calls to trusted local retrieval tools."""

    if not research_tasks:
        raise ValueError("Researcher requires at least one research task")

    for name, value in (
        ("max_rounds", max_rounds),
        ("max_search_actions", max_search_actions),
        ("hits_per_search", hits_per_search),
        ("max_evidence_per_question", max_evidence_per_question),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if max_expansion_actions < 0:
        raise ValueError("max_expansion_actions cannot be negative")
    if max_total_actions is not None and max_total_actions < 1:
        raise ValueError("max_total_actions must be positive when supplied")
    total_action_limit = max_total_actions or (
        max_search_actions + max_expansion_actions
    )

    task_by_id = {task.task_id: task for task in research_tasks}
    if len(task_by_id) != len(research_tasks):
        raise ValueError("Researcher task IDs must be unique")
    query_ids = tuple(task_by_id)
    query_id_set = frozenset(query_ids)
    available_specs = frozenset(
        document.specification
        for document in retriever.documents
        if document.specification is not None
    )
    active_scope = effective_corpus_scope(app_config.retrieval, retriever)
    ledger = EvidenceLedger(
        query_ids,
        max_per_question=max_evidence_per_question,
    )
    turns: list[AgentRetrievalTurn] = []
    errors: list[str] = []
    signatures: set[tuple[Any, ...]] = set()
    search_used = 0
    expansion_used = 0
    registry = tool_registry or build_retrieval_tool_registry(retriever)
    stop_reason: Literal[
        "agent_finish",
        "round_budget",
        "action_budget",
        "agent_error",
    ] = "round_budget"

    for round_number in range(1, max_rounds + 1):
        history = [
            {
                "round": outcome.round_number,
                "action_id": outcome.action.action_id,
                "tool": outcome.action.tool,
                "query_id": outcome.action.query_id,
                "status": outcome.status,
                "new_evidence_ids": list(outcome.new_evidence_ids),
                "errors": list(outcome.errors),
            }
            for turn in turns
            for outcome in turn.outcomes
        ]
        payload = {
            "research_tasks": [
                task.model_dump(mode="json") for task in research_tasks
            ],
            "shared_context": dict(shared_context or {}),
            "corpus_scope": active_scope.model_dump(mode="json"),
            "available_specifications": sorted(available_specs),
            "available_tools": [
                capability.model_dump(mode="json")
                for capability in registry.capabilities
            ],
            "budget": {
                "rounds_remaining_including_this": max_rounds - round_number + 1,
                "search_actions_remaining": max_search_actions - search_used,
                "expansion_actions_remaining": (
                    max_expansion_actions - expansion_used
                ),
                "tool_actions_remaining": (
                    total_action_limit - search_used - expansion_used
                ),
                "hits_per_search_ceiling": hits_per_search,
                "max_evidence_per_question": max_evidence_per_question,
            },
            "history": history,
            "evidence_catalog": ledger.payload(),
        }
        try:
            response = controller_agent.decide(
                input_payload=payload,
                output_type=AgentRetrievalDecision,
            )
        except Exception as exc:
            errors.append(f"controller: {type(exc).__name__}: {exc}")
            stop_reason = "agent_error"
            break

        actions = response.output.actions
        if any(action.tool == "finish" for action in actions):
            if len(actions) != 1 or actions[0].tool != "finish":
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=actions[0],
                    status="rejected",
                    errors=("finish must be the only action in a turn",),
                )
                turns.append(AgentRetrievalTurn(
                    round_number=round_number,
                    assessment=response.output.assessment,
                    receipt=response.receipt,
                    outcomes=(outcome,),
                ))
                errors.extend(outcome.errors)
                continue
            finish_errors = validate_action(
                actions[0],
                query_ids=query_id_set,
                available_specifications=available_specs,
                ledger=ledger,
            )
            outcome = AgentActionOutcome(
                round_number=round_number,
                action=actions[0],
                status="rejected" if finish_errors else "completed",
                errors=finish_errors,
            )
            turns.append(AgentRetrievalTurn(
                round_number=round_number,
                assessment=response.output.assessment,
                receipt=response.receipt,
                outcomes=(outcome,),
            ))
            if finish_errors:
                errors.extend(finish_errors)
                continue
            stop_reason = "agent_finish"
            break

        outcomes: list[AgentActionOutcome] = []
        for action in actions:
            action_errors = list(validate_action(
                action,
                query_ids=query_id_set,
                available_specifications=available_specs,
                ledger=ledger,
            ))
            signature = action_signature(action)
            if signature in signatures:
                action_errors.append("duplicate prior tool call")
            if (
                action.tool not in {"finish", "expand_same_section"}
                and search_used >= max_search_actions
            ):
                action_errors.append("search action budget exhausted")
            if (
                action.tool == "expand_same_section"
                and expansion_used >= max_expansion_actions
            ):
                action_errors.append("expansion action budget exhausted")
            if search_used + expansion_used >= total_action_limit:
                action_errors.append("total tool action budget exhausted")
            if action_errors:
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="rejected",
                    errors=tuple(action_errors),
                )
                outcomes.append(outcome)
                errors.extend(action_errors)
                continue

            signatures.add(signature)
            if action.tool == "expand_same_section":
                expansion_used += 1
            else:
                search_used += 1
            request = _tool_request(
                action,
                round_number=round_number,
                ledger=ledger,
                result_limit=hits_per_search,
            )
            try:
                result = registry.invoke(action.tool, request)
                if result.corpus_scope != active_scope:
                    raise RuntimeError(
                        "tool result corpus scope differs from the active run"
                    )
                planned_hits = _planned_hits_from_tool(
                    result,
                    task=task_by_id[action.query_id],
                )
                observed, added = ledger.add(
                    action.query_id,
                    planned_hits,
                    action_id=action.action_id,
                )
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status=result.status,
                    evidence_ids=observed,
                    new_evidence_ids=added,
                    errors=result.errors,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="unavailable",
                    errors=(message,),
                )
            outcomes.append(outcome)
            errors.extend(outcome.errors)

        turns.append(AgentRetrievalTurn(
            round_number=round_number,
            assessment=response.output.assessment,
            receipt=response.receipt,
            outcomes=tuple(outcomes),
        ))
        if (
            search_used + expansion_used >= total_action_limit
            or search_used >= max_search_actions
            and expansion_used >= max_expansion_actions
        ):
            stop_reason = "action_budget"
            break

    evidence = _evidence_execution(
        research_tasks=research_tasks,
        ledger=ledger,
        retriever=retriever,
        app_config=app_config,
    )
    has_hits = any(task.hits for task in evidence.tasks)
    every_question_has_hits = all(task.hits for task in evidence.tasks)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if not has_hits
        else "partial"
        if errors or not every_question_has_hits
        else "completed"
    )
    return AgenticRetrievalExecution(
        status=status,
        stop_reason=stop_reason,
        rounds_used=len(turns),
        search_actions_used=search_used,
        expansion_actions_used=expansion_used,
        turns=tuple(turns),
        evidence_execution=evidence,
        errors=tuple(errors),
    )


__all__ = [
    "AgentActionOutcome",
    "AgentRetrievalAction",
    "AgentRetrievalDecision",
    "AgentRetrievalTurn",
    "AgenticRetrievalExecution",
    "RegistryResearcherService",
    "run_agentic_retrieval_loop",
]
