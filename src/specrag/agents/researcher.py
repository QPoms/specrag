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
    PlannedRetrievalTaskResult,
)
from specrag.tools import (
    RetrievalToolRequest,
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
from .planning import propose_researcher_turn
from .policy import action_signature
from .state import EvidenceLedger
from .tool_results import planned_hits_from_tool


@dataclass
class RegistryResearcherService:
    """Framework-independent application service for one Researcher worker."""

    controller_agent: StructuredResearcherController
    retriever: HybridMarkdownRetriever
    app_config: AppConfig
    tool_registry: ToolRegistry | None = None
    memory_reader: Any | None = None
    max_rounds: int = 4
    max_total_actions: int = 9
    hits_per_search: int = 8
    max_evidence_per_question: int = 28
    max_finalization_turns: int = 1

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
            max_total_actions=(
                max_tool_calls
                if max_tool_calls is not None
                else self.max_total_actions
            ),
            hits_per_search=self.hits_per_search,
            max_evidence_per_question=self.max_evidence_per_question,
            max_finalization_turns=self.max_finalization_turns,
        )
        return execution, {task.task_id: task.task_id for task in tasks}


def build_retrieval_tool_request(
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
        hypothetical_document=action.hypothetical_document or None,
        target_specifications=action.target_specifications,
        clause=action.clause or None,
        seed_chunk_ids=seeds,
        limit=min(action.max_results, result_limit),
    )


def _evidence_execution(
    *,
    research_tasks: tuple[ResearchTask, ...],
    ledger: EvidenceLedger,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    candidate_pool: bool = False,
) -> PlannedRetrievalExecution:
    active_scope = effective_corpus_scope(app_config.retrieval, retriever)
    tasks: list[PlannedRetrievalTaskResult] = []
    for task in research_tasks:
        hits = (
            ledger.candidate_hits(task.task_id)
            if candidate_pool
            else ledger.active_hits(task.task_id)
        )
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
    max_total_actions: int = 9,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
    max_finalization_turns: int = 1,
) -> AgenticRetrievalExecution:
    """Let an agent choose bounded calls to trusted local retrieval tools."""

    if not research_tasks:
        raise ValueError("Researcher requires at least one research task")

    for name, value in (
        ("max_rounds", max_rounds),
        ("max_total_actions", max_total_actions),
        ("hits_per_search", hits_per_search),
        ("max_evidence_per_question", max_evidence_per_question),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if max_finalization_turns not in {0, 1}:
        raise ValueError("max_finalization_turns must be zero or one")
    total_action_limit = max_total_actions

    task_by_id = {task.task_id: task for task in research_tasks}
    if len(task_by_id) != len(research_tasks):
        raise ValueError("Researcher task IDs must be unique")
    query_ids = tuple(task_by_id)
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
        "submitted_for_validation",
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
        try:
            proposal = propose_researcher_turn(
                research_tasks,
                controller_agent=controller_agent,
                corpus_scope=active_scope,
                available_specifications=tuple(sorted(available_specs)),
                available_tools=registry.capabilities,
                ledger=ledger,
                shared_context=shared_context,
                round_number=round_number,
                max_rounds=max_rounds,
                actions_used=search_used + expansion_used,
                total_action_limit=total_action_limit,
                hits_per_search=hits_per_search,
                max_evidence_per_question=max_evidence_per_question,
                history=history,
                prior_signatures=frozenset(signatures),
            )
        except Exception as exc:
            errors.append(f"controller: {type(exc).__name__}: {exc}")
            stop_reason = "agent_error"
            break

        actions = proposal.decision.actions
        validations = proposal.validations
        selection_errors = tuple(
            error
            for validation in proposal.selection_validations
            for error in validation.errors
        )
        if selection_errors:
            outcomes = tuple(
                AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="rejected",
                    errors=tuple((*selection_errors, *validation.errors)),
                )
                for action, validation in zip(
                    actions,
                    validations,
                    strict=True,
                )
            )
            turns.append(AgentRetrievalTurn(
                round_number=round_number,
                assessment=proposal.decision.assessment,
                evidence_selections=proposal.decision.evidence_selections,
                selection_validations=proposal.selection_validations,
                receipt=proposal.receipt,
                outcomes=outcomes,
            ))
            errors.extend(selection_errors)
            errors.extend(
                error for validation in validations for error in validation.errors
            )
            continue

        for selection in proposal.decision.evidence_selections:
            ledger.select(
                selection.query_id,
                selection.retained_evidence_ids,
            )
        if any(
            action.tool == "submit_for_validation" for action in actions
        ):
            outcomes = tuple(
                AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="completed" if validation.accepted else "rejected",
                    errors=validation.errors,
                )
                for action, validation in zip(
                    actions,
                    validations,
                    strict=True,
                )
            )
            turns.append(AgentRetrievalTurn(
                round_number=round_number,
                assessment=proposal.decision.assessment,
                evidence_selections=proposal.decision.evidence_selections,
                selection_validations=proposal.selection_validations,
                receipt=proposal.receipt,
                outcomes=outcomes,
            ))
            turn_errors = tuple(
                error for validation in validations for error in validation.errors
            )
            if turn_errors:
                errors.extend(turn_errors)
                continue
            if (
                len(actions) == 1
                and actions[0].tool == "submit_for_validation"
            ):
                stop_reason = "submitted_for_validation"
                break
            continue

        outcomes: list[AgentActionOutcome] = []
        for action, validation in zip(actions, validations, strict=True):
            if not validation.accepted:
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="rejected",
                    errors=validation.errors,
                )
                outcomes.append(outcome)
                errors.extend(validation.errors)
                continue

            signatures.add(action_signature(action))
            if action.tool == "expand_source_context":
                expansion_used += 1
            else:
                search_used += 1
            request = build_retrieval_tool_request(
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
                planned_hits = planned_hits_from_tool(
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
            assessment=proposal.decision.assessment,
            evidence_selections=proposal.decision.evidence_selections,
            selection_validations=proposal.selection_validations,
            receipt=proposal.receipt,
            outcomes=tuple(outcomes),
        ))
        if search_used + expansion_used >= total_action_limit:
            stop_reason = "action_budget"
            break

    has_pending_candidates = any(
        ledger.pending_evidence_ids(query_id) for query_id in query_ids
    )
    if (
        max_finalization_turns == 1
        and stop_reason in {"round_budget", "action_budget"}
        and has_pending_candidates
    ):
        finalization_round = len(turns) + 1
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
        try:
            proposal = propose_researcher_turn(
                research_tasks,
                controller_agent=controller_agent,
                corpus_scope=active_scope,
                available_specifications=tuple(sorted(available_specs)),
                available_tools=registry.capabilities,
                ledger=ledger,
                shared_context=shared_context,
                round_number=finalization_round,
                max_rounds=max_rounds,
                actions_used=search_used + expansion_used,
                total_action_limit=total_action_limit,
                hits_per_search=hits_per_search,
                max_evidence_per_question=max_evidence_per_question,
                history=history,
                prior_signatures=frozenset(signatures),
                finalization_only=True,
            )
        except Exception as exc:
            errors.append(f"controller: {type(exc).__name__}: {exc}")
            stop_reason = "agent_error"
        else:
            selection_errors = tuple(
                error
                for validation in proposal.selection_validations
                for error in validation.errors
            )
            action_errors = tuple(
                error
                for validation in proposal.validations
                for error in validation.errors
            )
            accepted = not selection_errors and not action_errors
            if accepted:
                for selection in proposal.decision.evidence_selections:
                    ledger.select(
                        selection.query_id,
                        selection.retained_evidence_ids,
                    )
            outcomes = tuple(
                AgentActionOutcome(
                    round_number=finalization_round,
                    action=action,
                    status=(
                        "completed"
                        if accepted and validation.accepted
                        else "rejected"
                    ),
                    errors=tuple((
                        *selection_errors,
                        *validation.errors,
                    )),
                )
                for action, validation in zip(
                    proposal.decision.actions,
                    proposal.validations,
                    strict=True,
                )
            )
            turns.append(AgentRetrievalTurn(
                round_number=finalization_round,
                phase="finalization",
                assessment=proposal.decision.assessment,
                evidence_selections=proposal.decision.evidence_selections,
                selection_validations=proposal.selection_validations,
                receipt=proposal.receipt,
                outcomes=outcomes,
            ))
            errors.extend(selection_errors)
            errors.extend(action_errors)
            if (
                accepted
                and len(proposal.decision.actions) == 1
                and proposal.decision.actions[0].tool
                == "submit_for_validation"
            ):
                stop_reason = "submitted_for_validation"

    evidence = _evidence_execution(
        research_tasks=research_tasks,
        ledger=ledger,
        retriever=retriever,
        app_config=app_config,
    )
    candidate_evidence = _evidence_execution(
        research_tasks=research_tasks,
        ledger=ledger,
        retriever=retriever,
        app_config=app_config,
        candidate_pool=True,
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
        rounds_used=sum(turn.phase == "retrieval" for turn in turns),
        search_actions_used=search_used,
        expansion_actions_used=expansion_used,
        turns=tuple(turns),
        evidence_execution=evidence,
        candidate_evidence_execution=candidate_evidence,
        errors=tuple(errors),
    )


__all__ = [
    "AgentActionOutcome",
    "AgentRetrievalAction",
    "AgentRetrievalDecision",
    "AgentRetrievalTurn",
    "AgenticRetrievalExecution",
    "RegistryResearcherService",
    "build_retrieval_tool_request",
    "run_agentic_retrieval_loop",
]
