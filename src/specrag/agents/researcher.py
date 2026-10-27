"""Bounded Researcher loop over native trusted retrieval function calls."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from specrag.config import AppConfig
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.models import (
    CandidateEvidence,
    EvidenceRequirement,
    NativeFunctionCallOutput,
    ResearchTask,
)
from specrag.retrieval.hybrid import HybridMarkdownRetriever, effective_corpus_scope
from specrag.retrieval.planner import PlannedRetrievalExecution, PlannedRetrievalTaskResult
from specrag.tools import (
    ClauseLookupArguments,
    ClauseNavigationArguments,
    CrossReferenceLookupArguments,
    ExpandSourceContextArguments,
    HybridSearchArguments,
    HyDESearchArguments,
    KeywordSearchArguments,
    RerankArguments,
    RetrievalToolRequest,
    SemanticSearchArguments,
    ToolRegistry,
    build_retrieval_tool_registry,
    parse_retrieval_tool_arguments,
)
from .models import AgentRetrievalCall, AgentRetrievalTurn, AgentToolOutcome, AgenticRetrievalExecution
from .planning import propose_researcher_turn
from .policy import call_signature
from .state import EvidenceLedger
from .tool_results import native_tool_outputs, planned_hits_from_tool, research_history

@dataclass
class RegistryResearcherService:
    """Framework-independent application service for one Researcher worker."""

    controller_agent: StructuredResearcherController
    retriever: HybridMarkdownRetriever
    app_config: AppConfig
    tool_registry: ToolRegistry | None = None
    research_context: tuple[dict[str, object], ...] = ()
    max_rounds: int = 4
    max_total_actions: int = 9
    hits_per_search: int = 8
    max_evidence_per_question: int = 28
    max_finalization_turns: int = 1
    stop_requested: Callable[[str], Literal[
        "cancelled", "elapsed_time_budget"
    ] | None] | None = None
    tool_result_sink: Callable[[str, str, Any], str] | None = None
    def run(
        self,
        tasks: tuple[ResearchTask, ...],
        *,
        shared_context: Mapping[str, Any] | None = None,
        max_tool_calls: int | None = None,
        initial_evidence: tuple[CandidateEvidence, ...] = (),
        initial_active_evidence_ids: tuple[str, ...] = (),
        initial_evidence_requirements: tuple[EvidenceRequirement, ...] = (),
    ) -> tuple[AgenticRetrievalExecution, dict[str, str]]:
        if not tasks:
            raise ValueError("Researcher requires at least one research task")
        execution = run_agentic_retrieval_loop(
            tasks,
            controller_agent=self.controller_agent,
            retriever=self.retriever,
            app_config=self.app_config,
            tool_registry=self.tool_registry,
            shared_context={
                **dict(shared_context or {}),
                "authorized_research_context": list(self.research_context),
            },
            initial_evidence=initial_evidence,
            initial_active_evidence_ids=initial_active_evidence_ids,
            initial_evidence_requirements=initial_evidence_requirements,
            max_rounds=self.max_rounds,
            max_total_actions=(
                max_tool_calls
                if max_tool_calls is not None
                else self.max_total_actions
            ),
            hits_per_search=self.hits_per_search,
            max_evidence_per_question=self.max_evidence_per_question,
            max_finalization_turns=self.max_finalization_turns,
            stop_requested=(lambda: self.stop_requested(str((
                shared_context or {}
            ).get("research_id", "")))) if self.stop_requested is not None else None,
            tool_result_sink=self.tool_result_sink,
        )
        return execution, {task.task_id: task.task_id for task in tasks}
def build_retrieval_tool_request(
    call: AgentRetrievalCall,
    *,
    round_number: int,
    ledger: EvidenceLedger,
    result_limit: int,
) -> RetrievalToolRequest:
    arguments = parse_retrieval_tool_arguments(call.tool, call.arguments)
    seed_evidence_ids = tuple(
        getattr(arguments, "seed_evidence_ids", ())
    )
    seeds = tuple(
        hit.chunk_id
        for evidence_id in seed_evidence_ids
        if (hit := ledger.find(arguments.query_id, evidence_id)) is not None
    )
    query = ""
    semantic_query: str | None = None
    hypothetical_document: str | None = None
    clause: str | None = None
    clause_relation = None
    reference_specification = None
    reference_version = None
    reference_clause = None
    include_descendants = False
    cursor = 0
    if isinstance(arguments, KeywordSearchArguments):
        query = arguments.lexical_query
    elif isinstance(arguments, SemanticSearchArguments):
        semantic_query = arguments.semantic_query
    elif isinstance(arguments, HybridSearchArguments):
        query = arguments.lexical_query
        semantic_query = arguments.semantic_query
    elif isinstance(arguments, HyDESearchArguments):
        query = arguments.lexical_query
        hypothetical_document = arguments.hypothetical_document
    elif isinstance(arguments, RerankArguments):
        semantic_query = arguments.semantic_query
    elif isinstance(arguments, ClauseLookupArguments):
        clause = arguments.clause
        include_descendants = arguments.include_descendants
        cursor = arguments.cursor
    elif isinstance(arguments, ClauseNavigationArguments):
        clause_relation = arguments.relation
        cursor = arguments.cursor
    elif isinstance(arguments, CrossReferenceLookupArguments):
        reference_specification = arguments.target_specification
        reference_version = arguments.target_version
        reference_clause = arguments.target_clause
        cursor = arguments.cursor
    elif not isinstance(arguments, ExpandSourceContextArguments):
        raise TypeError(f"unsupported retrieval arguments: {type(arguments).__name__}")
    return RetrievalToolRequest(
        request_id=(
            f"{arguments.query_id}:agent:r{round_number}:{call.call_id}"
        ),
        query_id=arguments.query_id,
        query=query,
        semantic_query=semantic_query,
        hypothetical_document=hypothetical_document,
        target_specifications=tuple(
            getattr(arguments, "target_specifications", ())
        ),
        clause=clause,
        clause_relation=clause_relation,
        reference_specification=reference_specification,
        reference_version=reference_version,
        reference_clause=reference_clause,
        include_descendants=include_descendants,
        cursor=cursor,
        seed_chunk_ids=seeds,
        limit=min(arguments.max_results, result_limit),
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
def _rejected_outcomes(
    proposal: Any,
    *,
    round_number: int,
    shared_errors: tuple[str, ...],
) -> tuple[AgentToolOutcome, ...]:
    return tuple(
        AgentToolOutcome(
            round_number=round_number,
            call=call,
            status="rejected",
            errors=tuple((*shared_errors, *validation.errors)),
        )
        for call, validation in zip(
            proposal.retrieval_calls,
            proposal.validations,
            strict=True,
        )
    )


def run_agentic_retrieval_loop(
    research_tasks: tuple[ResearchTask, ...],
    *,
    controller_agent: StructuredResearcherController,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    tool_registry: ToolRegistry | None = None,
    shared_context: Mapping[str, Any] | None = None,
    initial_evidence: tuple[CandidateEvidence, ...] = (),
    initial_active_evidence_ids: tuple[str, ...] = (),
    initial_evidence_requirements: tuple[EvidenceRequirement, ...] = (),
    max_rounds: int = 4,
    max_total_actions: int = 9,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
    max_finalization_turns: int = 1,
    stop_requested: Callable[[], Literal[
        "cancelled", "elapsed_time_budget"
    ] | None] | None = None,
    tool_result_sink: Callable[[str, str, Any], str] | None = None,
) -> AgenticRetrievalExecution:
    """Let a model issue native calls to bounded local retrieval tools."""

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
    ledger.seed_candidates(
        initial_evidence,
        active_evidence_ids=initial_active_evidence_ids,
    )
    evidence_requirements = tuple(initial_evidence_requirements)
    turns: list[AgentRetrievalTurn] = []
    errors: list[str] = []
    signatures: set[tuple[Any, ...]] = set()
    search_used = 0
    expansion_used = 0
    registry = tool_registry or build_retrieval_tool_registry(retriever)
    prior_response_items: tuple[dict[str, Any], ...] = ()
    prior_outputs: tuple[NativeFunctionCallOutput, ...] = ()
    stop_reason: Literal[
        "submitted_for_validation",
        "round_budget",
        "action_budget",
        "agent_error",
        "cancelled",
        "elapsed_time_budget",
    ] = "round_budget"

    for round_number in range(1, max_rounds + 1):
        if stop_requested is not None and (requested := stop_requested()) is not None:
            stop_reason = requested
            break
        try:
            proposal = propose_researcher_turn(
                research_tasks,
                controller_agent=controller_agent,
                corpus_scope=active_scope,
                available_specifications=tuple(sorted(available_specs)),
                available_tools=registry.definitions,
                ledger=ledger,
                evidence_requirements=evidence_requirements,
                shared_context=shared_context,
                round_number=round_number,
                max_rounds=max_rounds,
                actions_used=search_used + expansion_used,
                total_action_limit=max_total_actions,
                hits_per_search=hits_per_search,
                max_evidence_per_question=max_evidence_per_question,
                history=research_history(turns),
                prior_signatures=frozenset(signatures),
                prior_response_items=prior_response_items,
                prior_outputs=prior_outputs,
            )
        except Exception as exc:
            errors.append(f"controller: {type(exc).__name__}: {exc}")
            stop_reason = "agent_error"
            break

        selection_errors = tuple(
            error
            for validation in proposal.selection_validations
            for error in validation.errors
        )
        control_errors = tuple((*proposal.protocol_errors, *selection_errors))
        if control_errors:
            outcomes = _rejected_outcomes(
                proposal,
                round_number=round_number,
                shared_errors=control_errors,
            )
            turns.append(AgentRetrievalTurn(
                round_number=round_number,
                control_call=proposal.control_call,
                selection_validations=proposal.selection_validations,
                receipt=proposal.receipt,
                outcomes=outcomes,
                protocol_errors=proposal.protocol_errors,
            ))
            errors.extend(control_errors)
            errors.extend(
                error for validation in proposal.validations
                for error in validation.errors
            )
            prior_response_items = proposal.replay_items
            prior_outputs = native_tool_outputs(
                proposal,
                outcomes,
                control_errors=control_errors,
            )
            continue

        if proposal.control_call is None:  # guarded by protocol validation
            raise RuntimeError("accepted native turn has no control call")
        review = proposal.control_call.review
        for selection in review.evidence_selections:
            ledger.select(
                selection.query_id,
                selection.retained_evidence_ids,
            )
        evidence_requirements = review.evidence_requirements
        if proposal.control_call.name == "submit_for_validation":
            turns.append(AgentRetrievalTurn(
                round_number=round_number,
                control_call=proposal.control_call,
                selection_validations=proposal.selection_validations,
                receipt=proposal.receipt,
                outcomes=(),
            ))
            stop_reason = "submitted_for_validation"
            break

        outcomes: list[AgentToolOutcome] = []
        for call, validation in zip(
            proposal.retrieval_calls,
            proposal.validations,
            strict=True,
        ):
            if stop_requested is not None and (requested := stop_requested()) is not None:
                stop_reason = requested
                break
            if not validation.accepted:
                outcome = AgentToolOutcome(
                    round_number=round_number,
                    call=call,
                    status="rejected",
                    errors=validation.errors,
                )
                outcomes.append(outcome)
                errors.extend(validation.errors)
                continue
            arguments = parse_retrieval_tool_arguments(
                call.tool,
                call.arguments,
            )
            signatures.add(call_signature(call, arguments))
            if call.tool in {
                "expand_source_context",
                "navigate_clause",
                "cross_reference_lookup",
            }:
                expansion_used += 1
            else:
                search_used += 1
            request = build_retrieval_tool_request(
                call,
                round_number=round_number,
                ledger=ledger,
                result_limit=hits_per_search,
            )
            try:
                result = registry.invoke(call.tool, request)
                if result.corpus_scope != active_scope:
                    raise RuntimeError(
                        "tool result corpus scope differs from the active run"
                    )
                tool_result_id = None
                context_id = str((shared_context or {}).get("research_id", ""))
                if tool_result_sink is not None and context_id:
                    tool_result_id = tool_result_sink(
                        context_id, call.call_id, result.model_dump(mode="json")
                    )
                planned_hits = planned_hits_from_tool(
                    result,
                    task=task_by_id[request.query_id],
                )
                observed, added = ledger.add(
                    request.query_id,
                    planned_hits,
                    action_id=call.call_id,
                )
                outcome = AgentToolOutcome(
                    round_number=round_number,
                    call=call,
                    status=result.status,
                    evidence_ids=observed,
                    new_evidence_ids=added,
                    errors=result.errors,
                    tool_result_id=tool_result_id,
                    next_cursor=result.next_cursor,
                )
            except Exception as exc:
                outcome = AgentToolOutcome(
                    round_number=round_number,
                    call=call,
                    status="unavailable",
                    errors=(f"{type(exc).__name__}: {exc}",),
                )
            outcomes.append(outcome)
            errors.extend(outcome.errors)

        outcome_tuple = tuple(outcomes)
        turns.append(AgentRetrievalTurn(
            round_number=round_number,
            control_call=proposal.control_call,
            selection_validations=proposal.selection_validations,
            receipt=proposal.receipt,
            outcomes=outcome_tuple,
        ))
        prior_response_items = proposal.replay_items
        prior_outputs = native_tool_outputs(proposal, outcome_tuple)
        if stop_reason in {"cancelled", "elapsed_time_budget"}:
            break
        if search_used + expansion_used >= max_total_actions:
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
        try:
            proposal = propose_researcher_turn(
                research_tasks,
                controller_agent=controller_agent,
                corpus_scope=active_scope,
                available_specifications=tuple(sorted(available_specs)),
                available_tools=registry.definitions,
                ledger=ledger,
                evidence_requirements=evidence_requirements,
                shared_context=shared_context,
                round_number=finalization_round,
                max_rounds=max_rounds,
                actions_used=search_used + expansion_used,
                total_action_limit=max_total_actions,
                hits_per_search=hits_per_search,
                max_evidence_per_question=max_evidence_per_question,
                history=research_history(turns),
                prior_signatures=frozenset(signatures),
                prior_response_items=prior_response_items,
                prior_outputs=prior_outputs,
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
            final_errors = tuple((*proposal.protocol_errors, *selection_errors))
            accepted = not final_errors and proposal.control_call is not None
            if accepted:
                for selection in proposal.control_call.review.evidence_selections:
                    ledger.select(
                        selection.query_id,
                        selection.retained_evidence_ids,
                    )
                evidence_requirements = (
                    proposal.control_call.review.evidence_requirements
                )
            turns.append(AgentRetrievalTurn(
                round_number=finalization_round,
                phase="finalization",
                control_call=proposal.control_call,
                selection_validations=proposal.selection_validations,
                receipt=proposal.receipt,
                outcomes=(),
                protocol_errors=proposal.protocol_errors,
            ))
            errors.extend(final_errors)
            if (
                accepted
                and proposal.control_call.name == "submit_for_validation"
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
        evidence_requirements=evidence_requirements,
        turns=tuple(turns),
        evidence_execution=evidence,
        candidate_evidence_execution=candidate_evidence,
        errors=tuple(errors),
    )


__all__ = [
    "AgentRetrievalCall",
    "AgentRetrievalTurn",
    "AgentToolOutcome",
    "AgenticRetrievalExecution",
    "RegistryResearcherService",
    "build_retrieval_tool_request",
    "run_agentic_retrieval_loop",
]
