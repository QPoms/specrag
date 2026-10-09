"""Bounded Researcher loop over trusted local retrieval capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from specrag.config import AppConfig
from specrag.infrastructure.structured_agent import CodexStructuredAgent
from specrag.retrieval.hypothesis import merge_parallel_retrieval_channels
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.planner import (
    DiscoveryRetrievalPlanV3,
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
    execute_planned_discovery_retrieval,
)

from .models import (
    AgentActionOutcome,
    AgentRetrievalAction,
    AgentRetrievalDecision,
    AgentRetrievalTurn,
    AgenticRetrievalExecution,
)
from .policy import (
    CONTROLLER_INSTRUCTIONS,
    action_signature,
    build_search_plan,
    validate_action,
)
from .state import EvidenceLedger


def _expand_same_section(
    *,
    query_id: str,
    seeds: tuple[PlannedRetrievalHit, ...],
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    max_results: int,
) -> tuple[PlannedRetrievalHit, ...]:
    task = PlannedRetrievalTaskResult(
        retrieval_id=f"{query_id}:agent:adjacent",
        query_id=query_id,
        planner_action="planned",
        status="completed",
        hits=seeds,
    )
    literal = PlannedRetrievalExecution(
        status="completed",
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=app_config.retrieval.embedding_model,
        corpus_version_policy=app_config.retrieval.version_policy,
        corpus_release=app_config.retrieval.target_release,
        tasks=(task,),
    )
    expanded = merge_parallel_retrieval_channels(
        literal,
        None,
        retriever=retriever,
        route_a_quota=len(seeds),
        route_b_quota=0,
        max_hits_per_question=len(seeds) + max_results,
        adjacent_chunk_budget=max_results,
    )
    seed_chunks = {hit.chunk_id for hit in seeds}
    return tuple(
        hit for hit in expanded.tasks[0].hits if hit.chunk_id not in seed_chunks
    )


def _evidence_execution(
    *,
    base_plan: DiscoveryRetrievalPlanV3,
    ledger: EvidenceLedger,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
) -> PlannedRetrievalExecution:
    tasks: list[PlannedRetrievalTaskResult] = []
    for item in base_plan.retrievals:
        hits = ledger.hits(item.query_id)
        tasks.append(PlannedRetrievalTaskResult(
            retrieval_id=item.retrieval_id,
            query_id=item.query_id,
            planner_action=item.planner_action,
            status="completed" if hits else "no_hits",
            hits=hits,
        ))
    has_hits = any(task.hits for task in tasks)
    return PlannedRetrievalExecution(
        status="completed" if has_hits else "unavailable",
        resolved_plan=base_plan,
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=app_config.retrieval.embedding_model,
        corpus_version_policy=app_config.retrieval.version_policy,
        corpus_release=app_config.retrieval.target_release,
        tasks=tuple(tasks),
    )


def run_agentic_retrieval_loop(
    base_plan: DiscoveryRetrievalPlanV3,
    *,
    controller_agent: CodexStructuredAgent,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    shared_context: Mapping[str, Any] | None = None,
    max_rounds: int = 4,
    max_search_actions: int = 6,
    max_expansion_actions: int = 3,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
) -> AgenticRetrievalExecution:
    """Let an agent choose bounded calls to trusted local retrieval tools."""

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

    base_by_query = {item.query_id: item for item in base_plan.retrievals}
    query_ids = tuple(base_by_query)
    query_id_set = frozenset(query_ids)
    available_specs = frozenset(
        document.specification
        for document in retriever.documents
        if document.specification is not None
    )
    ledger = EvidenceLedger(
        query_ids,
        max_per_question=max_evidence_per_question,
    )
    turns: list[AgentRetrievalTurn] = []
    errors: list[str] = []
    signatures: set[tuple[Any, ...]] = set()
    search_used = 0
    expansion_used = 0
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
            "questions": [
                {
                    "query_id": item.query_id,
                    "claim_id": item.claim_id,
                    "angle": item.angle,
                    "question": item.original_question,
                }
                for item in base_plan.retrievals
            ],
            "shared_context": dict(shared_context or {}),
            "available_specifications": sorted(available_specs),
            "budget": {
                "rounds_remaining_including_this": max_rounds - round_number + 1,
                "search_actions_remaining": max_search_actions - search_used,
                "expansion_actions_remaining": (
                    max_expansion_actions - expansion_used
                ),
                "hits_per_search_ceiling": hits_per_search,
                "max_evidence_per_question": max_evidence_per_question,
            },
            "history": history,
            "evidence_catalog": ledger.payload(),
        }
        try:
            response = controller_agent.parse(
                purpose="choose the next bounded Discovery retrieval tool calls",
                prompt_version="agentic-discovery-retrieval-controller/v0.1",
                instructions=CONTROLLER_INSTRUCTIONS,
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
            if action.tool == "hybrid_search" and search_used >= max_search_actions:
                action_errors.append("search action budget exhausted")
            if (
                action.tool == "expand_same_section"
                and expansion_used >= max_expansion_actions
            ):
                action_errors.append("expansion action budget exhausted")
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
            if action.tool == "hybrid_search":
                search_used += 1
                search_plan = build_search_plan(
                    base_by_query[action.query_id],
                    action,
                    round_number=round_number,
                )
                result = execute_planned_discovery_retrieval(
                    search_plan,
                    app_config=app_config,
                    final_hits=min(action.max_results, hits_per_search),
                    retriever=retriever,
                )
                task = result.tasks[0]
                observed, added = ledger.add(
                    action.query_id,
                    task.hits,
                    action_id=action.action_id,
                )
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status=(
                        "completed"
                        if observed
                        else "unavailable"
                        if task.status == "unavailable"
                        else "no_hits"
                    ),
                    evidence_ids=observed,
                    new_evidence_ids=added,
                    errors=task.errors,
                )
            else:
                expansion_used += 1
                seeds = tuple(
                    hit
                    for evidence_id in action.seed_evidence_ids
                    if (hit := ledger.find(action.query_id, evidence_id)) is not None
                )
                expanded_hits = _expand_same_section(
                    query_id=action.query_id,
                    seeds=seeds,
                    retriever=retriever,
                    app_config=app_config,
                    max_results=action.max_results,
                )
                observed, added = ledger.add(
                    action.query_id,
                    expanded_hits,
                    action_id=action.action_id,
                )
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="completed" if observed else "no_hits",
                    evidence_ids=observed,
                    new_evidence_ids=added,
                )
            outcomes.append(outcome)
            errors.extend(outcome.errors)

        turns.append(AgentRetrievalTurn(
            round_number=round_number,
            assessment=response.output.assessment,
            receipt=response.receipt,
            outcomes=tuple(outcomes),
        ))
        if search_used >= max_search_actions and expansion_used >= max_expansion_actions:
            stop_reason = "action_budget"
            break

    evidence = _evidence_execution(
        base_plan=base_plan,
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
    "run_agentic_retrieval_loop",
]
