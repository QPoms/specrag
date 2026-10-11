"""Model proposal and trusted validation for Researcher retrieval turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.models import ResearchTask
from specrag.retrieval.models import CorpusScope
from specrag.tools import ToolCapability

from .models import (
    AgentActionValidation,
    AgentRetrievalDecision,
    ResearcherActionPlan,
)
from .policy import action_signature, validate_action
from .state import EvidenceLedger


def _controller_payload(
    research_tasks: tuple[ResearchTask, ...],
    *,
    corpus_scope: CorpusScope,
    available_specifications: tuple[str, ...],
    available_tools: tuple[ToolCapability, ...],
    shared_context: Mapping[str, Any],
    round_number: int,
    max_rounds: int,
    search_used: int,
    max_search_actions: int,
    expansion_used: int,
    max_expansion_actions: int,
    total_action_limit: int,
    hits_per_search: int,
    max_evidence_per_question: int,
    history: Sequence[Mapping[str, Any]],
    evidence_catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "research_tasks": [
            task.model_dump(mode="json") for task in research_tasks
        ],
        "shared_context": dict(shared_context),
        "corpus_scope": corpus_scope.model_dump(mode="json"),
        "available_specifications": list(available_specifications),
        "available_tools": [
            capability.model_dump(mode="json")
            for capability in available_tools
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
        "history": [dict(item) for item in history],
        "evidence_catalog": [dict(item) for item in evidence_catalog],
    }


def _validate_decision(
    decision: AgentRetrievalDecision,
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    available_tool_names: frozenset[str],
    ledger: EvidenceLedger,
    prior_signatures: frozenset[tuple[Any, ...]],
    search_used: int,
    max_search_actions: int,
    expansion_used: int,
    max_expansion_actions: int,
    total_action_limit: int,
) -> tuple[AgentActionValidation, ...]:
    contains_finish = any(action.tool == "finish" for action in decision.actions)
    seen_action_ids: set[str] = set()
    signatures = set(prior_signatures)
    simulated_search = search_used
    simulated_expansion = expansion_used
    validations: list[AgentActionValidation] = []

    for action in decision.actions:
        errors = list(validate_action(
            action,
            query_ids=query_ids,
            available_specifications=available_specifications,
            ledger=ledger,
        ))
        if action.tool != "finish" and action.tool not in available_tool_names:
            errors.append(f"tool is not available: {action.tool}")
        if action.action_id in seen_action_ids:
            errors.append("duplicate action_id in the same turn")
        seen_action_ids.add(action.action_id)
        if contains_finish and len(decision.actions) != 1:
            errors.append("finish must be the only action in a turn")

        signature = action_signature(action)
        if signature in signatures:
            errors.append("duplicate prior tool call")
        if action.tool not in {"finish", "expand_same_section"}:
            if simulated_search >= max_search_actions:
                errors.append("search action budget exhausted")
        if action.tool == "expand_same_section":
            if simulated_expansion >= max_expansion_actions:
                errors.append("expansion action budget exhausted")
        if (
            action.tool != "finish"
            and simulated_search + simulated_expansion >= total_action_limit
        ):
            errors.append("total tool action budget exhausted")

        accepted = not errors
        validations.append(AgentActionValidation(
            action_id=action.action_id,
            accepted=accepted,
            errors=tuple(errors),
        ))
        if not accepted or action.tool == "finish":
            continue
        signatures.add(signature)
        if action.tool == "expand_same_section":
            simulated_expansion += 1
        else:
            simulated_search += 1
    return tuple(validations)


def propose_researcher_turn(
    research_tasks: tuple[ResearchTask, ...],
    *,
    controller_agent: StructuredResearcherController,
    corpus_scope: CorpusScope,
    available_specifications: tuple[str, ...],
    available_tools: tuple[ToolCapability, ...],
    ledger: EvidenceLedger,
    shared_context: Mapping[str, Any] | None = None,
    round_number: int = 1,
    max_rounds: int = 4,
    search_used: int = 0,
    max_search_actions: int = 6,
    expansion_used: int = 0,
    max_expansion_actions: int = 3,
    total_action_limit: int = 9,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
    history: Sequence[Mapping[str, Any]] = (),
    prior_signatures: frozenset[tuple[Any, ...]] = frozenset(),
) -> ResearcherActionPlan:
    """Propose and validate one turn without executing any retrieval tool."""

    if not research_tasks:
        raise ValueError("Researcher requires at least one research task")
    task_ids = tuple(task.task_id for task in research_tasks)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Researcher task IDs must be unique")
    payload = _controller_payload(
        research_tasks,
        corpus_scope=corpus_scope,
        available_specifications=available_specifications,
        available_tools=available_tools,
        shared_context=dict(shared_context or {}),
        round_number=round_number,
        max_rounds=max_rounds,
        search_used=search_used,
        max_search_actions=max_search_actions,
        expansion_used=expansion_used,
        max_expansion_actions=max_expansion_actions,
        total_action_limit=total_action_limit,
        hits_per_search=hits_per_search,
        max_evidence_per_question=max_evidence_per_question,
        history=history,
        evidence_catalog=ledger.payload(),
    )
    response = controller_agent.decide(
        input_payload=payload,
        output_type=AgentRetrievalDecision,
    )
    query_ids = frozenset(task_ids)
    validations = _validate_decision(
        response.output,
        query_ids=query_ids,
        available_specifications=frozenset(available_specifications),
        available_tool_names=frozenset(
            capability.name for capability in available_tools
        ),
        ledger=ledger,
        prior_signatures=prior_signatures,
        search_used=search_used,
        max_search_actions=max_search_actions,
        expansion_used=expansion_used,
        max_expansion_actions=max_expansion_actions,
        total_action_limit=total_action_limit,
    )
    return ResearcherActionPlan(
        input_payload=payload,
        decision=response.output,
        receipt=response.receipt,
        validations=validations,
        accepted=all(item.accepted for item in validations),
    )


@dataclass(frozen=True)
class ResearcherPlanningService:
    """Generate an auditable first turn without building or invoking tools."""

    controller_agent: StructuredResearcherController
    corpus_scope: CorpusScope
    available_specifications: tuple[str, ...]
    available_tools: tuple[ToolCapability, ...]
    max_rounds: int = 4
    max_search_actions: int = 6
    max_expansion_actions: int = 3
    max_total_actions: int = 9
    hits_per_search: int = 8
    max_evidence_per_question: int = 28

    def propose_first_turn(
        self,
        tasks: tuple[ResearchTask, ...],
        *,
        shared_context: Mapping[str, Any] | None = None,
    ) -> ResearcherActionPlan:
        ledger = EvidenceLedger(
            tuple(task.task_id for task in tasks),
            max_per_question=self.max_evidence_per_question,
        )
        return propose_researcher_turn(
            tasks,
            controller_agent=self.controller_agent,
            corpus_scope=self.corpus_scope,
            available_specifications=self.available_specifications,
            available_tools=self.available_tools,
            ledger=ledger,
            shared_context=shared_context,
            max_rounds=self.max_rounds,
            max_search_actions=self.max_search_actions,
            max_expansion_actions=self.max_expansion_actions,
            total_action_limit=self.max_total_actions,
            hits_per_search=self.hits_per_search,
            max_evidence_per_question=self.max_evidence_per_question,
        )


__all__ = [
    "ResearcherPlanningService",
    "propose_researcher_turn",
]
