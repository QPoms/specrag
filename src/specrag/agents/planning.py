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
    AgentEvidenceSelectionValidation,
    AgentRetrievalDecision,
    ResearcherActionPlan,
)
from .policy import action_signature, validate_action
from .state import EvidenceLedger


def _text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text
            for item in value.values()
            for text in _text_values(item)
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(text for item in value for text in _text_values(item))
    return ()


def _trusted_action_context(
    task: ResearchTask,
    *,
    shared_context: Mapping[str, Any],
    ledger: EvidenceLedger,
    available_specifications: frozenset[str],
) -> tuple[frozenset[str], tuple[str, ...]]:
    texts = [
        *_text_values(task.model_dump(mode="json")),
        *_text_values(shared_context),
    ]
    observed_specifications: set[str] = set()
    for hit in ledger.hits(task.task_id):
        texts.extend(hit.heading_path)
        texts.append(hit.excerpt)
        if hit.specification:
            observed_specifications.add(hit.specification)
    explicit_specifications = {
        specification
        for specification in available_specifications
        if any(specification in text for text in texts)
    }
    return (
        frozenset(explicit_specifications | observed_specifications),
        tuple(texts),
    )


def _controller_payload(
    research_tasks: tuple[ResearchTask, ...],
    *,
    corpus_scope: CorpusScope,
    available_specifications: tuple[str, ...],
    available_tools: tuple[ToolCapability, ...],
    shared_context: Mapping[str, Any],
    round_number: int,
    max_rounds: int,
    actions_used: int,
    total_action_limit: int,
    hits_per_search: int,
    max_evidence_per_question: int,
    history: Sequence[Mapping[str, Any]],
    evidence_views: Mapping[str, Sequence[Mapping[str, Any]]],
    finalization_only: bool,
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
            "rounds_remaining_including_this": max(
                max_rounds - round_number + 1,
                0,
            ),
            "tool_actions_remaining": total_action_limit - actions_used,
            "hits_per_search_ceiling": hits_per_search,
            "max_active_evidence_per_question": max_evidence_per_question,
        },
        "execution_mode": (
            "finalization_only" if finalization_only else "retrieval"
        ),
        "history": [dict(item) for item in history],
        "active_evidence": [
            dict(item) for item in evidence_views["active_evidence"]
        ],
        "pending_candidates": [
            dict(item) for item in evidence_views["pending_candidates"]
        ],
        "inactive_candidates": [
            dict(item) for item in evidence_views["inactive_candidates"]
        ],
    }


def _validate_evidence_selections(
    decision: AgentRetrievalDecision,
    *,
    query_ids: frozenset[str],
    ledger: EvidenceLedger,
) -> tuple[AgentEvidenceSelectionValidation, ...]:
    required_query_ids = frozenset(ledger.query_ids_with_candidates())
    seen_query_ids: set[str] = set()
    validations: list[AgentEvidenceSelectionValidation] = []

    for selection in decision.evidence_selections:
        errors: list[str] = []
        if selection.query_id not in query_ids:
            errors.append(f"unknown selection query_id: {selection.query_id}")
        if selection.query_id in seen_query_ids:
            errors.append("duplicate evidence selection for query_id")
        seen_query_ids.add(selection.query_id)
        retained = selection.retained_evidence_ids
        if len(set(retained)) != len(retained):
            errors.append("retained_evidence_ids contains duplicates")
        if len(retained) > ledger.max_per_question:
            errors.append(
                "retained_evidence_ids exceeds max active evidence limit: "
                f"{ledger.max_per_question}"
            )
        if selection.query_id in query_ids:
            unknown = tuple(
                evidence_id
                for evidence_id in retained
                if ledger.find(selection.query_id, evidence_id) is None
            )
            if unknown:
                errors.append(
                    "retained_evidence_ids contains unknown evidence: "
                    + ", ".join(unknown)
                )
        validations.append(AgentEvidenceSelectionValidation(
            query_id=selection.query_id,
            accepted=not errors,
            errors=tuple(errors),
        ))

    missing = required_query_ids - seen_query_ids
    for query_id in sorted(missing):
        validations.append(AgentEvidenceSelectionValidation(
            query_id=query_id,
            accepted=False,
            errors=(
                "evidence selection is required when candidates are present",
            ),
        ))
    return tuple(validations)


def _validate_decision(
    decision: AgentRetrievalDecision,
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    available_tool_names: frozenset[str],
    ledger: EvidenceLedger,
    trusted_context_by_query: Mapping[
        str,
        tuple[frozenset[str], tuple[str, ...]],
    ],
    prior_signatures: frozenset[tuple[Any, ...]],
    actions_used: int,
    total_action_limit: int,
    finalization_only: bool,
) -> tuple[AgentActionValidation, ...]:
    contains_submission = any(
        action.tool == "submit_for_validation"
        for action in decision.actions
    )
    seen_action_ids: set[str] = set()
    signatures = set(prior_signatures)
    simulated_actions = actions_used
    validations: list[AgentActionValidation] = []

    for action in decision.actions:
        grounded_specifications, trusted_reference_texts = (
            trusted_context_by_query.get(action.query_id, (frozenset(), ()))
        )
        errors = list(validate_action(
            action,
            query_ids=query_ids,
            available_specifications=available_specifications,
            grounded_specifications=grounded_specifications,
            trusted_reference_texts=trusted_reference_texts,
            ledger=ledger,
        ))
        if (
            action.tool != "submit_for_validation"
            and action.tool not in available_tool_names
        ):
            errors.append(f"tool is not available: {action.tool}")
        if action.action_id in seen_action_ids:
            errors.append("duplicate action_id in the same turn")
        seen_action_ids.add(action.action_id)
        if contains_submission and len(decision.actions) != 1:
            errors.append(
                "submit_for_validation must be the only action in a turn"
            )

        signature = action_signature(action)
        if signature in signatures:
            errors.append("duplicate prior tool call")
        if (
            action.tool != "submit_for_validation"
            and simulated_actions >= total_action_limit
        ):
            errors.append("total tool action budget exhausted")
        if finalization_only and action.tool != "submit_for_validation":
            errors.append(
                "finalization turn permits only submit_for_validation"
            )

        accepted = not errors
        validations.append(AgentActionValidation(
            action_id=action.action_id,
            accepted=accepted,
            errors=tuple(errors),
        ))
        if not accepted or action.tool == "submit_for_validation":
            continue
        signatures.add(signature)
        simulated_actions += 1
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
    actions_used: int = 0,
    total_action_limit: int = 9,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
    history: Sequence[Mapping[str, Any]] = (),
    prior_signatures: frozenset[tuple[Any, ...]] = frozenset(),
    finalization_only: bool = False,
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
        actions_used=actions_used,
        total_action_limit=total_action_limit,
        hits_per_search=hits_per_search,
        max_evidence_per_question=max_evidence_per_question,
        history=history,
        evidence_views=ledger.selection_payload(),
        finalization_only=finalization_only,
    )
    response = controller_agent.decide(
        input_payload=payload,
        output_type=AgentRetrievalDecision,
    )
    query_ids = frozenset(task_ids)
    available_specification_set = frozenset(available_specifications)
    trusted_context_by_query = {
        task.task_id: _trusted_action_context(
            task,
            shared_context=dict(shared_context or {}),
            ledger=ledger,
            available_specifications=available_specification_set,
        )
        for task in research_tasks
    }
    selection_validations = _validate_evidence_selections(
        response.output,
        query_ids=query_ids,
        ledger=ledger,
    )
    validations = _validate_decision(
        response.output,
        query_ids=query_ids,
        available_specifications=available_specification_set,
        available_tool_names=frozenset(
            capability.name for capability in available_tools
        ),
        ledger=ledger,
        trusted_context_by_query=trusted_context_by_query,
        prior_signatures=prior_signatures,
        actions_used=actions_used,
        total_action_limit=total_action_limit,
        finalization_only=finalization_only,
    )
    return ResearcherActionPlan(
        input_payload=payload,
        decision=response.output,
        receipt=response.receipt,
        selection_validations=selection_validations,
        validations=validations,
        accepted=(
            all(item.accepted for item in validations)
            and all(item.accepted for item in selection_validations)
        ),
    )


@dataclass(frozen=True)
class ResearcherPlanningService:
    """Generate an auditable first turn without building or invoking tools."""

    controller_agent: StructuredResearcherController
    corpus_scope: CorpusScope
    available_specifications: tuple[str, ...]
    available_tools: tuple[ToolCapability, ...]
    max_rounds: int = 4
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
            total_action_limit=self.max_total_actions,
            hits_per_search=self.hits_per_search,
            max_evidence_per_question=self.max_evidence_per_question,
        )


__all__ = [
    "ResearcherPlanningService",
    "propose_researcher_turn",
]
