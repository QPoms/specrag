"""Native Researcher turn construction and trusted pre-execution checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.models import (
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
    ResearchTask,
)
from specrag.retrieval.models import CorpusScope
from specrag.tools import parse_retrieval_tool_arguments

from .models import (
    AgentControlCall,
    AgentEvidenceSelectionValidation,
    AgentResearchReview,
    AgentRetrievalCall,
    AgentToolCallValidation,
    ResearcherActionPlan,
    researcher_control_definitions,
)
from .policy import call_signature, validate_call
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
        return tuple(
            text for item in value for text in _text_values(item)
        )
    return ()


def _trusted_call_context(
    task: ResearchTask,
    *,
    shared_context: Mapping[str, Any],
    ledger: EvidenceLedger,
    available_specifications: frozenset[str],
) -> frozenset[str]:
    unresolved_gaps = shared_context.get("unresolved_gaps", ())
    grounded_gap_anchors = tuple(
        anchor
        for gap in (
            unresolved_gaps
            if isinstance(unresolved_gaps, Sequence)
            else ()
        )
        if isinstance(gap, Mapping)
        for anchor in gap.get("trusted_anchors", ())
        if isinstance(anchor, str)
    )
    texts = [
        *_text_values(task.model_dump(mode="json")),
        *_text_values(shared_context.get("dependency_context", ())),
        *grounded_gap_anchors,
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
    return frozenset(explicit_specifications | observed_specifications)


def _controller_payload(
    research_tasks: tuple[ResearchTask, ...],
    *,
    corpus_scope: CorpusScope,
    available_specifications: tuple[str, ...],
    shared_context: Mapping[str, Any],
    round_number: int,
    max_rounds: int,
    calls_used: int,
    total_call_limit: int,
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
        "budget": {
            "rounds_remaining_including_this": max(
                max_rounds - round_number + 1,
                0,
            ),
            "tool_actions_remaining": total_call_limit - calls_used,
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


def _parse_native_calls(
    calls: Sequence[Any],
    *,
    finalization_only: bool,
) -> tuple[
    AgentControlCall | None,
    tuple[AgentRetrievalCall, ...],
    tuple[str, ...],
]:
    errors: list[str] = []
    call_ids = [call.call_id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        errors.append("native response contains duplicate call IDs")
    control_names = {"update_research_state", "submit_for_validation"}
    controls = [call for call in calls if call.name in control_names]
    retrieval_calls = tuple(
        AgentRetrievalCall(
            call_id=call.call_id,
            tool=call.name,
            arguments=call.arguments,
        )
        for call in calls
        if call.name not in control_names
    )
    control: AgentControlCall | None = None
    if len(controls) != 1:
        errors.append("native turn requires exactly one Researcher control call")
    else:
        raw_control = controls[0]
        try:
            review = AgentResearchReview.model_validate(raw_control.arguments)
        except ValidationError as exc:
            messages = "; ".join(
                error["msg"] for error in exc.errors(include_url=False)
            )
            errors.append("invalid Researcher control arguments: " + messages)
        else:
            control = AgentControlCall(
                call_id=raw_control.call_id,
                name=raw_control.name,
                review=review,
            )
    if control is not None:
        if control.name == "update_research_state" and not retrieval_calls:
            errors.append("state update requires at least one retrieval call")
        if control.name == "submit_for_validation" and retrieval_calls:
            errors.append("submission cannot include retrieval calls")
        if finalization_only and control.name != "submit_for_validation":
            errors.append("finalization permits only submit_for_validation")
    return control, retrieval_calls, tuple(errors)


def _validate_evidence_selections(
    review: AgentResearchReview | None,
    *,
    query_ids: frozenset[str],
    ledger: EvidenceLedger,
) -> tuple[AgentEvidenceSelectionValidation, ...]:
    if review is None:
        return ()
    required_query_ids = frozenset(ledger.query_ids_with_candidates())
    seen_query_ids: set[str] = set()
    validations: list[AgentEvidenceSelectionValidation] = []
    for selection in review.evidence_selections:
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
    for query_id in sorted(required_query_ids - seen_query_ids):
        validations.append(AgentEvidenceSelectionValidation(
            query_id=query_id,
            accepted=False,
            errors=("evidence selection is required when candidates are present",),
        ))
    return tuple(validations)


def _validate_calls(
    calls: tuple[AgentRetrievalCall, ...],
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    available_tool_names: frozenset[str],
    ledger: EvidenceLedger,
    trusted_context_by_query: Mapping[str, frozenset[str]],
    prior_signatures: frozenset[tuple[Any, ...]],
    calls_used: int,
    total_call_limit: int,
) -> tuple[AgentToolCallValidation, ...]:
    signatures = set(prior_signatures)
    simulated_calls = calls_used
    validations: list[AgentToolCallValidation] = []
    for call in calls:
        errors: list[str] = []
        arguments = None
        if call.tool not in available_tool_names:
            errors.append(f"tool is not available: {call.tool}")
        else:
            try:
                arguments = parse_retrieval_tool_arguments(
                    call.tool,
                    call.arguments,
                )
            except (KeyError, ValueError) as exc:
                errors.append(str(exc))
        if arguments is not None:
            errors.extend(validate_call(
                call,
                arguments,
                query_ids=query_ids,
                available_specifications=available_specifications,
                grounded_specifications=trusted_context_by_query.get(
                    arguments.query_id,
                    frozenset(),
                ),
                ledger=ledger,
            ))
            signature = call_signature(call, arguments)
            if signature in signatures:
                errors.append("duplicate prior tool call")
            if simulated_calls >= total_call_limit:
                errors.append("total tool action budget exhausted")
        accepted = not errors
        validations.append(AgentToolCallValidation(
            call_id=call.call_id,
            accepted=accepted,
            errors=tuple(errors),
        ))
        if accepted and arguments is not None:
            signatures.add(call_signature(call, arguments))
            simulated_calls += 1
    return tuple(validations)


def propose_researcher_turn(
    research_tasks: tuple[ResearchTask, ...],
    *,
    controller_agent: StructuredResearcherController,
    corpus_scope: CorpusScope,
    available_specifications: tuple[str, ...],
    available_tools: tuple[NativeFunctionDefinition, ...],
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
    prior_response_items: Sequence[Mapping[str, Any]] = (),
    prior_outputs: Sequence[NativeFunctionCallOutput] = (),
    finalization_only: bool = False,
) -> ResearcherActionPlan:
    if not research_tasks:
        raise ValueError("Researcher requires at least one research task")
    task_ids = tuple(task.task_id for task in research_tasks)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Researcher task IDs must be unique")
    payload = _controller_payload(
        research_tasks,
        corpus_scope=corpus_scope,
        available_specifications=available_specifications,
        shared_context=dict(shared_context or {}),
        round_number=round_number,
        max_rounds=max_rounds,
        calls_used=actions_used,
        total_call_limit=total_action_limit,
        hits_per_search=hits_per_search,
        max_evidence_per_question=max_evidence_per_question,
        history=history,
        evidence_views=ledger.selection_payload(),
        finalization_only=finalization_only,
    )
    control_definitions = researcher_control_definitions()
    exposed_tools = (
        (control_definitions[1],)
        if finalization_only
        else (*control_definitions, *available_tools)
    )
    response = controller_agent.decide(
        input_payload=payload,
        tools=exposed_tools,
        prompt_variant=(
            "evidence_gap"
            if dict(shared_context or {}).get("gap_retrieval_mode")
            == "evidence_gap"
            else "initial"
        ),
        prior_response_items=prior_response_items,
        prior_outputs=prior_outputs,
        max_tool_calls=(
            1
            if finalization_only
            else max(1, total_action_limit - actions_used + 1)
        ),
    )
    control, retrieval_calls, protocol_errors = _parse_native_calls(
        response.calls,
        finalization_only=finalization_only,
    )
    review = control.review if control is not None else None
    query_ids = frozenset(task_ids)
    unknown_gap_ids = sorted({
        gap.query_id
        for gap in (review.unresolved_gaps if review is not None else ())
        if gap.query_id not in query_ids
    })
    if unknown_gap_ids:
        protocol_errors = (*protocol_errors, (
            "unresolved gaps contain unknown query IDs: "
            + ", ".join(unknown_gap_ids)
        ))
    available_specification_set = frozenset(available_specifications)
    trusted_context_by_query = {
        task.task_id: _trusted_call_context(
            task,
            shared_context=dict(shared_context or {}),
            ledger=ledger,
            available_specifications=available_specification_set,
        )
        for task in research_tasks
    }
    selection_validations = _validate_evidence_selections(
        review,
        query_ids=query_ids,
        ledger=ledger,
    )
    validations = _validate_calls(
        retrieval_calls,
        query_ids=query_ids,
        available_specifications=available_specification_set,
        available_tool_names=frozenset(tool.name for tool in available_tools),
        ledger=ledger,
        trusted_context_by_query=trusted_context_by_query,
        prior_signatures=prior_signatures,
        calls_used=actions_used,
        total_call_limit=total_action_limit,
    )
    accepted = (
        not protocol_errors
        and all(item.accepted for item in validations)
        and all(item.accepted for item in selection_validations)
    )
    return ResearcherActionPlan(
        input_payload=payload,
        native_calls=response.calls,
        control_call=control,
        retrieval_calls=retrieval_calls,
        receipt=response.receipt,
        selection_validations=selection_validations,
        validations=validations,
        protocol_errors=protocol_errors,
        accepted=accepted,
        replay_items=response.replay_items,
    )


@dataclass(frozen=True)
class ResearcherPlanningService:
    """Generate an auditable first native turn without invoking retrieval."""

    controller_agent: StructuredResearcherController
    corpus_scope: CorpusScope
    available_specifications: tuple[str, ...]
    available_tools: tuple[NativeFunctionDefinition, ...]
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


__all__ = ["ResearcherPlanningService", "propose_researcher_turn"]
