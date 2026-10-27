"""Mechanical adapters between agent executions and graph runtime state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from specrag.agents.models import AgenticRetrievalExecution
from specrag.agents.researcher import RegistryResearcherService
from specrag.evidence import (
    EvidenceSynthesisService,
    SourceValidationService,
    SynthesisContractError,
    publication_decision,
)
from specrag.models import (
    CandidateEvidence,
    ResearcherEvidenceSelectionRecord,
    TaskEvidenceSet,
    ToolCallRecord,
)

from .dependencies import build_dependency_context
from .gap_context import build_corrective_research_input
from .state import ResearchState


@dataclass(frozen=True)
class SingleResearcherHandler:
    """Select one dependency-ready task and run the bounded worker service."""

    service: RegistryResearcherService

    def __call__(self, state: ResearchState) -> dict[str, object]:
        completed = {
            task.task_id
            for task in state["research_tasks"]
            if task.status == "completed"
        }
        ready = tuple(
            task
            for task in state["research_tasks"]
            if task.status in {"pending", "running"}
            and set(task.dependencies).issubset(completed)
        )
        if not ready:
            raise ValueError("no dependency-ready research task is available")
        budget = state["budget"]

        def has_judge_gap(task_id: str) -> bool:
            return any(
                gap.task_id == task_id and not gap.resolved
                for gap in state["gaps"]
            )

        selected = next(
            (
                task
                for task in ready
                if budget.remaining_tool_calls_for(
                    task.task_id,
                    corrective=has_judge_gap(task.task_id),
                ) > 0
            ),
            None,
        )
        if selected is None:
            raise ValueError("Researcher task tool-call budgets are exhausted")
        running = selected.model_copy(update={"status": "running"})
        tasks = tuple(
            running if task.task_id == selected.task_id else task
            for task in state["research_tasks"]
        )
        corrective = has_judge_gap(selected.task_id)
        remaining_calls = budget.remaining_tool_calls_for(
            selected.task_id,
            corrective=corrective,
        )

        shared_context: dict[str, object] = {
            "research_id": state["research_id"],
            "iteration": state["iteration"],
            "gap_retrieval_mode": "initial",
            "unresolved_gaps": (),
            "dependency_context": build_dependency_context(
                state,
                selected,
            ),
        }
        corrective_options: dict[str, object] = {}
        if corrective:
            corrective_input = build_corrective_research_input(
                state,
                selected,
            )
            shared_context.update(corrective_input.shared_context)
            corrective_options = {
                "initial_evidence": corrective_input.initial_evidence,
                "initial_active_evidence_ids": (
                    corrective_input.initial_active_evidence_ids
                ),
                "initial_evidence_requirements": (
                    corrective_input.initial_evidence_requirements
                ),
            }
        execution, query_to_task = self.service.run(
            (running,),
            shared_context=shared_context,
            max_tool_calls=remaining_calls,
            **corrective_options,
        )
        adapted_state = cast(ResearchState, {**state, "research_tasks": tasks})
        update = researcher_execution_update(
            adapted_state,
            execution,
            query_to_task=query_to_task,
            task_id=selected.task_id,
            corrective=corrective,
        )
        update["research_tasks"] = tasks
        return update


@dataclass(frozen=True)
class SourceValidationHandler:
    """Expose deterministic source validation as a graph node handler."""

    service: SourceValidationService

    def __call__(self, state: ResearchState) -> dict[str, object]:
        batch = self.service.validate(
            state["research_tasks"],
            state["candidate_evidence"],
        )
        return {
            "source_validations": batch.source_validations,
        }


@dataclass(frozen=True)
class EvidenceSynthesisHandler:
    """Produce a complete or explicitly qualified evidence-backed answer."""

    service: EvidenceSynthesisService

    def __call__(self, state: ResearchState) -> dict[str, object]:
        decision = publication_decision(
            state["research_tasks"],
            state["validated_evidence"],
            state["sufficiency_assessments"],
        )
        mode = decision.synthesis_mode
        if mode is None:
            raise ValueError("status-only research cannot invoke synthesis")
        try:
            synthesis = self.service.synthesize(
                state["user_question"],
                state["research_tasks"],
                state["candidate_evidence"],
                state["validated_evidence"],
                state["sufficiency_assessments"],
                mode=mode,
            )
        except SynthesisContractError:
            synthesis = self.service.evidence_digest(
                state["research_tasks"],
                state["candidate_evidence"],
                state["validated_evidence"],
                state["sufficiency_assessments"],
                mode=mode,
            )
            errors = (
                *state["errors"],
                "model synthesis violated the grounded output contract; "
                "deterministic evidence digest published",
            )
        except Exception as exc:
            if getattr(exc, "failure_kind", None) is None:
                raise
            synthesis = self.service.evidence_digest(
                state["research_tasks"],
                state["candidate_evidence"],
                state["validated_evidence"],
                state["sufficiency_assessments"],
                mode=mode,
            )
            errors = (
                *state["errors"],
                "model synthesis unavailable; deterministic evidence digest published",
            )
        else:
            errors = state["errors"]
        return {
            "synthesis": synthesis,
            "final_answer": synthesis.answer,
            "publication_path": decision.path,
            "stop_reason": state["stop_reason"] or "completed",
            "errors": errors,
        }


def _candidate_id(task_id: str, chunk_id: str) -> str:
    return f"E:{task_id}:{chunk_id}"


def researcher_execution_update(
    state: ResearchState,
    execution: AgenticRetrievalExecution,
    *,
    query_to_task: Mapping[str, str],
    task_id: str | None = None,
    execution_id: str | None = None,
    corrective: bool = False,
) -> dict[str, object]:
    """Translate an audited Researcher execution into a state update.

    This function copies provenance and accounting only. It does not validate
    evidence or decide sufficiency; those responsibilities remain in their
    dedicated graph handlers.
    """

    known_tasks = {task.task_id for task in state["research_tasks"]}
    unknown_targets = sorted(set(query_to_task.values()) - known_tasks)
    if unknown_targets:
        raise ValueError(
            "query mapping references unknown research tasks: "
            + ", ".join(unknown_targets)
        )

    candidates = list(state["candidate_evidence"])
    known_sources = {
        (candidate.task_id, candidate.chunk_id): candidate.evidence_id
        for candidate in candidates
    }
    evidence_by_id = {
        candidate.evidence_id: candidate for candidate in candidates
    }
    evidence_ids: dict[tuple[str, str], str] = {}

    def convert_hits(retrieval_execution, *, promote: bool) -> None:
        for task_result in retrieval_execution.tasks:
            try:
                mapped_task_id = query_to_task[task_result.query_id]
            except KeyError as exc:
                raise ValueError(
                    "missing research-task mapping for "
                    f"{task_result.query_id}"
                ) from exc
            for hit in task_result.hits:
                source_identity = (mapped_task_id, hit.chunk_id)
                candidate_id = known_sources.get(
                    source_identity,
                    _candidate_id(mapped_task_id, hit.chunk_id),
                )
                evidence_ids[(task_result.query_id, hit.hit_id)] = candidate_id
                candidate = evidence_by_id.get(candidate_id)
                if candidate is None:
                    if hit.release is None:
                        raise ValueError(
                            f"retrieval hit has no Release: {hit.hit_id}"
                        )
                    candidate = CandidateEvidence(
                        evidence_id=candidate_id,
                        task_id=mapped_task_id,
                        chunk_id=hit.chunk_id,
                        specification=hit.specification,
                        release=hit.release,
                        version=hit.version,
                        source_path=hit.source_path,
                        document_sha256=hit.document_sha256,
                        heading_path=hit.heading_path,
                        line_start=hit.line_start,
                        line_end=hit.line_end,
                        excerpt=hit.excerpt,
                    )
                    evidence_by_id[candidate_id] = candidate
                if promote and source_identity not in known_sources:
                    candidates.append(candidate)
                    known_sources[source_identity] = candidate_id

    convert_hits(
        execution.candidate_evidence_execution
        or execution.evidence_execution,
        promote=True,
    )
    convert_hits(execution.evidence_execution, promote=True)

    calls = list(state["tool_calls"])
    iteration = state["iteration"] + 1
    for turn in execution.turns:
        for outcome in turn.outcomes:
            arguments = outcome.call.arguments
            query_id = str(arguments.get("query_id", ""))
            outcome_task_id = query_to_task.get(
                query_id,
                "unassigned",
            )
            calls.append(ToolCallRecord(
                call_id=outcome.call.call_id,
                task_id=outcome_task_id,
                tool_name=outcome.call.tool,
                query_id=query_id,
                lexical_query=str(arguments.get("lexical_query", "")),
                semantic_query=str(arguments.get("semantic_query", "")),
                hypothetical_document=str(
                    arguments.get("hypothetical_document", "")
                ),
                clause=str(arguments.get("clause", "")),
                target_specifications=tuple(
                    arguments.get("target_specifications", ())
                ),
                seed_evidence_ids=tuple(
                    arguments.get("seed_evidence_ids", ())
                ),
                status=outcome.status,
                evidence_ids=tuple(
                    evidence_ids.get(
                        (query_id, evidence_id),
                        evidence_id,
                    )
                    for evidence_id in outcome.evidence_ids
                ),
                new_evidence_ids=tuple(
                    evidence_ids.get(
                        (query_id, evidence_id),
                        evidence_id,
                    )
                    for evidence_id in outcome.new_evidence_ids
                ),
                observed_evidence=tuple(
                    evidence_by_id[mapped_id]
                    for evidence_id in outcome.evidence_ids
                    if (
                        mapped_id := evidence_ids.get(
                            (query_id, evidence_id),
                            evidence_id,
                        )
                    ) in evidence_by_id
                ),
                error="; ".join(outcome.errors) or None,
            ))

    selections = list(state["researcher_evidence_selections"])
    for turn in execution.turns:
        if turn.control_call is None:
            continue
        for selection in turn.control_call.review.evidence_selections:
            mapped_task_id = query_to_task.get(selection.query_id)
            if mapped_task_id is None:
                raise ValueError(
                    f"missing research-task mapping for {selection.query_id}"
                )
            selections.append(ResearcherEvidenceSelectionRecord(
                selection_id=(
                    f"I{iteration}:R{turn.round_number}:"
                    f"{selection.query_id}"
                ),
                task_id=mapped_task_id,
                round_number=turn.round_number,
                phase=turn.phase,
                query_id=selection.query_id,
                retained_evidence_ids=tuple(
                    evidence_ids.get(
                        (selection.query_id, evidence_id),
                        evidence_id,
                    )
                    for evidence_id in selection.retained_evidence_ids
                ),
                assessment=turn.control_call.review.assessment,
            ))

    budget = state["budget"]
    actions_used = (
        execution.search_actions_used + execution.expansion_actions_used
    )
    charged_task_id = task_id
    if charged_task_id is None:
        mapped_task_ids = set(query_to_task.values())
        if len(mapped_task_ids) != 1:
            raise ValueError(
                "Researcher execution accounting requires one task ID"
            )
        charged_task_id = mapped_task_ids.pop()
    budget = budget.record_tool_calls(
        charged_task_id,
        actions_used,
        corrective=corrective,
    )

    active_evidence_ids = tuple(dict.fromkeys(
        evidence_ids[(task_result.query_id, hit.hit_id)]
        for task_result in execution.evidence_execution.tasks
        if query_to_task[task_result.query_id] == charged_task_id
        for hit in task_result.hits
    ))
    evidence_sets = {
        item.task_id: item for item in state.get("task_evidence_sets", ())
    }
    evidence_sets[charged_task_id] = TaskEvidenceSet(
        task_id=charged_task_id,
        execution_id=execution_id,
        active_evidence_ids=active_evidence_ids,
    )
    requirements_by_task = {
        known_task_id: tuple(
            item
            for item in state.get("evidence_requirements", ())
            if item.task_id == known_task_id
        )
        for known_task_id in known_tasks
    }
    mapped_requirements = []
    for requirement in execution.evidence_requirements:
        mapped_task_id = query_to_task.get(requirement.task_id)
        if mapped_task_id is None:
            raise ValueError(
                "missing research-task mapping for requirement task "
                + requirement.task_id
            )
        mapped_requirements.append(requirement.model_copy(update={
            "task_id": mapped_task_id,
            "evidence_ids": tuple(
                evidence_ids.get(
                    (requirement.task_id, evidence_id), evidence_id
                )
                for evidence_id in requirement.evidence_ids
            ),
        }))
    requirements_by_task[charged_task_id] = tuple(mapped_requirements)

    receipts = tuple((*state["model_receipts"], *(
        turn.receipt.model_dump(mode="json") for turn in execution.turns
    )))
    update = {
        "candidate_evidence": tuple(candidates),
        "evidence_requirements": tuple(
            requirement
            for known_task_id in (
                task.task_id for task in state["research_tasks"]
            )
            for requirement in requirements_by_task[known_task_id]
        ),
        "tool_calls": tuple(calls),
        "researcher_evidence_selections": tuple(selections),
        "task_evidence_sets": tuple(
            evidence_sets[known_task_id]
            for known_task_id in (
                task.task_id for task in state["research_tasks"]
            )
            if known_task_id in evidence_sets
        ),
        "budget": budget,
        "last_researcher_stop_reason": execution.stop_reason,
        "errors": tuple((*state["errors"], *execution.errors)),
        "model_receipts": receipts,
    }
    if execution.stop_reason in {"cancelled", "elapsed_time_budget"}:
        update["stop_reason"] = execution.stop_reason
    return update


__all__ = [
    "SourceValidationHandler",
    "EvidenceSynthesisHandler",
    "SingleResearcherHandler",
    "researcher_execution_update",
]
