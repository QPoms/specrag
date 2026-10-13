"""Mechanical adapters between agent executions and graph runtime state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from specrag.agents.models import AgenticRetrievalExecution
from specrag.agents.researcher import RegistryResearcherService
from specrag.evidence import (
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    SourceValidationService,
)
from specrag.memory import MemoryDistiller, ValidatedMemoryService
from specrag.models import (
    CandidateEvidence,
    ResearcherEvidenceSelectionRecord,
    ToolCallRecord,
)

from .dependencies import build_dependency_context
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
        selected = next(
            (
                task
                for task in ready
                if budget.remaining_tool_calls_for(task.task_id) > 0
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
        remaining_calls = budget.remaining_tool_calls_for(selected.task_id)

        relevant_gaps = tuple(
            gap.model_dump(mode="json")
            for gap in state["gaps"]
            if gap.task_id == selected.task_id and not gap.resolved
        )
        execution, query_to_task = self.service.run(
            (running,),
            shared_context={
                "research_id": state["research_id"],
                "iteration": state["iteration"],
                "unresolved_gaps": relevant_gaps,
                "dependency_context": build_dependency_context(
                    state,
                    selected,
                ),
            },
            max_tool_calls=remaining_calls,
        )
        adapted_state = cast(ResearchState, {**state, "research_tasks": tasks})
        update = researcher_execution_update(
            adapted_state,
            execution,
            query_to_task=query_to_task,
            task_id=selected.task_id,
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
class EvidenceSufficiencyHandler:
    """Expose task-level sufficiency and Gap generation as a graph handler."""

    service: EvidenceSufficiencyService

    def __call__(self, state: ResearchState) -> dict[str, object]:
        batch = self.service.evaluate(
            state["research_tasks"],
            state["candidate_evidence"],
            state["source_validations"],
        )
        return {
            "research_tasks": batch.tasks,
            "validated_evidence": batch.validated_evidence,
            "sufficiency_assessments": batch.assessments,
            "gaps": batch.gaps,
            "evidence_sufficient": batch.evidence_sufficient,
        }


@dataclass(frozen=True)
class EvidenceSynthesisHandler:
    """Produce the final answer only from sufficient validated evidence."""

    service: EvidenceSynthesisService

    def __call__(self, state: ResearchState) -> dict[str, object]:
        if state["evidence_sufficient"] is not True:
            raise ValueError("cannot synthesize while evidence is insufficient")
        synthesis = self.service.synthesize(
            state["user_question"],
            state["research_tasks"],
            state["candidate_evidence"],
            state["validated_evidence"],
            state["sufficiency_assessments"],
        )
        return {
            "synthesis": synthesis,
            "final_answer": synthesis.answer,
            "stop_reason": "completed",
        }


@dataclass(frozen=True)
class MemoryWriteHandler:
    """Persist only knowledge distilled from a completed validated workflow."""

    service: ValidatedMemoryService
    distiller: MemoryDistiller

    def __call__(self, state: ResearchState) -> dict[str, object]:
        if state["synthesis"] is None:
            raise ValueError("memory write requires an evidence synthesis")
        records = self.service.distill_and_store(
            synthesis=state["synthesis"],
            tasks=state["research_tasks"],
            candidates=state["candidate_evidence"],
            validated=state["validated_evidence"],
            sufficiency=state["sufficiency_assessments"],
            distiller=self.distiller,
        )
        return {
            "memory_knowledge_ids": tuple(
                record.knowledge_id for record in records
            ),
        }


def _candidate_id(task_id: str, chunk_id: str) -> str:
    return f"E:{task_id}:{chunk_id}"


def researcher_execution_update(
    state: ResearchState,
    execution: AgenticRetrievalExecution,
    *,
    query_to_task: Mapping[str, str],
    task_id: str | None = None,
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
        promote=False,
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
    budget = budget.record_tool_calls(charged_task_id, actions_used)

    return {
        "candidate_evidence": tuple(candidates),
        "tool_calls": tuple(calls),
        "researcher_evidence_selections": tuple(selections),
        "budget": budget,
        "last_researcher_stop_reason": execution.stop_reason,
        "errors": tuple((*state["errors"], *execution.errors)),
    }


__all__ = [
    "SourceValidationHandler",
    "EvidenceSufficiencyHandler",
    "EvidenceSynthesisHandler",
    "MemoryWriteHandler",
    "SingleResearcherHandler",
    "researcher_execution_update",
]
