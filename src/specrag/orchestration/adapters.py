"""Mechanical adapters between agent executions and graph runtime state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from specrag.agents.models import AgenticRetrievalExecution
from specrag.agents.researcher import RegistryResearcherService
from specrag.evidence import (
    EvidenceConflictService,
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    EvidenceValidationService,
)
from specrag.memory import MemoryDistiller, ValidatedMemoryService
from specrag.models import CandidateEvidence, ToolCallRecord

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
        selected = ready[0]
        running = selected.model_copy(update={"status": "running"})
        tasks = tuple(
            running if task.task_id == selected.task_id else task
            for task in state["research_tasks"]
        )
        budget = state["budget"]
        remaining_calls = budget.max_tool_calls - budget.tool_calls_used
        if remaining_calls < 1:
            raise ValueError("Researcher tool-call budget is exhausted")

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
        )
        update["research_tasks"] = tasks
        return update


@dataclass(frozen=True)
class EvidenceValidationHandler:
    """Expose the framework-independent validator as a graph node handler."""

    service: EvidenceValidationService
    conflict_service: EvidenceConflictService | None = None

    def __call__(self, state: ResearchState) -> dict[str, object]:
        batch = self.service.validate(
            state["research_tasks"],
            state["candidate_evidence"],
        )
        conflicts = (
            self.conflict_service.detect(
                state["research_tasks"],
                state["candidate_evidence"],
                batch.validated_evidence,
            )
            if self.conflict_service is not None
            else ()
        )
        return {
            "source_validations": batch.source_validations,
            "validated_evidence": batch.validated_evidence,
            "evidence_conflicts": conflicts,
        }


@dataclass(frozen=True)
class EvidenceSufficiencyHandler:
    """Expose task-level sufficiency and Gap generation as a graph handler."""

    service: EvidenceSufficiencyService

    def __call__(self, state: ResearchState) -> dict[str, object]:
        batch = self.service.evaluate(
            state["research_tasks"],
            state["candidate_evidence"],
            state["validated_evidence"],
            state["evidence_conflicts"],
        )
        return {
            "research_tasks": batch.tasks,
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
    evidence_ids: dict[tuple[str, str], str] = {}
    for task_result in execution.evidence_execution.tasks:
        try:
            task_id = query_to_task[task_result.query_id]
        except KeyError as exc:
            raise ValueError(
                f"missing research-task mapping for {task_result.query_id}"
            ) from exc
        for hit in task_result.hits:
            source_identity = (task_id, hit.chunk_id)
            candidate_id = known_sources.get(
                source_identity,
                _candidate_id(task_id, hit.chunk_id),
            )
            evidence_ids[(task_result.query_id, hit.hit_id)] = candidate_id
            if source_identity in known_sources:
                continue
            if hit.release is None:
                raise ValueError(f"retrieval hit has no Release: {hit.hit_id}")
            candidates.append(CandidateEvidence(
                evidence_id=candidate_id,
                task_id=task_id,
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
            ))
            known_sources[source_identity] = candidate_id

    calls = list(state["tool_calls"])
    iteration = state["iteration"] + 1
    for turn in execution.turns:
        for outcome in turn.outcomes:
            if outcome.action.tool == "finish":
                continue
            task_id = query_to_task.get(
                outcome.action.query_id,
                "unassigned",
            )
            calls.append(ToolCallRecord(
                call_id=(
                    f"I{iteration}:R{turn.round_number}:"
                    f"{outcome.action.action_id}"
                ),
                task_id=task_id,
                tool_name=outcome.action.tool,
                status=outcome.status,
                evidence_ids=tuple(
                    evidence_ids.get(
                        (outcome.action.query_id, evidence_id),
                        evidence_id,
                    )
                    for evidence_id in outcome.evidence_ids
                ),
                error="; ".join(outcome.errors) or None,
            ))

    budget = state["budget"]
    used = (
        budget.tool_calls_used
        + execution.search_actions_used
        + execution.expansion_actions_used
    )
    if used > budget.max_tool_calls:
        raise ValueError(
            f"Researcher used {used} tool calls against budget "
            f"{budget.max_tool_calls}"
        )

    return {
        "candidate_evidence": tuple(candidates),
        "tool_calls": tuple(calls),
        "budget": budget.model_copy(update={"tool_calls_used": used}),
        "last_researcher_stop_reason": execution.stop_reason,
        "errors": tuple((*state["errors"], *execution.errors)),
    }


__all__ = [
    "EvidenceValidationHandler",
    "EvidenceSufficiencyHandler",
    "EvidenceSynthesisHandler",
    "MemoryWriteHandler",
    "SingleResearcherHandler",
    "researcher_execution_update",
]
