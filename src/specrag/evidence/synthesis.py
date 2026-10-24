"""Grounded answer synthesis constrained to validated evidence identifiers."""

from __future__ import annotations

import re
from typing import Literal, Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


SynthesisMode = Literal["complete", "partial"]


class EvidenceSynthesisGenerator(Protocol):
    def synthesize(
        self,
        question: str,
        tasks: tuple[ResearchTask, ...],
        evidence: tuple[CandidateEvidence, ...],
        *,
        mode: SynthesisMode,
        covered_task_ids: tuple[str, ...],
        unresolved_task_ids: tuple[str, ...],
        conflicting_task_ids: tuple[str, ...],
    ) -> EvidenceSynthesis: ...


def partial_synthesis_eligible(
    validated: tuple[ValidatedEvidence, ...],
    assessments: tuple[TaskSufficiencyAssessment, ...],
) -> bool:
    """Return whether Judge-selected supported evidence can support an answer."""

    selected_ids = {
        evidence_id
        for assessment in assessments
        if assessment.status in {"sufficient", "partial"}
        for evidence_id in assessment.supporting_evidence_ids
    }
    return any(
        item.evidence_id in selected_ids
        and item.source_valid
        and item.supports_target
        and item.validation_status in {"supported", "partial"}
        for item in validated
    )


class EvidenceSynthesisService:
    def __init__(self, generator: EvidenceSynthesisGenerator) -> None:
        self.generator = generator

    def synthesize(
        self,
        question: str,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        validated: tuple[ValidatedEvidence, ...],
        assessments: tuple[TaskSufficiencyAssessment, ...],
        *,
        mode: SynthesisMode = "complete",
    ) -> EvidenceSynthesis:
        expected_tasks = {task.task_id for task in tasks}
        assessment_by_task = {item.task_id: item for item in assessments}
        sufficient_tasks = {
            item.task_id for item in assessments if item.status == "sufficient"
        }
        if not expected_tasks:
            raise ValueError("cannot synthesize without research tasks")
        if set(assessment_by_task) != expected_tasks:
            raise ValueError("cannot synthesize without one assessment per task")
        if mode == "complete" and sufficient_tasks != expected_tasks:
            raise ValueError("cannot synthesize before every task is sufficient")
        if mode == "partial" and sufficient_tasks == expected_tasks:
            raise ValueError("partial synthesis requires at least one unresolved task")

        selected_ids = {
            evidence_id
            for assessment in assessments
            if assessment.status in {"sufficient", "partial"}
            for evidence_id in assessment.supporting_evidence_ids
        }
        eligible_ids = {
            item.evidence_id
            for item in validated
            if (
                item.task_id in expected_tasks
                and item.evidence_id in selected_ids
                and item.source_valid
                and item.supports_target
                and item.validation_status in (
                    {"supported"} if mode == "complete" else {"supported", "partial"}
                )
            )
        }
        evidence = tuple(
            item for item in candidates if item.evidence_id in eligible_ids
        )
        if not evidence:
            raise ValueError("cannot synthesize without supported evidence")
        evidence_by_id = {item.evidence_id: item for item in evidence}
        evidence_tasks = {item.task_id for item in evidence}
        covered_task_ids = tuple(
            task.task_id for task in tasks if task.task_id in evidence_tasks
        )
        required_evidence_tasks = (
            expected_tasks if mode == "complete" else set(covered_task_ids)
        )
        missing_evidence_tasks = required_evidence_tasks - evidence_tasks
        if missing_evidence_tasks:
            raise ValueError(
                "cannot synthesize tasks without supported evidence: "
                + ", ".join(sorted(missing_evidence_tasks))
            )
        unresolved_task_ids = tuple(
            task.task_id
            for task in tasks
            if assessment_by_task[task.task_id].status != "sufficient"
        )
        conflicting_task_ids = tuple(
            task.task_id
            for task in tasks
            if assessment_by_task[task.task_id].status == "conflict"
        )
        proposal = self.generator.synthesize(
            question,
            tasks,
            evidence,
            mode=mode,
            covered_task_ids=covered_task_ids,
            unresolved_task_ids=unresolved_task_ids,
            conflicting_task_ids=conflicting_task_ids,
        )
        unknown = set(proposal.evidence_ids) - eligible_ids
        if unknown:
            raise ValueError(
                "synthesis cites ineligible evidence: "
                + ", ".join(sorted(unknown))
            )
        inline_ids = tuple(dict.fromkeys(re.findall(
            r"\[([A-Za-z0-9_.:-]+)\]",
            proposal.answer,
        )))
        if inline_ids != proposal.evidence_ids:
            raise ValueError(
                "inline answer citations must match synthesis evidence IDs"
            )
        cited_tasks = {
            evidence_by_id[evidence_id].task_id
            for evidence_id in proposal.evidence_ids
        }
        missing_answer_tasks = required_evidence_tasks - cited_tasks
        if missing_answer_tasks:
            raise ValueError(
                "synthesis citations do not cover tasks: "
                + ", ".join(sorted(missing_answer_tasks))
            )
        return proposal.model_copy(update={
            "mode": mode,
            "covered_task_ids": covered_task_ids,
            "unresolved_task_ids": unresolved_task_ids,
            "conflicting_task_ids": conflicting_task_ids,
        })

    def evidence_digest(
        self,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        validated: tuple[ValidatedEvidence, ...],
        assessments: tuple[TaskSufficiencyAssessment, ...],
        *,
        mode: SynthesisMode,
    ) -> EvidenceSynthesis:
        """Build a deterministic, claim-limited fallback from eligible evidence."""

        expected_tasks = {task.task_id for task in tasks}
        assessment_by_task = {item.task_id: item for item in assessments}
        if not expected_tasks or set(assessment_by_task) != expected_tasks:
            raise ValueError("evidence digest requires one assessment per task")
        selected_ids = {
            evidence_id
            for assessment in assessments
            if assessment.status in {"sufficient", "partial"}
            for evidence_id in assessment.supporting_evidence_ids
        }
        eligible_ids = {
            item.evidence_id
            for item in validated
            if item.task_id in expected_tasks
            and item.evidence_id in selected_ids
            and item.source_valid
            and item.supports_target
            and item.validation_status
            in ({"supported"} if mode == "complete" else {"supported", "partial"})
        }
        evidence = tuple(
            item for item in candidates if item.evidence_id in eligible_ids
        )
        if not evidence:
            raise ValueError("evidence digest requires supported evidence")
        evidence_tasks = {item.task_id for item in evidence}
        if mode == "complete" and evidence_tasks != expected_tasks:
            raise ValueError("complete evidence digest must cover every task")
        covered_task_ids = tuple(
            task.task_id for task in tasks if task.task_id in evidence_tasks
        )
        unresolved_task_ids = tuple(
            task.task_id
            for task in tasks
            if assessment_by_task[task.task_id].status != "sufficient"
        )
        conflicting_task_ids = tuple(
            task.task_id
            for task in tasks
            if assessment_by_task[task.task_id].status == "conflict"
        )
        lines = [
            "Validated evidence digest: the model synthesis step was unavailable; "
            "the excerpts below are preserved without extending them into new claims."
        ]
        evidence_ids: list[str] = []
        for task in tasks:
            task_evidence = tuple(
                item for item in evidence if item.task_id == task.task_id
            )
            if not task_evidence:
                continue
            lines.append(f"\n- {task.task_id}: {task.objective}")
            for item in task_evidence:
                location = " > ".join(item.heading_path) or "document body"
                excerpt = " ".join(item.excerpt.split())[:600]
                lines.append(
                    f"  - TS {item.specification} v{item.version}; {location}; "
                    f"lines {item.line_start}-{item.line_end}: {excerpt} "
                    f"[{item.evidence_id}]"
                )
                evidence_ids.append(item.evidence_id)
        if unresolved_task_ids:
            lines.append(
                "\nUnresolved task IDs: " + ", ".join(unresolved_task_ids)
            )
        return EvidenceSynthesis(
            answer="\n".join(lines),
            evidence_ids=tuple(evidence_ids),
            mode=mode,
            covered_task_ids=covered_task_ids,
            unresolved_task_ids=unresolved_task_ids,
            conflicting_task_ids=conflicting_task_ids,
        )


__all__ = [
    "EvidenceSynthesisGenerator",
    "EvidenceSynthesisService",
    "partial_synthesis_eligible",
]
