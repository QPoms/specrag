"""Grounded answer synthesis constrained to validated evidence identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


SynthesisMode = Literal["complete", "partial"]
PublicationPath = Literal[
    "full_synthesis", "qualified_synthesis", "status_only"
]


class SynthesisContractError(ValueError):
    """The model returned prose that violates the grounded output contract."""


@dataclass(frozen=True)
class PublicationDecision:
    path: PublicationPath
    covered_task_ids: tuple[str, ...]
    unresolved_task_ids: tuple[str, ...]
    conflicting_task_ids: tuple[str, ...]

    @property
    def synthesis_mode(self) -> SynthesisMode | None:
        if self.path == "full_synthesis":
            return "complete"
        if self.path == "qualified_synthesis":
            return "partial"
        return None


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

    return publication_decision((), validated, assessments).path != "status_only"


def publication_decision(
    tasks: tuple[ResearchTask, ...],
    validated: tuple[ValidatedEvidence, ...],
    assessments: tuple[TaskSufficiencyAssessment, ...],
) -> PublicationDecision:
    """Derive the application-owned publication path from validated state."""

    assessment_by_task = {item.task_id: item for item in assessments}
    ordered_task_ids = (
        tuple(task.task_id for task in tasks)
        if tasks
        else tuple(item.task_id for item in assessments)
    )
    selected_ids = {
        evidence_id
        for assessment in assessments
        if assessment.status in {"sufficient", "partial"}
        for evidence_id in assessment.supporting_evidence_ids
    }
    covered_set = {
        item.task_id
        for item in validated
        if (
            item.evidence_id in selected_ids
            and item.source_valid
            and item.supports_target
            and item.validation_status in {"supported", "partial"}
        )
    }
    covered = tuple(
        task_id for task_id in ordered_task_ids if task_id in covered_set
    )
    unresolved = tuple(
        task_id
        for task_id in ordered_task_ids
        if assessment_by_task.get(task_id) is None
        or assessment_by_task[task_id].status != "sufficient"
    )
    conflicts = tuple(
        task_id
        for task_id in ordered_task_ids
        if assessment_by_task.get(task_id) is not None
        and assessment_by_task[task_id].status == "conflict"
    )
    full = bool(ordered_task_ids) and not unresolved and set(covered) == set(
        ordered_task_ids
    )
    if full:
        path: PublicationPath = "full_synthesis"
    elif covered:
        path = "qualified_synthesis"
    else:
        path = "status_only"
    return PublicationDecision(
        path=path,
        covered_task_ids=covered,
        unresolved_task_ids=unresolved,
        conflicting_task_ids=conflicts,
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
        if not expected_tasks:
            raise ValueError("cannot synthesize without research tasks")
        if set(assessment_by_task) != expected_tasks:
            raise ValueError("cannot synthesize without one assessment per task")
        decision = publication_decision(tasks, validated, assessments)
        if decision.synthesis_mode is None:
            raise ValueError("status-only research cannot invoke synthesis")
        if decision.synthesis_mode != mode:
            raise ValueError(
                f"publication decision requires {decision.synthesis_mode} synthesis"
            )

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
        covered_task_ids = decision.covered_task_ids
        required_evidence_tasks = (
            expected_tasks if mode == "complete" else set(covered_task_ids)
        )
        missing_evidence_tasks = required_evidence_tasks - evidence_tasks
        if missing_evidence_tasks:
            raise ValueError(
                "cannot synthesize tasks without supported evidence: "
                + ", ".join(sorted(missing_evidence_tasks))
            )
        unresolved_task_ids = decision.unresolved_task_ids
        conflicting_task_ids = decision.conflicting_task_ids
        covered_tasks = tuple(
            task for task in tasks if task.task_id in set(covered_task_ids)
        )
        proposal = self.generator.synthesize(
            question,
            covered_tasks,
            evidence,
            mode=mode,
            covered_task_ids=covered_task_ids,
            unresolved_task_ids=unresolved_task_ids,
            conflicting_task_ids=conflicting_task_ids,
        )
        unknown = set(proposal.evidence_ids) - eligible_ids
        if unknown:
            raise SynthesisContractError(
                "synthesis cites ineligible evidence: "
                + ", ".join(sorted(unknown))
            )
        inline_ids = tuple(dict.fromkeys(re.findall(
            r"\[([A-Za-z0-9_.:-]+)\]",
            proposal.answer,
        )))
        if inline_ids != proposal.evidence_ids:
            raise SynthesisContractError(
                "inline answer citations must match synthesis evidence IDs"
            )
        cited_tasks = {
            evidence_by_id[evidence_id].task_id
            for evidence_id in proposal.evidence_ids
        }
        missing_answer_tasks = required_evidence_tasks - cited_tasks
        if missing_answer_tasks:
            raise SynthesisContractError(
                "synthesis citations do not cover tasks: "
                + ", ".join(sorted(missing_answer_tasks))
            )
        answer = proposal.answer
        if decision.path == "qualified_synthesis":
            answer = _append_qualification(
                answer,
                tasks=tasks,
                assessments=assessments,
                answer_language=str(
                    getattr(self.generator, "answer_language", "English")
                ),
            )
        return proposal.model_copy(update={
            "answer": answer,
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
        answer = "\n".join(lines)
        if mode == "partial":
            answer = _append_qualification(
                answer,
                tasks=tasks,
                assessments=assessments,
                answer_language=str(
                    getattr(self.generator, "answer_language", "English")
                ),
            )
        return EvidenceSynthesis(
            answer=answer,
            evidence_ids=tuple(evidence_ids),
            mode=mode,
            covered_task_ids=covered_task_ids,
            unresolved_task_ids=unresolved_task_ids,
            conflicting_task_ids=conflicting_task_ids,
        )


def _append_qualification(
    answer: str,
    *,
    tasks: tuple[ResearchTask, ...],
    assessments: tuple[TaskSufficiencyAssessment, ...],
    answer_language: str,
) -> str:
    """Append mandatory non-normative disclosure outside model control."""

    assessment_by_task = {item.task_id: item for item in assessments}
    chinese = answer_language.casefold().startswith(("zh", "chinese", "中文"))
    status_labels = {
        "sufficient": "证据充分",
        "partial": "部分支持",
        "insufficient": "证据不足",
        "conflict": "冲突未解决",
    }
    unresolved_lines: list[str] = []
    conflict_lines: list[str] = []
    for task in tasks:
        assessment = assessment_by_task[task.task_id]
        if assessment.status == "sufficient":
            continue
        diagnostic = assessment.missing_relation or (
            "尚未建立完整的、由来源支持的结论。" if chinese else
            "No complete source-backed conclusion was established."
        )
        line = (
            f"- {task.task_id}（{status_labels[assessment.status]}）；"
            f"研究范围：{task.question}；证据缺口：{diagnostic}"
            if chinese else
            f"- {task.task_id} ({assessment.status}); research scope: "
            f"{task.question}; evidence-review gap: {diagnostic}"
        )
        if assessment.status == "conflict":
            conflict_lines.append(line)
        else:
            unresolved_lines.append(line)
    sections = [answer.rstrip()]
    if unresolved_lines:
        sections.append(
            ("研究限制（由后端强制附加，不属于规范事实）：\n" if chinese else
             "Research limitations (backend-enforced; not specification facts):\n")
            + "\n".join(unresolved_lines)
        )
    if conflict_lines:
        sections.append(
            ("尚未解决的证据冲突（由后端强制附加；这些任务不发布确定性结论）：\n"
             if chinese else
             "Unresolved evidence conflicts (backend-enforced; no definitive "
             "conclusion is published for these tasks):\n")
            + "\n".join(conflict_lines)
        )
    return "\n\n".join(sections)


__all__ = [
    "EvidenceSynthesisGenerator",
    "EvidenceSynthesisService",
    "PublicationDecision",
    "PublicationPath",
    "SynthesisContractError",
    "partial_synthesis_eligible",
    "publication_decision",
]
