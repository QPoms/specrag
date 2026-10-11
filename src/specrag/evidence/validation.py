"""Source-faithful validation for candidate research evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceSupportAssessment,
    ResearchTask,
    SourceValidationResult,
    ValidatedEvidence,
)


class EvidenceSupportAssessor(Protocol):
    def assess(
        self,
        task: ResearchTask,
        evidence: CandidateEvidence,
    ) -> EvidenceSupportAssessment: ...


@dataclass(frozen=True)
class EvidenceValidationBatch:
    source_validations: tuple[SourceValidationResult, ...]
    validated_evidence: tuple[ValidatedEvidence, ...]


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def deduplicate_candidates(
    candidates: Iterable[CandidateEvidence],
) -> tuple[CandidateEvidence, ...]:
    """Remove repeated or overlapping copies of the same source passage."""

    selected: list[CandidateEvidence] = []
    seen_chunks: set[tuple[str, str]] = set()
    for candidate in candidates:
        chunk_identity = (candidate.task_id, candidate.chunk_id)
        if chunk_identity in seen_chunks:
            continue
        candidate_text = _normalized_text(candidate.excerpt)
        duplicate = False
        for existing in selected:
            same_document = (
                existing.task_id == candidate.task_id
                and existing.source_path == candidate.source_path
                and existing.document_sha256 == candidate.document_sha256
                and existing.version == candidate.version
            )
            overlaps = not (
                candidate.line_end < existing.line_start
                or existing.line_end < candidate.line_start
            )
            existing_text = _normalized_text(existing.excerpt)
            text_contains = (
                candidate_text in existing_text
                or existing_text in candidate_text
            )
            if same_document and overlaps and text_contains:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(candidate)
        seen_chunks.add(chunk_identity)
    return tuple(selected)


class SourceEvidenceValidator:
    """Verify path, document hash, locator, and exact excerpt against disk."""

    def __init__(
        self,
        repository_root: Path,
        *,
        allowed_roots: tuple[str, ...] = ("data/3gpp",),
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.allowed_roots = tuple(PurePosixPath(root) for root in allowed_roots)

    def validate(self, evidence: CandidateEvidence) -> SourceValidationResult:
        errors: list[str] = []
        relative = PurePosixPath(evidence.source_path.replace("\\", "/"))
        path_allowed = (
            not relative.is_absolute()
            and ".." not in relative.parts
            and any(relative.is_relative_to(root) for root in self.allowed_roots)
        )
        source = (self.repository_root / Path(*relative.parts)).resolve()
        try:
            source.relative_to(self.repository_root)
        except ValueError:
            path_allowed = False
        if not path_allowed:
            errors.append("source path is outside the allowed corpus roots")

        document_hash_valid = False
        locator_valid = False
        excerpt_valid = False
        if path_allowed and source.is_file():
            raw = source.read_bytes()
            document_hash_valid = (
                hashlib.sha256(raw).hexdigest().upper()
                == evidence.document_sha256
            )
            if not document_hash_valid:
                errors.append("document hash does not match the indexed source")
            lines = source.read_text(encoding="utf-8").splitlines()
            locator_valid = (
                1 <= evidence.line_start <= evidence.line_end <= len(lines)
            )
            if not locator_valid:
                errors.append("source line locator is outside the document")
            else:
                located = "\n".join(
                    lines[evidence.line_start - 1:evidence.line_end]
                )
                excerpt_valid = evidence.excerpt.strip() in located
                if not excerpt_valid:
                    errors.append("evidence excerpt is not present at its locator")
        elif path_allowed:
            errors.append("source document does not exist")

        return SourceValidationResult(
            evidence_id=evidence.evidence_id,
            valid=(
                path_allowed
                and document_hash_valid
                and locator_valid
                and excerpt_valid
            ),
            path_allowed=path_allowed,
            document_hash_valid=document_hash_valid,
            locator_valid=locator_valid,
            excerpt_valid=excerpt_valid,
            errors=tuple(errors),
        )


class EvidenceValidationService:
    def __init__(
        self,
        source_validator: SourceEvidenceValidator,
        support_assessor: EvidenceSupportAssessor,
    ) -> None:
        self.source_validator = source_validator
        self.support_assessor = support_assessor

    def validate(
        self,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
    ) -> EvidenceValidationBatch:
        task_by_id = {task.task_id: task for task in tasks}
        source_results: list[SourceValidationResult] = []
        validated: list[ValidatedEvidence] = []
        for candidate in deduplicate_candidates(candidates):
            try:
                task = task_by_id[candidate.task_id]
            except KeyError as exc:
                raise ValueError(
                    f"candidate references unknown task: {candidate.task_id}"
                ) from exc
            source_result = self.source_validator.validate(candidate)
            source_results.append(source_result)
            if not source_result.valid:
                validated.append(ValidatedEvidence(
                    evidence_id=candidate.evidence_id,
                    task_id=candidate.task_id,
                    source_valid=False,
                    supports_target=False,
                    validation_status="rejected",
                    rationale="; ".join(source_result.errors),
                ))
                continue
            assessment = self.support_assessor.assess(task, candidate)
            if assessment.evidence_id != candidate.evidence_id:
                raise ValueError(
                    "support assessment returned a mismatched evidence ID"
                )
            validated.append(ValidatedEvidence(
                evidence_id=candidate.evidence_id,
                task_id=candidate.task_id,
                source_valid=True,
                supports_target=assessment.status == "supported",
                validation_status=assessment.status,
                rationale=assessment.rationale,
            ))
        return EvidenceValidationBatch(
            source_validations=tuple(source_results),
            validated_evidence=tuple(validated),
        )


__all__ = [
    "EvidenceSupportAssessor",
    "EvidenceValidationBatch",
    "EvidenceValidationService",
    "SourceEvidenceValidator",
    "deduplicate_candidates",
]
