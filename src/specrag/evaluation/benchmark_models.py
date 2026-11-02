"""Strict contracts for the final-answer end-to-end Benchmark."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Difficulty = Literal["simple", "complex"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreBand(_StrictModel):
    ratio: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1)


class RubricItem(_StrictModel):
    rubric_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    criterion: str = Field(min_length=1)
    weight: int = Field(gt=0, le=100)
    full: ScoreBand
    partial: ScoreBand
    none: ScoreBand

    @model_validator(mode="after")
    def _fixed_bands(self) -> "RubricItem":
        ratios = (self.full.ratio, self.partial.ratio, self.none.ratio)
        if ratios != (1.0, 0.5, 0.0):
            raise ValueError("Rubric bands must use full=1, partial=0.5, none=0")
        if self.weight % 5:
            raise ValueError("Rubric weights must be multiples of five")
        return self


class BenchmarkRubric(_StrictModel):
    max_score: Literal[100] = 100
    items: tuple[RubricItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_items(self) -> "BenchmarkRubric":
        identifiers = [item.rubric_id for item in self.items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Rubric item IDs must be unique")
        expected = [f"R{index}" for index in range(1, len(self.items) + 1)]
        if identifiers != expected:
            raise ValueError("Rubric item IDs must be contiguous R1..Rn")
        if sum(item.weight for item in self.items) != self.max_score:
            raise ValueError("Rubric weights must sum to 100")
        return self


class GoldEvidence(_StrictModel):
    evidence_id: str = Field(pattern=r"^GE[1-9][0-9]*$")
    specification: str = Field(min_length=1)
    release: Literal[17] = 17
    version: str = Field(pattern=r"^17\.[0-9]+\.[0-9]+$")
    clause: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_range(self) -> "GoldEvidence":
        if self.line_end < self.line_start:
            raise ValueError("Gold Evidence line range is reversed")
        return self


class CorpusDocument(_StrictModel):
    specification: str = Field(min_length=1)
    version: str = Field(pattern=r"^17\.[0-9]+\.[0-9]+$")
    release: Literal[17] = 17
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class BenchmarkEnvironment(_StrictModel):
    environment_id: str = Field(min_length=1)
    release: Literal[17] = 17
    version_policy: Literal["exact"] = "exact"
    index_fingerprint: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    documents: tuple[CorpusDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_documents(self) -> "BenchmarkEnvironment":
        specifications = [item.specification for item in self.documents]
        paths = [item.source_path for item in self.documents]
        if len(specifications) != len(set(specifications)):
            raise ValueError("Environment must contain one version per specification")
        if len(paths) != len(set(paths)):
            raise ValueError("Environment document paths must be unique")
        return self


class SourceMetadata(_StrictModel):
    dataset: str = Field(min_length=1)
    record_kind: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    license: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    transformation: str = Field(min_length=1)


class BenchmarkReview(_StrictModel):
    status: Literal["self_reviewed"] = "self_reviewed"
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime

    @model_validator(mode="after")
    def _timezone_aware(self) -> "BenchmarkReview":
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("Benchmark review time must include a timezone")
        return self


class TaskMetadata(_StrictModel):
    task_id: str = Field(pattern=r"^r17-qa-[0-9]{3}$")
    difficulty: Difficulty
    source: SourceMetadata
    review: BenchmarkReview


class BenchmarkWorkflowInput(_StrictModel):
    """The only Benchmark data permitted to cross into the Workflow."""

    task_id: str
    question: str
    environment: BenchmarkEnvironment


class BenchmarkCaseEnvironment(_StrictModel):
    """Question-local view of the frozen documents relevant to one case."""

    environment_id: str = Field(min_length=1)
    release: Literal[17] = 17
    relevant_documents: tuple[CorpusDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_relevant_documents(self) -> "BenchmarkCaseEnvironment":
        paths = [item.source_path for item in self.relevant_documents]
        if len(paths) != len(set(paths)):
            raise ValueError("Case Environment documents must be unique")
        return self


class BenchmarkCase(_StrictModel):
    """One composed case with all six agreed Benchmark sections."""

    schema_version: Literal["specrag.e2e-benchmark-case/v1"] = (
        "specrag.e2e-benchmark-case/v1"
    )
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    rubric: BenchmarkRubric
    gold_evidence: tuple[GoldEvidence, ...] = Field(min_length=1)
    environment: BenchmarkEnvironment
    task_metadata: TaskMetadata

    def workflow_input(self) -> BenchmarkWorkflowInput:
        return BenchmarkWorkflowInput(
            task_id=self.task_metadata.task_id,
            question=self.question,
            environment=self.environment,
        )


class RubricSourceItem(_StrictModel):
    answer_unit_id: str = Field(pattern=r"^AU[1-9][0-9]*$")
    weight: int = Field(gt=0, le=100)
    partial_credit_guidance: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _weight_step(self) -> "RubricSourceItem":
        if self.weight % 5:
            raise ValueError("Rubric weights must be multiples of five")
        return self


class RubricAnnotation(_StrictModel):
    schema_version: Literal["specrag.e2e-rubric/v1"] = "specrag.e2e-rubric/v1"
    question_id: str = Field(pattern=r"^r17-qa-[0-9]{3}$")
    curation_status: Literal["self_reviewed"] = "self_reviewed"
    curated_by: str = Field(min_length=1)
    curated_at: datetime
    items: tuple[RubricSourceItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_annotation(self) -> "RubricAnnotation":
        if self.curated_at.utcoffset() is None:
            raise ValueError("Rubric curation time must include a timezone")
        identifiers = [item.answer_unit_id for item in self.items]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Rubric Answer Unit IDs must be unique")
        if sum(item.weight for item in self.items) != 100:
            raise ValueError("Rubric source weights must sum to 100")
        return self


class FrozenSource(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkManifest(_StrictModel):
    schema_version: Literal["specrag.e2e-benchmark-manifest/v1"] = (
        "specrag.e2e-benchmark-manifest/v1"
    )
    benchmark_id: Literal["specrag-release17-final-answer"]
    benchmark_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    case_count: Literal[50]
    trial_count: Literal[1]
    scoring_scope: Literal["final_answer_only"]
    difficulty_counts: dict[Difficulty, int]
    sources: tuple[FrozenSource, ...] = Field(min_length=3, max_length=3)
    environment: BenchmarkEnvironment

    @model_validator(mode="after")
    def _fixed_counts(self) -> "BenchmarkManifest":
        if self.difficulty_counts != {"simple": 34, "complex": 16}:
            raise ValueError("Benchmark must freeze 34 simple and 16 complex cases")
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("Benchmark source paths must be unique")
        return self


class EndToEndBenchmark(_StrictModel):
    manifest: BenchmarkManifest
    cases: tuple[BenchmarkCase, ...] = Field(min_length=50, max_length=50)

    @model_validator(mode="after")
    def _valid_cases(self) -> "EndToEndBenchmark":
        identifiers = [item.task_metadata.task_id for item in self.cases]
        expected = [f"r17-qa-{index:03d}" for index in range(1, 51)]
        if identifiers != expected:
            raise ValueError("Benchmark cases must be ordered r17-qa-001..050")
        if len(self.cases) != self.manifest.case_count:
            raise ValueError("Benchmark case count disagrees with manifest")
        return self


class BenchmarkReviewCase(_StrictModel):
    """Self-contained case view with a reference to the top-level Environment."""

    schema_version: Literal["specrag.e2e-benchmark-review-case/v1"] = (
        "specrag.e2e-benchmark-review-case/v1"
    )
    task_metadata: TaskMetadata
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    rubric: BenchmarkRubric
    gold_evidence: tuple[GoldEvidence, ...] = Field(min_length=1)
    environment: BenchmarkCaseEnvironment


class BenchmarkReviewFile(_StrictModel):
    """Materialized single-file view intended for human review."""

    schema_version: Literal["specrag.e2e-benchmark-review/v1"] = (
        "specrag.e2e-benchmark-review/v1"
    )
    benchmark_id: Literal["specrag-release17-final-answer"]
    benchmark_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    case_count: Literal[50]
    trial_count: Literal[1]
    scoring_scope: Literal["final_answer_only"]
    difficulty_counts: dict[Difficulty, int]
    cases: tuple[BenchmarkReviewCase, ...] = Field(min_length=50, max_length=50)
    frozen_execution_environment: BenchmarkEnvironment

    @model_validator(mode="after")
    def _coherent_review_view(self) -> "BenchmarkReviewFile":
        identifiers = [item.task_metadata.task_id for item in self.cases]
        expected = [f"r17-qa-{index:03d}" for index in range(1, 51)]
        if identifiers != expected:
            raise ValueError("Review cases must be ordered r17-qa-001..050")
        if any(
            item.environment.environment_id
            != self.frozen_execution_environment.environment_id
            for item in self.cases
        ):
            raise ValueError("Review case references an unknown Environment")
        frozen_paths = {
            item.source_path for item in self.frozen_execution_environment.documents
        }
        if any(
            document.source_path not in frozen_paths
            for item in self.cases
            for document in item.environment.relevant_documents
        ):
            raise ValueError("Review case contains a document outside the Environment")
        counts = {
            "simple": sum(item.task_metadata.difficulty == "simple" for item in self.cases),
            "complex": sum(item.task_metadata.difficulty == "complex" for item in self.cases),
        }
        if counts != self.difficulty_counts:
            raise ValueError("Review difficulty counts disagree with cases")
        return self


def model_json(value: BaseModel) -> dict[str, Any]:
    """Return a JSON-safe strict-model representation for artifact callers."""

    return value.model_dump(mode="json")


__all__ = [
    "BenchmarkCase",
    "BenchmarkCaseEnvironment",
    "BenchmarkEnvironment",
    "BenchmarkManifest",
    "BenchmarkReviewCase",
    "BenchmarkReviewFile",
    "BenchmarkRubric",
    "BenchmarkWorkflowInput",
    "EndToEndBenchmark",
    "GoldEvidence",
    "RubricAnnotation",
    "RubricItem",
    "ScoreBand",
    "TaskMetadata",
    "model_json",
]
