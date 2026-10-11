"""Framework-independent evidence validation and sufficiency services."""

from .conflicts import EvidenceConflictAssessor, EvidenceConflictService
from .validation import (
    EvidenceSupportAssessor,
    EvidenceValidationBatch,
    EvidenceValidationService,
    SourceEvidenceValidator,
    deduplicate_candidates,
)
from .sufficiency import (
    EvidenceSufficiencyBatch,
    EvidenceSufficiencyService,
    TaskSufficiencyAssessor,
)
from .synthesis import EvidenceSynthesisGenerator, EvidenceSynthesisService

__all__ = [
    "EvidenceSupportAssessor",
    "EvidenceConflictAssessor",
    "EvidenceConflictService",
    "EvidenceSufficiencyBatch",
    "EvidenceSufficiencyService",
    "EvidenceSynthesisGenerator",
    "EvidenceSynthesisService",
    "EvidenceValidationBatch",
    "EvidenceValidationService",
    "SourceEvidenceValidator",
    "TaskSufficiencyAssessor",
    "deduplicate_candidates",
]
