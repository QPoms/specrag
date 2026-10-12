"""Framework-independent evidence validation and sufficiency services."""

from .validation import (
    SourceEvidenceValidator,
    SourceValidationBatch,
    SourceValidationService,
    deduplicate_candidates,
)
from .sufficiency import (
    EvidenceSufficiencyBatch,
    EvidenceSufficiencyService,
    TaskSufficiencyAssessor,
)
from .synthesis import EvidenceSynthesisGenerator, EvidenceSynthesisService

__all__ = [
    "EvidenceSufficiencyBatch",
    "EvidenceSufficiencyService",
    "EvidenceSynthesisGenerator",
    "EvidenceSynthesisService",
    "SourceEvidenceValidator",
    "SourceValidationBatch",
    "SourceValidationService",
    "TaskSufficiencyAssessor",
    "deduplicate_candidates",
]
