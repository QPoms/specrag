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
from .synthesis import (
    EvidenceSynthesisGenerator,
    EvidenceSynthesisService,
    partial_synthesis_eligible,
)
from .results import ResearchResultMaterializer

__all__ = [
    "EvidenceSufficiencyBatch",
    "EvidenceSufficiencyService",
    "EvidenceSynthesisGenerator",
    "EvidenceSynthesisService",
    "partial_synthesis_eligible",
    "ResearchResultMaterializer",
    "SourceEvidenceValidator",
    "SourceValidationBatch",
    "SourceValidationService",
    "TaskSufficiencyAssessor",
    "deduplicate_candidates",
]
