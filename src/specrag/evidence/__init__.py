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
    PublicationDecision,
    PublicationPath,
    SynthesisContractError,
    partial_synthesis_eligible,
    publication_decision,
)
from .results import ResearchResultMaterializer

__all__ = [
    "EvidenceSufficiencyBatch",
    "EvidenceSufficiencyService",
    "EvidenceSynthesisGenerator",
    "EvidenceSynthesisService",
    "PublicationDecision",
    "PublicationPath",
    "SynthesisContractError",
    "partial_synthesis_eligible",
    "publication_decision",
    "ResearchResultMaterializer",
    "SourceEvidenceValidator",
    "SourceValidationBatch",
    "SourceValidationService",
    "TaskSufficiencyAssessor",
    "deduplicate_candidates",
]
