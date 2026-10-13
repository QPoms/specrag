"""Shared framework-independent domain and transport models."""

from .knowledge import (
    KnowledgeProposal,
    KnowledgeSource,
    MemorySearchResult,
    ValidatedKnowledge,
)

from .research import (
    CandidateEvidence,
    EvidenceBackedClaim,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearcherStopReason,
    ResearcherEvidenceSelectionRecord,
    ResearchTask,
    ResearchWorkerResult,
    SourceValidationResult,
    StopReason,
    TaskSufficiencyAssessment,
    TaskStatus,
    ToolCallRecord,
    ValidatedEvidence,
)
from .tool_calling import (
    NativeFunctionCall,
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
)

__all__ = [
    "CandidateEvidence",
    "EvidenceBackedClaim",
    "EvidenceSynthesis",
    "ResearchBudget",
    "ResearchGap",
    "ResearchPlan",
    "ResearchRouteDecision",
    "ResearcherStopReason",
    "ResearcherEvidenceSelectionRecord",
    "ResearchTask",
    "ResearchWorkerResult",
    "SourceValidationResult",
    "StopReason",
    "TaskSufficiencyAssessment",
    "TaskStatus",
    "ToolCallRecord",
    "ValidatedEvidence",
    "KnowledgeProposal",
    "KnowledgeSource",
    "MemorySearchResult",
    "ValidatedKnowledge",
    "NativeFunctionCall",
    "NativeFunctionCallOutput",
    "NativeFunctionDefinition",
]
