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
    ResearchTask,
    ResearchWorkerResult,
    SourceValidationResult,
    StopReason,
    TaskSufficiencyAssessment,
    TaskStatus,
    ToolCallRecord,
    ValidatedEvidence,
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
]
