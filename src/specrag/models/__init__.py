"""Shared framework-independent domain and transport models."""

from .access import ActorContext, Membership, User, Workspace
from .runtime import ResearchRequest, ResearchRun, RunEvent, RunStatus, RunStopReason
from .specfacts import (
    SpecFact,
    SpecFactDraft,
    SpecFactProposal,
    SpecFactSearchResult,
    SpecFactSource,
)
from .user_memory import MemoryOrigin, MemoryStatus, UserMemoryItem, UserMemoryKind

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
    "ActorContext",
    "Membership",
    "ResearchRun",
    "ResearchRequest",
    "RunEvent",
    "RunStatus",
    "RunStopReason",
    "SpecFact",
    "SpecFactDraft",
    "SpecFactProposal",
    "SpecFactSearchResult",
    "SpecFactSource",
    "User",
    "UserMemoryItem",
    "UserMemoryKind",
    "MemoryOrigin",
    "MemoryStatus",
    "Workspace",
    "NativeFunctionCall",
    "NativeFunctionCallOutput",
    "NativeFunctionDefinition",
]
