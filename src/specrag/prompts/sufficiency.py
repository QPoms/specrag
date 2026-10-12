"""Task-level evidence selection and sufficiency prompt contract."""

SUFFICIENCY_PROMPT_VERSION = "evidence-sufficiency/v1.1"

SUFFICIENCY_INSTRUCTIONS = (
    "The supplied passages have already passed deterministic source validation. "
    "Select only evidence IDs that directly support the ResearchTask objective, "
    "then decide whether those selected passages cover the complete objective. "
    "Use conflict only for a direct contradiction between supplied passages. "
    "If the task is not sufficient, state one concrete missing_relation that "
    "can guide the next retrieval. supporting_evidence_ids is the authoritative "
    "set promoted to downstream dependency context and synthesis. Preserve "
    "task_id."
)

__all__ = ["SUFFICIENCY_INSTRUCTIONS", "SUFFICIENCY_PROMPT_VERSION"]
