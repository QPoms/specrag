"""Composable evidence-grounded final synthesis prompt contract."""

from __future__ import annotations

from typing import Literal


SYNTHESIS_PROMPT_VERSION = "evidence-synthesis/v1.10"

_COMMON_GROUNDING = """\
You produce source-backed content for one 3GPP research answer. The backend has
already decided the publication path and filtered the input through source
validation and task-level sufficiency review.

The original question is the authoritative answer scope. Supplied task
questions are coverage checkpoints only. Use only supplied evidence. Do not
retrieve, re-evaluate sufficiency, cite rejected material, fill gaps from
memory, or introduce a normative conclusion that the passages do not establish.

Every factual statement, relation, comparison, or conclusion must cite one or
more supplied evidence IDs directly after the supported clause using the exact
format `[evidence_id]`. Preserve specification-specific actors, interfaces,
messages, states, conditions, exceptions, scope, and implication direction.
Do not combine analogous procedures unless the evidence directly establishes
that relationship. Omit true but unrequested detail.

Follow the supplied presentation language and detail level. Return `answer` as
one coherent answer and `evidence_ids` as the ordered, deduplicated IDs cited
inline. Return no claim list, status label, limitation section, or second answer
representation; the backend owns publication status and qualification text.
"""

_FULL_SYNTHESIS = """\
Publication path: full synthesis. Every supplied task is evidence-supported.
Answer every substantive part of the original question and cite evidence owned
by every supplied task. Do not invent an unresolved issue or qualification.
"""

_QUALIFIED_SYNTHESIS = """\
Publication path: qualified synthesis. Generate only the supported portion of
the answer from `covered_task_ids`. A partially supported task can appear in
both `covered_task_ids` and `unresolved_task_ids`: for such a task, answer only
the propositions directly established by its supplied evidence and leave its
unsupported relation or condition unresolved. Do not answer or speculate about
an unresolved task that is absent from `covered_task_ids`. Do not imply that
the complete original question has been resolved.
Do not turn missing evidence into a negative conclusion.
Cite evidence owned by every covered task.
The backend will append the mandatory unresolved-task disclosure; do not try
to paraphrase or replace it.
"""

_CONFLICT_RESTRICTION = """\
Some unresolved tasks are listed in `conflicting_task_ids`. Do not reconcile,
select between, or derive a conclusion from their conflicting evidence. The
backend will publish the conflict disclosure separately.
"""


def synthesis_prompt(
    mode: Literal["complete", "partial"],
    *,
    has_conflicts: bool,
) -> tuple[str, str]:
    """Return concrete version and instructions for one publication path."""

    if mode == "complete":
        if has_conflicts:
            raise ValueError("complete synthesis cannot contain conflicts")
        suffix = "full"
        blocks = (_COMMON_GROUNDING, _FULL_SYNTHESIS)
    else:
        suffix = "qualified-conflict" if has_conflicts else "qualified"
        blocks = (
            _COMMON_GROUNDING,
            _QUALIFIED_SYNTHESIS,
            *(() if not has_conflicts else (_CONFLICT_RESTRICTION,)),
        )
    return f"{SYNTHESIS_PROMPT_VERSION}-{suffix}", "\n".join(blocks)


# Compatibility export for callers that inspect the base contract. Runtime
# calls use `synthesis_prompt` and receive exactly one publication branch.
SYNTHESIS_INSTRUCTIONS = "\n".join((_COMMON_GROUNDING, _FULL_SYNTHESIS))


__all__ = [
    "SYNTHESIS_INSTRUCTIONS",
    "SYNTHESIS_PROMPT_VERSION",
    "synthesis_prompt",
]
