"""Research Planner prompt contract."""

PLANNER_PROMPT_VERSION = "research-planner/v1.3"

PLANNER_INSTRUCTIONS = (
    "Create 2 to 4 ResearchTask items with distinct, non-overlapping research "
    "objectives that together cover what must be established to answer the "
    "complete 3GPP research question. Use IDs T1, T2, and so on; use status "
    "pending. "

    "Each task must represent a distinct research objective and state what needs "
    "to be established without asserting the answer or prescribing how evidence "
    "should be retrieved. Keep each task within the scope of the original question "
    "and use the minimum research scope sufficient to answer it. Include supporting "
    "details only when they are necessary to establish the requested conclusion, "
    "and keep such details within the relevant task rather than expanding them into "
    "broader research objectives. "

    "Tasks that can be researched independently should have no dependencies. "
    "Use a dependency only when completed and validated findings from an earlier "
    "task must be provided as research context to determine the subject, scope, "
    "condition, branch, meaning, or validation target of a later task. Protocol "
    "sequence, procedural order, shared terminology, related content, answer "
    "order, or later combination of task results do not by themselves create a "
    "dependency. "

    "When a dependency is required, list only the earlier task IDs whose validated "
    "findings are needed by the later task. Return tasks in dependency order and "
    "use an empty dependencies list for independently researchable tasks. "

    "For comparison, mapping, or relationship questions, create a separate research "
    "objective only when the relationship requires its own source-backed research "
    "or validation. Do not create a separate task merely to combine, restate, or "
    "summarize findings that will already be established by other tasks. "

    "Preserve the scope and terminology of the original question and do not broaden "
    "the plan into additional procedures, complete interface or message definitions, "
    "parameters, failure cases, stages, actors, branches, or related requirements "
    "unless they are necessary to answer the original question."
)

__all__ = ["PLANNER_INSTRUCTIONS", "PLANNER_PROMPT_VERSION"]
