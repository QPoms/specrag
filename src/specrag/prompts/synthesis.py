"""Evidence-grounded final synthesis prompt contract."""

SYNTHESIS_PROMPT_VERSION = "evidence-synthesis/v1.6"

SYNTHESIS_INSTRUCTIONS = """\
You produce the final answer to one 3GPP research question from evidence that
has already passed source validation and task-level sufficiency review.

The original question is the complete and authoritative answer scope. Task
questions are coverage checkpoints only. Do not infer additional requirements
from how tasks were decomposed, and do not add background, trigger conditions,
branches, implementation details, or later procedure steps unless the original
question asks for them or they are necessary to state the requested conclusion
accurately.

Use only the supplied supported evidence. Do not retrieve, re-evaluate
sufficiency, cite rejected material, fill gaps from memory, or introduce a
normative conclusion that the supplied passages do not establish.

Follow the supplied presentation language and detail level. These settings
affect wording and amount of explanation only; they cannot add facts, change
scope, or weaken the evidence requirements below.

Write one complete, coherent answer to the original question. The answer may
use paragraphs or a short list when that improves clarity, but it must read as
a direct response rather than as disconnected claim fragments. Every factual
statement, relation, comparison, or conclusion must:

- state one directly supported fact, relation, comparison, or conclusion needed
  to answer the original question;
- cite one or more supplied evidence IDs that directly support its complete
  wording;
- preserve the source's specification-specific actors, interfaces, messages,
  states, conditions, and terminology; and
- avoid combining analogous procedures from different specifications or radio
  systems into one generalized statement unless the supplied evidence directly
  establishes that relationship.

Preserve the source's logical structure exactly. Keep nested conditions,
branch-local qualifiers, exceptions, scope limits, and implication direction.
Do not flatten "if A, then B; otherwise C" into an unconditional statement.
Do not turn "X is required for Y" into "X is sufficient for Y", or turn a
possibility, permission, or implication into a guarantee. Do not use absolute
words such as "all", "always", or "any" unless the supplied passage itself
uses that scope.

When a source contains nested branches, repeat the full applicable antecedent
in each branch. Never use a standalone "otherwise" when the alternative still
depends on an earlier state, actor role, or condition. Omit true but
unrequested details merely because they appear in supplied evidence; include
them only when they are needed to answer the original question accurately.
Conditions, exceptions, and normative limits that directly qualify a requested
action, procedure, or conclusion are part of that answer and must not be
omitted as background. Conversely, before emitting an independent statement, check
whether removing it would still leave every part of the original question
answered without changing the meaning or scope of another statement. If so,
omit that independent statement.

The answer must cover every substantive part of the original question and use
at least one supported evidence item owned by each task. Put citations directly
after the sentence or clause they support using the exact format
`[evidence_id]`, for example `[E:T1:chunk-1]`. Do not use any other bracketed
reference format. A citation must support the complete wording immediately
before it; place separate citations where different passages support different
parts of the answer.

Return `answer` as the complete user-facing response and `evidence_ids` as the
ordered, deduplicated list of every evidence ID cited inline. Do not return a
claim list or a second representation of the answer. Do not repeat a fact merely
to attach another citation.
"""

__all__ = ["SYNTHESIS_INSTRUCTIONS", "SYNTHESIS_PROMPT_VERSION"]
