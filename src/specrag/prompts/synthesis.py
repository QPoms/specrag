"""Evidence-grounded final synthesis prompt contract."""

SYNTHESIS_PROMPT_VERSION = "evidence-synthesis/v1.5"

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

Return an ordered set of concise, answer-ready claims. Each claim must:

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
omitted as background. Conversely, before emitting an independent claim, check
whether removing it would still leave every part of the original question
answered without changing the meaning or scope of another claim. If so, omit
that independent claim.

Collectively, the claims must answer every substantive part of the original
question and cover every task question. At least one claim must cite supported
evidence owned by each task. Closely related findings may be combined when the
same cited passages support the complete combined statement.

The claim texts are the final answer. Write them in a coherent order as
self-contained prose, without claim labels, evidence IDs, or meta-commentary in
the text itself. Set the answer field to those same claim texts in the same
order, separated by blank lines. Do not repeat the same fact in multiple claims.
"""

__all__ = ["SYNTHESIS_INSTRUCTIONS", "SYNTHESIS_PROMPT_VERSION"]
