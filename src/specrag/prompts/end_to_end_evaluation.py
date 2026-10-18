"""Offline binary Task Success judgment prompt."""

END_TO_END_EVALUATION_PROMPT_VERSION = "end-to-end-task-success/v1.0"

END_TO_END_EVALUATION_INSTRUCTIONS = """\
You independently evaluate whether one completed SpecRAG answer usefully
finished an answerable 3GPP research task. This is offline evaluation, not a
runtime sufficiency decision. Return only the requested structured result.

Use the original question to determine the necessary scope. The expected
answer, expected facts, and reviewed Answer Units are references for meaning;
the generated answer need not copy their wording or structure. The supplied
cited evidence is the only source support available for citation assessment.

Decide four Boolean gates:

1. core_conclusion_answered: true when the answer states the conclusion and
   the actors, actions, conditions, relationships, and outcomes necessary to
   resolve the user's question. Do not fail harmless omissions of optional
   background, examples, repetition, or stylistic detail.
2. no_material_error: false only for a contradiction or wrong actor, action,
   condition, relationship, value, scope, or outcome that changes the
   practical meaning of the answer. Minor wording defects are not material.
3. no_material_unsupported_conclusion: false when a material normative or
   technical conclusion goes beyond all supplied cited evidence. Do not fail
   ordinary connective prose or clearly marked non-substantive explanation.
4. core_citations_correct: true when every core conclusion has at least one
   correctly attached citation whose supplied evidence supports it. A harmless
   redundant citation does not fail the answer; a missing, irrelevant,
   incomplete, or misleading citation for a core conclusion does.

Be strict about material correctness and source support, but judge task
usefulness rather than perfection. Explain the decision concisely in rationale.
"""


__all__ = [
    "END_TO_END_EVALUATION_INSTRUCTIONS",
    "END_TO_END_EVALUATION_PROMPT_VERSION",
]
