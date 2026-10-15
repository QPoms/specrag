"""Offline prompt contract for evaluating generated synthesis answers."""

SYNTHESIS_QUALITY_PROMPT_VERSION = "synthesis-quality-evaluator/v1.0"

SYNTHESIS_QUALITY_INSTRUCTIONS = """\
You evaluate one generated answer to a 3GPP research question. This is an
offline quality assessment, not part of the runtime research workflow.

Use the original question as the answer scope. Use expected_facts and
expected_answer as benchmark references, not as text that the runtime model
was required to see. Use only the supplied Judge-supported evidence when
assessing whether the generated answer is source-grounded.

Assess four dimensions independently from 0 to 4:

1. factual_completeness: whether every substantive expected fact is covered.
   Score 4 when all are covered, 3 for a minor omission, 2 when a material
   answer unit is missing, 1 when most units are missing, and 0 when the
   answer is absent or contradicts the benchmark.
2. evidence_faithfulness: whether each normative or technical claim follows
   directly from the supplied evidence. Penalize unsupported inference,
   invented conditions, actors, values, or cross-specification relations.
3. scope_precision: whether the answer stays within the original question.
   Penalize material extra branches or conclusions even if they are true in a
   related specification.
4. clarity: whether the answer is coherent, concise, unambiguous, and uses
   the source terminology correctly.

For every expected fact, classify it as covered, partial, missing, or
contradicted and explain the decision briefly. List any material unsupported
claims and scope issues. Do not treat an alternative directly equivalent
source passage as missing merely because its clause differs from an annotated
target. Return only the requested structured assessment.
"""

__all__ = [
    "SYNTHESIS_QUALITY_INSTRUCTIONS",
    "SYNTHESIS_QUALITY_PROMPT_VERSION",
]
