"""Offline judge contracts for component-evaluation annotations."""

COMPONENT_ANNOTATION_PROMPT_VERSION = "component-annotation-judge/v1.1"

COMPONENT_ANNOTATION_INSTRUCTIONS = """\
You draft semantic gold units for one offline 3GPP component-evaluation case.
This is annotation work, not the runtime answer path. Use only the supplied
question, expected facts, and exact source targets.

If existing_research_units is non-empty, reproduce those ResearchUnits exactly
and build the other layers around them. They were already reviewed and must not
be renamed, rewritten, split, or merged.

Return four linked layers with stable IDs beginning at 1 and no gaps:

1. ResearchUnits (RU1, RU2, ...): independent research objectives required by
   the question. For a single_task route, create exactly one ResearchUnit. For
   a decompose route, create two to four genuinely separable objectives. Do not
   split a cohesive rule merely because it contains several fields. Include a
   relationship/integration unit only when the question explicitly asks how
   separate rules connect and that relation must itself be verified.
2. GoldResearchTasks (T1, T2, ...): one minimal task per ResearchUnit, in the
   same order. A task question and objective must not reveal the expected
   answer or an evidence locator. Do not name a specification, clause, version,
   source path, or line unless the original question names it. Add dependencies
   only when a later relationship task truly requires earlier source-specific
   results.
3. EvidenceUnits (EU1, EU2, ...): atomic propositions the evidence must
   establish. Preserve actors, direction, messages, conditions, outcomes,
   quantities, and cross-specification relationships. Every expected fact must
   be represented, and every proposition must be directly grounded by at least
   one supplied exact source target. Create only propositions necessary to
   establish an expected fact: do not turn every additional detail in a long
   source excerpt into a requirement. Use one or at most two EvidenceUnits per
   expected fact. Map each EvidenceUnit to every relevant ResearchUnit.
4. AnswerUnits (AU1, AU2, ...): minimum statements a correct user answer must
   express. Create exactly one AnswerUnit per expected fact, in the same order,
   and map it to the EvidenceUnits that support it. Do not add useful but
   unrequested source facts.

Do not judge a generated plan or answer. Do not copy source excerpts wholesale.
Return only the requested structured draft.
"""


COMPONENT_PLANNER_JUDGE_PROMPT_VERSION = "component-planner-judge/v1.0"

COMPONENT_PLANNER_JUDGE_INSTRUCTIONS = """\
You evaluate one generated research plan against independently supplied
ResearchUnits. This is an offline component judgment.

For every ResearchUnit, list all generated task IDs that materially cover it.
Use an empty task list when it is missing. A task covers a unit only if its
question/objective directs research toward the required normative conclusion;
shared terminology is insufficient.

List a task pair as overlapping only when both seek substantially the same
normative conclusion and either could be removed without losing a distinct
objective. Source-specific tasks plus a dependent integration task are not
automatically overlapping.

Mark a task out of scope only when it introduces an independent objective not
required by the original question. Extra wording, context, or search hints do
not alone make a task out of scope. Use diagnostic tags for material atomicity,
dependency, answer-leakage, or researchability issues without turning them into
new metrics.

Use only the supplied question, ResearchUnits, and generated tasks. Return only
the requested structured judgment.
"""


COMPONENT_RESEARCHER_JUDGE_PROMPT_VERSION = "component-researcher-judge/v1.0"

COMPONENT_RETRIEVAL_JUDGE_PROMPT_VERSION = "component-retrieval-judge/v1.0"

COMPONENT_RETRIEVAL_JUDGE_INSTRUCTIONS = """\
Create relevance judgments for one fixed ranked retrieval request belonging to
a reviewed GoldResearchTask. Use only the supplied task, EvidenceUnits, exact
gold target locators with excerpts, and ranked result candidates.

Select in target_locator_ids every supplied gold target passage that directly
establishes a required EvidenceUnit for this task. Select at least one. Do not
select a passage merely because it is relevant to another task in the original
question.

Grade every ranked candidate exactly once: 2 for direct support of a required
EvidenceUnit, 1 for partial support or context necessary to interpret or join
direct support, and 0 for topical, analogous, wrong-branch, or irrelevant
content. Direct support must preserve the necessary actors, direction,
conditions, action, and outcome. Refer only to the stable IDs supplied; never
invent or rewrite a locator. Return only the requested structured draft.
"""

COMPONENT_RESEARCHER_JUDGE_INSTRUCTIONS = """\
Evaluate the final active evidence submitted by Researcher for every supplied
fixed ResearchTask. This is offline component judgment, not runtime research.

Return exactly one task_judgment for every supplied task. Within each task,
assess every required EvidenceUnit exactly once and every submitted evidence
item exactly once. Use only the supplied evidence text and provenance.

An EvidenceUnit is covered only when the task evidence fully establishes its
actors, direction, conditions, action, and outcome. Mark partial when evidence
establishes a useful but incomplete part, and missing when it establishes none.
List only evidence IDs that actually contribute to that unit.

Classify an evidence item as direct_support when it independently establishes
one or more units, or necessary_chain when it is required with other passages
to establish a unit. Otherwise classify it as background, analogous,
wrong_branch, or duplicate and leave evidence_unit_ids empty. A passage is not
precise merely because it is topically related. Use diagnostic tags only for
useful error analysis. Return only the requested structured draft.
"""


COMPONENT_FINAL_EVIDENCE_JUDGE_PROMPT_VERSION = (
    "component-final-evidence-judge/v1.0"
)

COMPONENT_FINAL_EVIDENCE_JUDGE_INSTRUCTIONS = """\
Evaluate the deduplicated whole-question Final Evidence selected by the runtime
Sufficiency Judge. The supplied candidate pool is independently known to be
gold-sufficient; score only the runtime's selected Final Evidence.

Assess every required EvidenceUnit exactly once and every selected evidence
item exactly once. Covered requires complete support for the proposition;
partial means useful but incomplete support; missing means no selected support.
Use direct_support for an independently sufficient passage and necessary_chain
for a passage required as part of a multi-passage proof. Background, analogous,
wrong-branch, and duplicate items must not map to EvidenceUnits.

List in redundant_evidence_ids every selected item that can be removed without
reducing EvidenceUnit coverage or breaking a necessary evidence chain. Textual
similarity alone is not redundancy, but non-contributing selected items are
removable. Use only supplied evidence and return only the structured draft.
"""


COMPONENT_SYNTHESIZER_JUDGE_PROMPT_VERSION = (
    "component-synthesizer-judge/v1.0"
)

COMPONENT_SYNTHESIZER_JUDGE_INSTRUCTIONS = """\
Evaluate one answer generated from fixed, approved Final Evidence. Use only the
supplied question, AnswerUnits, answer text, eligible evidence, and inline
citations.

Assess every AnswerUnit exactly once as covered, partial, or missing. Then
decompose all factual or normative answer content into exhaustive atomic
claims with sequential IDs AC1, AC2, and so on. A supported claim must be fully
entailed by eligible evidence; partial, unsupported, and contradicted do not
count as grounded.

For every evidence-requiring claim, reproduce the evidence IDs actually
attached to that claim in the answer. Citation correctness is correct only when
that attached set completely supports the claim. Use missing for no citation,
unknown for an ineligible ID, irrelevant for citations that do not support the
claim, and incomplete when cited evidence supports only part of the claim or
omits a necessary chain member. Use not_required only for genuinely
non-evidentiary connective text. Return only the structured draft.
"""


__all__ = [
    "COMPONENT_ANNOTATION_INSTRUCTIONS",
    "COMPONENT_ANNOTATION_PROMPT_VERSION",
    "COMPONENT_PLANNER_JUDGE_INSTRUCTIONS",
    "COMPONENT_PLANNER_JUDGE_PROMPT_VERSION",
    "COMPONENT_RESEARCHER_JUDGE_INSTRUCTIONS",
    "COMPONENT_RESEARCHER_JUDGE_PROMPT_VERSION",
    "COMPONENT_RETRIEVAL_JUDGE_INSTRUCTIONS",
    "COMPONENT_RETRIEVAL_JUDGE_PROMPT_VERSION",
    "COMPONENT_FINAL_EVIDENCE_JUDGE_INSTRUCTIONS",
    "COMPONENT_FINAL_EVIDENCE_JUDGE_PROMPT_VERSION",
    "COMPONENT_SYNTHESIZER_JUDGE_INSTRUCTIONS",
    "COMPONENT_SYNTHESIZER_JUDGE_PROMPT_VERSION",
]
