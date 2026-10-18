# Component evaluation system

Status: complete for the 50-question Release 17 development baseline

## Goal

Define a small, engineering-oriented evaluation system that isolates the
quality of each model-controlled research component. This is not a public or
academic benchmark: the immediate purpose is to locate regressions, compare
prompt or implementation changes, and make component failures explainable.

The existing answer evaluator in plan 027 remains the implemented offline
answer benchmark. This plan is a separate component-level milestone and does
not change runtime behavior.

## Evaluation principles

- Give each component one primary responsibility and only the minimum metrics
  needed to evaluate it.
- Isolate the component with fixed, reviewed upstream inputs. A Router error
  must not become a Planner error, and a Researcher query error must not become
  a Retriever error.
- Use deterministic scoring whenever source identity, locators, labels, or
  structured mappings make it possible.
- Humans establish and approve the gold annotations. An independent,
  structured LLM judge may perform repeated semantic mappings against that
  gold, with the first batch and ambiguous cases reviewed by a human.
- Do not use one opaque aggregate score across components. Report component
  metrics and their diagnostics separately.
- Record useful operational fields without promoting all of them to quality
  metrics.

## Shared annotation layers

The evaluation dataset needs three related but distinct semantic units:

- `ResearchUnit`: an independent research objective required by the original
  question. These units are the Planner gold.
- `EvidenceUnit`: a proposition that the evidence for one ResearchTask must
  establish, including any necessary actor, action, message, direction,
  condition, result, or relationship.
- `AnswerUnit`: a fact or relationship that the user-facing answer must state.

Existing QA `expected_facts` and exact evidence targets seed these annotations,
but a human must review the component-specific units. A gold decomposition is
not a required exact task list: more than one plan may validly cover the same
ResearchUnits.

## Component metric contract

### Router

The Router is evaluated on the binary decision `single_task` versus
`decompose`.

Primary metric:

- **Route Accuracy**: fraction of decisions matching the reviewed gold route.

Diagnostics:

- confusion matrix separating under-decomposition from over-decomposition;
- Macro-F1 derived from that matrix because the current development set is not
  class-balanced.

The existing `simple`/`complex` labels are annotation seeds, not unquestioned
gold. A human reviews all Router labels against the active rule: use
`decompose` only when the question contains multiple planner-separable,
independently source-verifiable normative conclusions. LLM-as-judge is not the
final Router adjudicator.

### Planner

Planner evaluation receives a reviewed complex question directly rather than
the Router's predicted output. It compares generated ResearchTasks with the
question's gold ResearchUnits without requiring one exact gold plan.

Metrics:

- **Task Coverage**: ResearchUnits covered by at least one generated task,
  divided by all required ResearchUnits. Also report the derived Full Task
  Coverage Rate.
- **Task Overlap Rate**: materially overlapping generated task pairs divided
  by all generated task pairs. Shared terminology, actors, or protocol context
  alone is not overlap; both tasks must seek the same normative conclusion.
- **Out-of-Scope Task Rate**: generated tasks that introduce an independent
  research objective not required by the original question, divided by all
  generated tasks.

Task atomicity, unnecessary or missing dependencies, answer leakage, and
unresearchable tasks are diagnostic error tags rather than first-stage
metrics. The evaluator uses a structured LLM judge to map units, task pairs,
and out-of-scope tasks; humans approve the gold units and review the initial
judgments.

### Single retrieval call

This evaluation covers one ranked search call. It receives a fixed, reviewed
retrieval request containing the ResearchTask, lexical query, semantic query,
optional target specifications, and an immutable corpus scope. It therefore
evaluates retrieval and ranking rather than Researcher query generation.
Clause navigation and context-expansion tools are not forced into ranked-search
metrics; their contribution is evaluated through the Researcher.

Metrics:

- **Target Recall@K**, reported at K=5 and K=12: annotated target evidence
  returned in the Top-K divided by all target evidence for that retrieval
  request. Recall@12 is the primary recall result. Source path, document hash,
  and overlapping line locators determine an exact target match.
- **nDCG@K**, reported at K=5 and K=12: ranking quality against relevance grades
  `2` (direct support), `1` (partial or necessary context), and `0` (topical or
  irrelevant). nDCG@5 is the primary ranking result.

Existing exact QA evidence targets begin at relevance 2. Candidate pooling
across retrieval variants supplies additional passages for structured LLM
relevance labeling; humans review all direct-support labels and a sample of the
other grades. The resulting qrels are frozen, after which metric computation
is deterministic. Precision@K and MRR are not first-stage metrics.

### Researcher

Researcher evaluation receives a fixed gold ResearchTask and evaluates the
active evidence set submitted downstream at the end of its bounded loop. It
does not score every observed candidate, Judge-selected evidence, or final
Synthesizer citations.

Metrics:

- **Evidence Coverage**: EvidenceUnits adequately established by the submitted
  active evidence set, divided by all EvidenceUnits for the ResearchTask. Also
  report Full Evidence Coverage Rate.
- **Evidence Precision**: submitted evidence items that directly support an
  EvidenceUnit or are necessary to complete a valid multi-passage evidence
  chain, divided by all submitted evidence items.

Background-only, analogous-protocol, wrong-branch, and non-contributing
duplicate passages are not precise evidence. An independent structured LLM
judge maps evidence to units and evaluates compositional closure; humans
approve the EvidenceUnits and review the initial judgments. Tool calls,
rounds, stop reason, budget exhaustion, candidate count, and submitted evidence
count remain diagnostics rather than quality metrics.

### Sufficiency Judge final evidence selection

This first component slice evaluates only the Judge's evidence-curation result
when the supplied, source-valid candidates are gold-sufficient. It does not
evaluate sufficient/partial/insufficient/conflict classification, gap quality,
or corrective retrieval.

The runtime may judge evidence per ResearchTask, but the scored Final Evidence
is the deduplicated union of all Judge-supported evidence after every task has
been processed and before synthesis. The main score is therefore at original-
question level; per-task mappings are diagnostics.

Metrics:

- **Final Evidence Coverage**: global EvidenceUnits covered by the merged Final
  Evidence, divided by all EvidenceUnits required by the original question.
  Also report Full Final Evidence Coverage Rate.
- **Final Evidence Precision**: final evidence items that directly support a
  global EvidenceUnit or are necessary to complete a valid evidence chain,
  divided by all merged final evidence items.
- **Evidence Overlap Rate**: final evidence items that can be removed without
  reducing global EvidenceUnit coverage or breaking a necessary evidence
  chain, divided by all merged final evidence items. Lower is better; textual
  similarity alone does not establish redundancy.

The fixed candidate pools contain complete support together with partial,
background, analogous, and duplicate passages. If the Judge fails to produce
a final selection for a gold-sufficient case, the case is not discarded and
its Final Evidence Coverage is zero. An independent structured LLM judge may
map and test the selected evidence, but humans approve the gold-sufficient
inputs and initial judgments.

### Synthesizer

Synthesizer evaluation receives the original question, fixed ResearchTasks,
fixed presentation controls, and an approved sufficient Final Evidence set.
It therefore measures answer generation rather than upstream retrieval or
selection failures.

Metrics:

- **Answer Coverage**: fully expressed AnswerUnits divided by all AnswerUnits
  required by the original question. A structured assessment retains
  `covered`, `partial`, and `missing`; the numeric score may assign 1, 0.5, and
  0 respectively. Also report Full Answer Coverage Rate.
- **Answer Groundedness**: atomic factual or normative claims fully supported
  by the complete Final Evidence input, divided by all factual or normative
  claims in the answer. Retain `supported`, `partial`, `unsupported`, and
  `contradicted` labels, but only `supported` enters the numerator.
- **Citation Correctness**: claims requiring evidence that have a complete,
  correctly attached supporting citation set, divided by all claims requiring
  evidence. A missing citation, unknown or ineligible ID, irrelevant citation,
  or incomplete multi-evidence chain fails the claim. This definition covers
  citation completeness without adding a fourth primary metric.

Groundedness checks whether support exists anywhere in Final Evidence;
Citation Correctness checks whether the claim's actual inline citations supply
that support. Scope, clarity, terminology, excessive detail, and language
issues remain structured diagnostics. Expected answers guide AnswerUnit
annotation but are not string-matching references; BLEU and ROUGE are not core
metrics.

## Deliberate exclusions

- Source Validator correctness remains a deterministic contract covered by
  focused tests and adversarial locator/hash/excerpt cases, not an LLM-scored
  component benchmark.
- Judge sufficiency classification, false-sufficient rate, gap diagnosis, and
  corrective-retrieval recovery are deferred from this first component slice.
- End-to-end completion, latency, token cost, Worker throughput, repeated-run
  variance, Memory ablation, statistical significance, and cross-dataset
  comparison remain separate milestones.
- No single weighted component score may hide a serious coverage, grounding,
  or citation failure.

## Implementation sequence

### Phase 0: shared foundation

1. Define one reviewed component-annotation schema over the existing Release
   17 development questions, with stable IDs and mappings between Research,
   Evidence, and Answer Units.
2. Add one supported offline component-evaluation entry point. Component
   implementations remain pure functions under `specrag.evaluation`; generated
   reports remain under ignored artifact directories.
3. Every report records the question/case IDs, corpus scope where relevant,
   artifact hashes, runtime prompt/model versions, annotation version and
   review status, per-case judgments, aggregate metrics, and skipped/errors.

### Phase 1: ten-question calibration pilot

Use the existing question-text-only Router review as an annotation seed and the
frozen `research-router/v3.7` outputs as predictions. The deliberately
challenging pilot is not a representative benchmark sample:

- reviewed `single_task`: `001`, `002`, `005`, `019`, and `034`;
- reviewed `decompose`: `007`, `008`, `035`, `044`, and `046`.

Questions `019` and `034` exercise over-decomposition, while `035` and `044`
exercise under-decomposition in the frozen v3.7 batch. The other six cases
verify unambiguous behavior. The frozen v3.7 alignment is 6/10, with two
errors in each direction.

Planner is evaluated only on the five reviewed `decompose` cases. Their
ResearchUnits and task mappings have been reviewed, and the schemas, metrics,
runner output, and review workflow operate without case-specific code.

### Phase 2: evidence-path pilot

For the reviewed pilot tasks:

1. Freeze one retrieval request per selected EvidenceUnit and pool Top-12
   results from the supported ranked retrieval variants.
2. Approve 0/1/2 qrels and compute Target Recall@5/@12 and nDCG@5/@12.
3. Run Researcher from the fixed tasks and score its final active evidence.
4. Build gold-sufficient, source-valid Judge pools with useful, partial,
   analogous, background, and duplicate evidence; score the merged Final
   Evidence at whole-question level.
5. Run Synthesizer from approved Final Evidence and score AnswerUnits, atomic
   claims, and their inline citations.

The evidence-path pilot may use fewer than all ten questions, but it must
include at least one single-task case, one multi-task case, and one
multi-passage compositional-evidence case.

The reviewed pilot baseline covers questions
001, 007, and 008. The selected set satisfies those three shapes. Retrieval is
scored on three fixed ranked calls, Researcher on four fixed tasks, and Judge
and Synthesizer on three whole-question cases. Reports record immutable source
artifact hashes, fixed corpus scope, input annotation/judgment hashes, schema
versions, review status, and available prompt/model metadata. The annotations,
qrels, and semantic judgments passed the delegated project review, and all six
reports were rerun without proposed-input overrides.

### Human promotion gate

Promotion from `proposed` to `reviewed` is an explicit annotation decision,
not an automatic consequence of passing tests. A reviewer must inspect:

1. the ten Router routes, especially 019/034 as `single_task` and 035/044 as
   `decompose`;
2. all ResearchUnits and the five Planner task mappings;
3. the EvidenceUnits and AnswerUnits for 001, 007, and 008;
4. every relevance-2 qrel plus the relevance-0/1 labels in the three retrieval
   pools;
5. all four Researcher evidence-set judgments, all three whole-question Final
   Evidence judgments, and all three Synthesizer claim/citation judgments.

Only after those decisions are accepted or corrected may the corresponding
records be changed to `reviewed` and rerun without `--allow-proposed`. The
runner is intentionally the enforcement point for this gate.

Every reviewed record must additionally contain `reviewed_by` and a timezone-
aware `reviewed_at`. Schema validation rejects unauditable reviewed records.
The compact decision packet is maintained in
`docs/component_evaluation_review.md`; it identifies the route conflicts,
semantic-unit ambiguities, evidence-path judgments, and their resolutions.

The first pilot gate completed on 2026-10-17 after the project owner delegated
the manual content audit to Codex. The auditable reviewer identity is
`project-owner-delegated:codex-manual-audit`. This qualifies the results as an
internal engineering baseline, not independent domain-expert or academic
adjudication. The audit split 044's accepted/refused outcome units, classified
non-essential 007/T2 RRC context and the secondary 001 procedure passage as
background, and adjudicated 017 as `single_task`.

### Phase 3: full development baseline

1. Extend reviewed Router labels and shared semantic units to all 50 Release 17
   development questions.
2. Freeze the full retrieval qrels, gold ResearchTasks, and gold-sufficient
   Judge candidate pools.
3. Add versioned structured offline-judge contracts and calibrate them against
   the human-reviewed pilot before using them on the remaining cases.
4. Run one baseline per component, grouped by route, difficulty, and reasoning
   scope where applicable; retain per-case diagnostics and do not combine the
   components into one score.
5. Update this plan and `docs/current_status.md`, then run
   `python scripts/check.py`.

The full 50-question annotation set is reviewed and contains all four semantic
layers. After adjudicating 017 as `single_task`, it contains 32 `single_task`
and 18 `decompose` labels. Against the frozen v3.7 Router it produces 0.90
Route Accuracy and 0.8927 Macro-F1, with three over-decompositions and two
under-decompositions. Routes 035 and 044 differ from the original QA
difficulty seed; 017, 019, 034, 035, and 044 are the five frozen-v3.7
disagreements.

Planner judgments for all 18 decomposition cases are also reviewed. Seven
LLM-drafted mappings were tightened because they covered only part of a
ResearchUnit. The formal Planner baseline reports mean Task Coverage 0.8241,
Full Task Coverage Rate 0.6111, and zero Task Overlap and Out-of-Scope Task
rates. Both full reports have `calibration_only=false`.

The full evidence-path baseline is also complete. Researcher was rerun from
the 78 reviewed GoldResearchTasks rather than inherited Planner output. The
Retriever benchmark freezes the first successful ranked call for every task;
target locators use overlap matching for recall, while per-result relevance
uses exact source/hash/line identity for nDCG. Judge receives a source-valid,
gold-sufficient pool containing the runtime evidence plus every exact target.
Synthesizer is isolated from Judge output and instead receives a fixed,
approved gold-sufficient evidence set with task-bound evidence IDs preserved.

All four full judgment files were manually audited under the delegated reviewer
identity and promoted to `reviewed`. Formal results are:

- Retriever, 78 calls: Target Recall@5/@12 0.8761 and nDCG@5/@12 0.8738.
- Researcher, 78 tasks: Evidence Coverage 0.9872 and Evidence Precision 0.6178.
- Judge, 50 questions: Final Evidence Coverage 0.98, Precision 0.908, and
  Overlap Rate 0.102.
- Synthesizer, 50 questions: Answer Coverage 0.975, Groundedness 0.9937, and
  Citation Correctness 0.9567.

Every report has zero evaluation errors, complete reviewed counts, and
`calibration_only=false`.

Fixed-route Planner artifacts now preserve the annotation's actual review
state. A proposed annotation is recorded as `proposed_annotation`, including
empty reviewer metadata, rather than being reported as a reviewed label. The
historical 017 two-task candidate is excluded because its route was adjudicated
as `single_task`.

### Definition of done for each component

A component is not complete merely because a score was printed. It needs:

- reviewed and schema-valid input annotations;
- a pure metric implementation and focused tests, including empty, partial,
  duplicate, skipped, and malformed cases as applicable;
- one documented command through the shared runner;
- immutable per-case and aggregate JSON reports;
- an initial human audit of semantic judgments and explicit unresolved cases;
- a retained baseline whose prompt/model/corpus and annotation versions are
  recorded.

## Design acceptance

- Router: Route Accuracy with confusion-matrix diagnostics.
- Planner: Task Coverage, Task Overlap Rate, and Out-of-Scope Task Rate.
- Single retrieval call: Target Recall@5/@12 and nDCG@5/@12.
- Researcher: Evidence Coverage and Evidence Precision.
- Sufficiency Judge final evidence: Final Evidence Coverage, Final Evidence
  Precision, and Evidence Overlap Rate at whole-question level.
- Synthesizer: Answer Coverage, Answer Groundedness, and claim-level Citation
  Correctness.
- The shared contract, all six component metric families, the unified runner,
  reviewed pilot, and reviewed full-development baselines are implemented.
