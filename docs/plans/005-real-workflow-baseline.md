# Real workflow baseline

Updated: 2026-10-11

## Objective

Connect the existing Release 17 retrieval, Researcher, evidence, and LangGraph
components through one supported application path, then establish the first
real baseline on the curated development questions. This is diagnostic work,
not a formal evaluation framework.

## Guardrails

- The Agent receives only the question ID, question text, and Release. Expected
  answers, expected facts, and evidence targets remain post-run diagnostics.
- All runs use one uniform Release 17 corpus scope.
- Runtime artifacts are written under ignored `artifacts/runs/`; generated
  indexes remain under ignored `.local/indexes/`.
- Release is not a Router or Planner input. Retrieval receives it later as one
  run parameter and resolves each specification to its highest local version
  within that Release.
- Product assembly extends the existing services. It must not duplicate
  retrieval, evidence, or orchestration implementations.
- Begin with one simple and one complex development question. Expand only after
  both complete with inspectable plans, tool calls, evidence, gaps, budgets,
  stop reasons, and final claim-to-evidence output.
- All 50 current questions are development data. A fresh external test set will
  be collected after end-to-end workflow optimization.
- Planner prompt experiments use the reviewed `difficulty` labels to select the
  simple or complex branch and never rerun Router. The label is diagnostic
  control input only; expected answers, facts, evidence targets, and Release do
  not enter Planner.

## Implementation

- [x] Add one application-level builder for the real Single Researcher graph.
- [x] Add a Router-only command whose artifacts live under
  `artifacts/runs/routing/<timestamp>/` and contain no Release or gold labels.
- [x] Add a standalone planning command that does not load the corpus index or
  execute retrieval.
- [x] Add a structured Router that selects `single_task/simple` or
  `decompose/complex` from question text alone. Full goal extraction and task
  construction belong exclusively to Planner; Router may internally assess a
  bounded independent-goal count solely to make the binary routing decision.
- [x] Generate one task for a simple question and 2-4 bounded tasks with
  explicit dependencies for a complex question.
- [x] Add a Gap Re-planner handler without placing business logic inside
  LangGraph nodes.
- [x] Fix the complex Research Planner contract at 2-4 ordered, pending tasks
  with validated acyclic dependencies; keep Release outside Planner input.
- [x] Move the versioned Research Planner prompt into `specrag.prompts` without
  changing its instructions.
- [x] Refine `research-planner/v1.3` so dependencies mean validated Researcher
  context rather than protocol or presentation order, and constrain every task
  to the minimum scope needed by the original question.
- [x] Pass each dependent Single or parallel Researcher only the predecessor
  evidence selected by a sufficient assessment and validated as supported.
- [x] Separate normal dependency-wave advancement from evidence-gap re-planning.
  A newly unlocked dependent task proceeds directly to Researcher without
  consuming `replans_used`; only an attempted incomplete or blocked task enters
  Gap Re-planner, while untouched dependent tasks remain pending.
- [x] Preserve Planner execution provenance through a structured response and
  write the generation mode, contract version, prompt version, model metadata,
  request/response hashes, token counts, and latency into planning artifacts.
  Simple deterministic plans explicitly carry no model-call receipt.
- [x] Validate the deterministic path on all 34 reviewed simple questions. Each
  preserves the original question in one dependency-free task and proceeds
  directly to Researcher without a second high-level Planner model call.
- [x] Make the first Researcher controller round the only model-based retrieval
  planning entry. Remove pre-loop anchor, facet, and suggested-query generation.
  The controller receives Planner tasks directly together with the registered
  tool capabilities, active corpus scope, dependency context, and budget; it
  generates the initial queries and executable calls. Later rounds observe
  evidence and adapt the next calls within the same bounded loop.
- [x] Add a planning-only Researcher boundary and `research-plan` CLI stage.
  Persist the exact model input, proposed calls, prompt/model receipt, and
  trusted per-action validation without building an index or invoking a tool.
  Plan dependency-ready tasks independently and record blocked dependency tasks
  as deferred.
- [ ] Wire the R17 retriever, Tool Registry, structured model adapters, source
  validation, sufficiency, conflict detection, and grounded synthesis.
- [x] Add one baseline command that selects only development records and writes
  a self-contained JSON artifact for each run.
- [x] Test that diagnostic fields are never included in routing/planning model
  inputs.
- [x] Run the canonical quality gate.

## Baseline sequence

- [x] Run Router alone on a small mixed development subset and inspect inferred
  difficulty and post-run label comparison.
- [x] Run the revised Router on all 50 questions under the earlier temporary
  40-development/10-holdout split.
- [x] Refine the Router around cohesive retrieval-and-evidence goals and rerun
  the same 50 questions. Accuracy improved from 29/40 to 34/40, but six complex
  cases were over-grouped and require a final boundary decision.
- [x] Remove goal extraction from Router, adopt the routing-only three-field
  contract, and rerun all 50 questions. The existing labels matched 31/40; label
  adjudication is required because they partly encode post-retrieval scope.
- [x] Move the active Router prompt and its version identifier into the dedicated
  `specrag.prompts` package; the Router service imports them instead of embedding
  prompt text beside provider calls.
- [x] Produce a question-text-only routing-label review draft for all 50 cases.
  It proposes 31 `single_task` and 19 `decompose` routes and differs from the
  existing difficulty field on three cases.
- [x] Accept the 50 reviewed Router labels, decouple them from evidence scope,
  and remove the temporary internal holdout split.
- [x] Define routing by independent research objectives rather than source or
  clause count, and relabel questions 017, 035, and 044 as simple. The active
  set contains 34 simple and 16 complex questions.
- [x] Run `research-router/v3.3` on all 50 questions. All calls completed; the
  outputs match 44 of the subsequently adjudicated labels, with three
  simple-to-complex and three complex-to-simple disagreements retained for
  review.
- [x] Run `research-router/v3.4` after its decomposition test is approved. It
  regressed to 37/50 by over-grouping end-to-end cross-boundary questions and
  is rejected as the Router baseline.
- [x] Implement `research-router/v3.5` as one structured independent-goal
  assessment followed by deterministic code mapping. Keep its public decision
  at three fields and retain the internal assessment only in routing artifacts.
- [x] Run `research-router/v3.5` on all 50 questions. All calls completed with
  valid output; it matched 43/50 (86%) and therefore did not exceed the v3.3
  result of 44/50 (88%).
- [x] Prepare `research-router/v3.6` with planner-separable normative
  conclusions, narrow direct-representation and branch exceptions, and explicit
  actor, interface, stage, and requirement-family boundaries. Do not run it
  until the prompt text is reviewed.
- [x] Run `research-router/v3.6` on all 50 questions. It matched 38/50 (76%):
  all 16 complex labels were recovered, but 12 simple questions were
  over-decomposed because separately verifiable attributes inside one contract
  were treated as separate Planner tasks. Reject v3.6 as the baseline.
- [x] Prepare `research-router/v3.7` with a same-object contract-cohesion rule:
  contract slots that one retrieval context can cover remain one task, while
  different research entry points or normative contexts favor decomposition.
  Accept at least 84% overall accuracy with a limited simple-to-complex bias;
  do not run until the revised prompt is reviewed.
- [x] Run `research-router/v3.7` on all 50 questions. It matched 47/50 (94%),
  correctly retained all 16 complex labels, and made only three accepted
  simple-to-complex errors (017, 019, and 034). Accept v3.7 as the Router
  baseline and move to Planner inspection.
- [x] Run the unchanged `research-planner/v1.0` prompt on the 16 reviewed
  complex development questions and inspect task boundaries and dependencies.
  With the accepted v3.7 complex decisions fixed, all 16 plans passed the
  structural contract: eight contained three tasks and eight contained four.
  Every plan added a final integration task dependent on all preceding factual
  tasks. These 16 synthesis-oriented tasks are not suitable independent
  retrieval objectives, so v1.0 is a diagnostic baseline rather than an
  accepted Planner prompt. A fresh Router call also classified question 048 as
  simple despite its accepted v3.7 complex result; the Planner-only 048 run
  therefore reused the accepted v3.7 route instead of treating that stochastic
  Router change as Planner behavior.
- [x] Prepare `research-planner/v1.2` around distinct research objectives while
  retaining genuinely researchable relationship tasks. Define dependencies as
  validated information flow, require dependency-ordered output, and enforce
  that every dependency references an earlier task. Do not run the revised
  prompt until its text and contract are accepted.
- [x] Run `research-planner/v1.2` on all 16 reviewed complex questions without
  invoking Router. All calls completed and passed the structural contract. The
  run produced 45 tasks: three questions had two tasks and thirteen had three,
  down from 56 tasks under v1.0. The automatic terminal integration task
  disappeared. Six questions contained nine dependency-bearing tasks; clear
  information dependencies and merely procedural ordering still require
  focused review before v1.2 is accepted.
- [x] Prepare `research-planner/v1.3` with the reviewed scope boundary and
  dependency semantics. The runtime now provides dependency task metadata,
  sufficiency rationale, and selected source-backed evidence to dependent
  Researchers through one shared orchestration helper; v1.3 has not yet been
  run on the focused dependency cases.
- [x] Run `research-planner/v1.3` on the six previously dependency-bearing
  questions. All six completed. Their task count fell from 18 to 14, while
  dependency-bearing tasks fell from nine to two. Questions 008, 042, 043, and
  047 now contain only independent tasks; 040 retains one source-backed mapping
  task dependent on its two inputs, and 048 retains one content-evolution task
  dependent on the signaling-path result. Task scope is materially narrower,
  with minor overlap in 040 still suitable for observation in the full run.
- [x] Run `research-planner/v1.3` unchanged on all 16 complex questions. All
  completed with valid structure, producing 38 tasks: ten two-task plans and six
  three-task plans. No generic terminal synthesis task or substantive coverage
  omission was found. Only questions 008, 045, and 048 retained dependencies,
  each tied to an explicit cross-specification relation, successful branch, or
  path-dependent content transformation. Accept and freeze v1.3 as the Planner
  baseline; observe the 045 branch dependency during real Researcher execution.
- [x] After Router behavior is accepted, run Planner on the reviewed simple and
  complex questions and inspect task boundaries and dependencies. All 34
  simple questions produced one deterministic task; all 16 complex questions
  passed the frozen v1.3 contract.
- [x] Replace the first-turn Researcher action contract with
  `researcher-controller/v1.4`: remove per-action rationale and
  expected-evidence prose, require direct tool coverage for every substantive
  answer unit, clarify specification scoping and tool-selection boundaries,
  and allow any number of actions that fits the explicit worker budget.
- [x] Add `hyde_search` as a thin retrieval tool. The controller supplies an
  untrusted hypothetical normative passage, the tool embeds it to retrieve real
  source chunks, and only those source chunks may become evidence.
- [x] Reject the v1.3 diagnostic first-turn run after one plan guessed an
  unavailable specification and another incorrectly narrowed a security rule.
  V1.4 binds a specification only when its exact number appears in the task or
  validated shared context, avoids keyword calls that merely repeat a hybrid
  anchor, and separates answer units likely governed by different rules.
- [x] Remove separate search and section-expansion quotas from the Researcher.
  Every executed retrieval capability now consumes the same bounded total tool
  budget across all controller rounds; per-kind counts remain audit metadata.
- [x] Run planning-only inspection on ten representative simple development
  questions. V1.4 produced 23 actions and all ten first turns passed trusted
  validation. Because none of the questions explicitly names a TS number, all
  calls correctly left specification scope open across the active Release.
- [x] Execute the exact 23 frozen calls without a second model round. All calls
  completed; strict source-path, document-hash, and line-overlap scoring found
  10/12 evidence targets and complete target coverage for 8/10 questions.
  Question 003 selected a neighboring sidelink-RLF procedure, while question
  035 found the resume rule but not the separate 33.501 Annex A.10 derivation.
- [x] Run the complete bounded Researcher loop for simple questions 003, 035,
  and 044. Question 035 covered both targets; 003 stopped at an answer-bearing
  passage that delegated SRB detail to clause 5.8.9.1a.3; 044 recognized its
  refusal-path gap but same-section expansion could not reach the adjacent
  unsuccessful-operation subsection before the action budget was exhausted.
- [x] Implement `researcher-controller/v1.5` without changing the rejected
  closure-assessment schema. Follow only answer-bearing normative references,
  permit clause identifiers and source specifications copied from trusted
  evidence, reject ungrounded clause/specification scope, and replace the old
  same-section capability with bounded source-context expansion across
  immediately adjacent sibling subsections.
- [x] Run the complete v1.5 loop on the same ten simple questions. All ten
  covered every annotated target (12/12 total); 044 reached both adjacent
  unsuccessful-operation subsections through source-context expansion, and no
  tool action was rejected or unavailable. Question 003 collected complete
  target evidence but ended at the three-round limit after its final expansion.
- [x] Run a fresh complete v1.5 loop on all 34 simple questions. It covered all
  targets for 32/34 questions and 36/38 targets overall, with no rejected,
  unavailable, or no-hit action. Later retrieval repaired five initially
  incomplete target sets, but 003 and 044 prematurely finished with different
  missing answer units despite succeeding in the earlier focused run. Do not
  freeze v1.5 until this finish-decision instability is addressed and retested.
- [x] Remove the overlapping evidence judgments before tuning closure quality.
  Rename the Researcher terminal action to `submit_for_validation`; retain only
  deterministic deduplication and source provenance checks in the
  `source_validator`; remove the per-evidence support model and separate
  conflict-model call; and make the task-level Sufficiency Judge solely
  responsible for selecting support, deciding completeness/conflict, and
  generating `ResearchGap` feedback for another bounded Researcher pass.
- [x] Run a live `researcher-controller/v1.6` compatibility check on simple
  questions 003, 033, and 044 without providing QA answers or evidence targets.
  All model actions passed trusted validation; 003 and 033 submitted in round
  two, while 044 used its six-action budget after filling the 12-item evidence
  ledger and is retained as a bounded-retrieval observation.
- [x] Replace the first-arrival 12-item ledger cap with the
  `researcher-controller/v1.7` candidate-pool/active-set contract. Preserve all
  unique tool hits for audit, require structured model selection of the bounded
  active set, permit later gap-closing evidence to replace weaker early items,
  and add one no-tools finalization turn after retrieval budget exhaustion.
- [x] Rerun simple question 044 with v1.7. The loop retained 12 active items
  from 15 candidates, promoted the late XnAP refusal passage (`E-T1:15`),
  discarded redundant and irrelevant early results, covered both annotated
  sources, and submitted after four tool actions without errors.
- [ ] Run planning-only inspection on three to five complex questions, including
  independent tasks and dependency-bearing cases; defer downstream tasks until
  validated predecessor evidence exists.
- [x] Run the first full Researcher-loop complex gate on question 007 using its
  frozen two-task dependency-free plan. Both isolated workers submitted without
  errors and their final active evidence covered all four annotated facts and
  both annotated source locations.
- [x] Run the first dependency-bearing diagnostic on question 045 with the
  existing six-call global budget. T1 passed source validation and the
  Sufficiency Judge with no gap, but consumed all six calls, so the budget guard
  stopped before the now-ready T2 could receive dependency context.
- [ ] Define complex-task budget allocation so one predecessor cannot consume
  the calls required by pending dependent tasks, then rerun 045 through T2.
- [ ] Run planning-only inspection on all 50 development questions after the
  simple and complex gates pass.
- [ ] Inspect failure modes and adjust only the assembly contract when needed.
- [ ] Run the full end-to-end development set only after planning and retrieval
  quality are separately accepted.
