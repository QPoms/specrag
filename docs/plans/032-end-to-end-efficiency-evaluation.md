# End-to-end and efficiency evaluation

Status: complete; ten-question pilot and manually reviewed 50-question formal
baseline finished on 2026-10-18

## Goal

Establish a reproducible baseline for the current complete SpecRAG workflow
over the existing 50 answerable Release 17 development questions. The quality
score is one binary metric, **Task Success Rate**. The same Runs provide the
efficiency record: end-to-end latency, total token usage, every observed LLM
call and its token usage, and every tool call.

This milestone evaluates the real workflow from the user question through
Router, Planner, Researcher, Source Validator, Sufficiency Judge, and
Synthesizer. It does not inject reviewed intermediate tasks or gold evidence.

## Frozen scope

- Use all 50 records in `data/QA/questions.jsonl`; every case is expected to be
  answerable from its frozen Release 17 corpus scope.
- Reuse the reviewed Answer Units and source targets from the component
  evaluation as evaluation references, not as runtime inputs.
- Run the current supported workflow and configuration. Record the Git commit,
  prompt/model identifiers, budgets, question-file hash, and exact corpus
  scope with the baseline.
- Run one formal attempt per question after the pilot. Repeated-run variance is
  deferred.
- Run benchmark questions sequentially so separate questions do not contend
  with one another. Parallel work inside one normal SpecRAG Run remains
  enabled.
- Treat the populated retrieval/index cache as a warm-cache baseline and state
  that condition explicitly in the report.

True corpus-insufficient or unanswerable challenge cases are outside this
milestone. Cost estimation, component scores, throughput/load testing, and a
weighted quality score are also out of scope.

## Task Success contract

Each question receives exactly one reviewed Boolean `task_success` decision.
The system's own Sufficiency Judge cannot assign this evaluation label.

For these 50 answerable questions:

```text
Task Success =
    normal final answer produced
    AND core conclusion answered
    AND no material error or unsupported conclusion
    AND core conclusions have correct supporting citations
    AND specification version and corpus scope comply with the request
```

The conditions mean:

1. **Normal final answer produced**: the Run reaches its supported completed
   state and returns a non-empty user-facing answer. Runtime failure, budget
   exhaustion, or a terminal insufficient result is a failure because every
   benchmark case is known to be answerable in scope.
2. **Core conclusion answered**: the answer states the facts, relationships,
   conditions, actors, and outcomes necessary to resolve the user's question.
   Semantically equivalent wording is accepted. Missing optional background,
   examples, or stylistic detail does not fail the task.
3. **No material error or unsupported conclusion**: the answer contains no
   contradiction, wrong actor/action/condition/outcome, or unsupported claim
   that changes the practical meaning of the answer. Harmless wording issues
   and non-material omissions do not fail the task.
4. **Correct support for core conclusions**: every core conclusion has at
   least one correctly attached eligible citation whose evidence supports it.
   A harmless redundant citation is diagnostic; a missing, irrelevant, or
   misleading citation for a core conclusion fails the task.
5. **Version/scope compliance**: cited evidence belongs to the Run's frozen
   corpus scope and satisfies the requested Release or exact-version
   constraint. Compliance is checked from evidence provenance rather than
   surface text.

The evaluator records failure reasons without turning them into additional
quality metrics:

- `runtime_failure`
- `missing_core_conclusion`
- `material_error`
- `material_unsupported_claim`
- `missing_core_citation`
- `incorrect_core_citation`
- `version_scope_violation`

Minor presentation defects may be retained as review notes but do not change
the Boolean decision.

## Adjudication method

Use deterministic checks for Run completion, answer presence, citation ID
integrity, evidence eligibility, provenance, and corpus-scope compliance. Use
a versioned structured semantic judge to propose core-conclusion coverage,
material-error, unsupported-claim, and citation-support judgments against the
reviewed references and the evidence actually available to the answer.

The structured judge proposes a decision; it does not promote its own output
to reviewed status. The ten-question pilot and all 50 formal Task Success
decisions receive a manual content audit. Reviewed records retain reviewer
identity, review time, judge/model metadata, source Run ID, and artifact hashes.

The aggregate quality result is only:

```text
Task Success Rate = reviewed successful tasks / all 50 attempted tasks
```

No failed, skipped, or budget-exhausted Run may be removed from the denominator.

## Efficiency record

Efficiency is extracted from the same end-to-end Runs and introduces no new
model calls. Each per-question record contains:

- end-to-end wall-clock latency;
- aggregate input tokens, output tokens, and total tokens;
- LLM call count;
- one ordered record per LLM call containing its stage/purpose, model, input
  tokens, output tokens, total tokens, latency, and available retry metadata;
- tool call count;
- one ordered record per tool call containing its ResearchTask when available,
  initial/corrective phase when available, tool name, and execution outcome.

The summary reports:

- latency mean and P95;
- total token sum, mean, and P95;
- LLM call count sum, mean, and P95;
- tool call count sum, mean, and P95.

Per-call token and tool records remain drill-down data rather than additional
metrics. Input and output token totals remain separately visible. The runner
must record telemetry completeness and must not silently treat missing usage as
zero in a formal baseline.

## Implementation sequence

### Phase 0: freeze contracts

1. Add versioned per-case and summary models under `specrag.evaluation`.
2. Define the reviewed binary Task Success judgment contract and failure-reason
   vocabulary above.
3. Define one efficiency extraction contract over persisted Run events and
   model receipts.
4. Keep generated Runs and reports under ignored artifact directories.

### Phase 1: implement the offline evaluator

1. Extend the existing answer-evaluation path instead of creating a parallel
   benchmark implementation.
2. Add deterministic completion, citation-integrity, evidence-eligibility, and
   version/scope checks.
3. Add the structured semantic-judgment input/output contract.
4. Extract total and per-call token usage plus tool-call details from persisted
   Run events.
5. Add focused tests for success, material failure, minor non-failing defects,
   runtime failure, missing telemetry, invalid citations, and scope violations.

### Phase 2: ten-question pilot

Run the complete workflow on the existing representative set:

- single-task references: `001`, `002`, `005`, `019`, and `034`;
- decomposition references: `007`, `008`, `035`, `044`, and `046`.

For all ten cases:

1. retain the complete Run and event artifacts;
2. extract efficiency records;
3. create structured Task Success proposals;
4. manually review every decision and failure reason;
5. correct the rubric or implementation before the full run if material
   ambiguity or missing telemetry remains.

Pilot acceptance requires ten attempted Runs, ten reviewed decisions, no
silent skips, valid aggregate calculations, and complete or explicitly
accounted-for telemetry.

### Phase 3: formal 50-question baseline

1. Freeze the tested code, configuration, model/prompt identifiers, corpus
   scope, and input hashes.
2. Run all 50 questions once through the complete workflow.
3. Preserve failures in the denominator and preserve partial artifacts for
   diagnosis.
4. Produce structured Task Success proposals and manually audit all 50.
5. Generate the final Task Success Rate and efficiency summary only from the
   reviewed records.

### Phase 4: publish the baseline

1. Record the reproducible command, environment assumptions, configuration,
   artifact paths, Task Success Rate, and efficiency summary.
2. Document per-case outcomes and failure reasons without promoting those
   reasons to headline metrics.
3. Update `README.md`, `docs/current_status.md`, and this plan with the finished
   baseline and its limitations.
4. Run `python scripts/check.py` before declaring the goal complete.

Generated Runs, judgments, and reports remain ignored artifacts. Versioned
schemas, evaluator code, tests, commands, and concise baseline documentation
belong in Git.

## Definition of done

- all 50 questions have a fresh complete-workflow attempt from the same frozen
  baseline configuration;
- every attempt remains in the Task Success denominator;
- all 50 Task Success decisions are manually reviewed and auditable;
- the final report contains one Task Success Rate and no competing aggregate
  quality score;
- every Run has latency, total token usage, per-LLM-call token records, LLM
  call count, and tool-call records, or an explicit telemetry-incomplete error;
- the evaluator and aggregation paths have focused automated tests;
- the supported command and baseline are documented in the repository; and
- `python scripts/check.py` passes.

## Progress log

### 2026-10-18

- The long-running evaluation goal was activated and this metric contract was
  frozen.
- Complete-workflow receipt capture was added to the existing baseline runner
  so Router, Planner, Researcher, Sufficiency Judge, and Synthesizer calls can
  all contribute to total and per-call Token usage. Missing usage remains
  explicit rather than being converted to zero.
- Versioned Task Success, efficiency, semantic-draft, per-LLM-call, and
  per-tool-call records plus reviewed-only aggregation were implemented with a
  single offline runner.
- Focused evaluation, answer-quality, model-boundary, and planning tests pass
  (39 tests). The canonical quality gate also passes with 209 tests.
- A one-question live smoke attempt for `r17-qa-001` reached the configured
  model provider but received HTTP 403 `insufficient_user_quota` on the first
  structured call. The failed attempt was retained and correctly evaluates as
  `task_success=false`, `runtime_failure`, with missing Token telemetry recorded
  as `null`, not zero.
- At that point, the representative ten-question pilot remained blocked on
  usable model quota and the full 50-question run had not started.
- The baseline manifest now records the question-set SHA-256, selected IDs,
  Git commit and dirty state, exact budget, Release scope, index fingerprint,
  all 34 selected corpus documents and their hashes, warm-cache condition, and
  sequential question execution.
- Failed logical model calls now receive a safe receipt with stage/purpose,
  model, latency, attempt count, outcome, and failure kind. Missing provider
  Token usage remains `null`; an empty Token aggregate also reports `null`
  rather than a misleading zero.
- A second one-question quota probe for `r17-qa-001` again received HTTP 403
  `insufficient_user_quota` on the first Router call. Its manually reviewed
  result is a runtime failure with one LLM call, zero tool calls, 3212.058 ms
  end-to-end latency, and incomplete Token telemetry. This smoke result is not
  the ten-question pilot and is not a quality baseline.
- Exact corpus membership is now a per-question deterministic Task Success
  gate: every cited evidence path, version, Release, and document SHA-256 must
  match one document in that Run's frozen scope. The canonical quality gate
  passes with 217 tests.
- A third probe confirmed the newly supplied key is the value loaded from the
  project `.env`, but the configured `gpt-5.6-sol` still returned HTTP 403
  `insufficient_user_quota`. A diagnostic Router-only call to the configured
  `gpt-5.6-luna` fallback ended in HTTP 503 after its bounded retries. No model
  substitution was made, so the formal pilot configuration remains frozen.
- Token completeness is conservative across retries: because provider receipts
  expose usage only for the final successful response, any logical call with
  more than one API attempt makes the Run aggregate incomplete. Final-response
  usage and attempt metadata remain visible, but are not presented as total
  Run Token usage.
- Evaluation batches freeze their expected question IDs and input hashes in a
  manifest. Aggregation refuses missing or unexpected per-case records, so an
  offline Judge failure cannot silently remove a Run from the Task Success
  denominator. Resume also rejects a judgment when its source Run hash changed.
- Batch Token sum, mean, and P95 are emitted only when every reviewed case has
  complete Token telemetry. If even one case is incomplete, those aggregates
  are `null`; the complete-case count and all per-case observations remain
  available for diagnosis.
- Per-case evaluation verifies the Run's embedded question ID and question text
  against the QA record; a correctly named but misbound artifact is rejected.
- Provider diagnostics confirmed the current external blocker is channel
  assignment rather than local configuration. The loaded key can list models,
  but `gpt-5.6-sol` and `gpt-5.6-terra` return 403
  `insufficient_user_quota`; both the `gpt-5.5` alias and dated model return
  503 `model_not_found` on Responses, and a minimal Chat Completions request
  reports no available `gpt-5.5` channel for the key's distributor group.
  Account balance alone therefore does not make this key usable for the pilot.
- A replacement key restored the frozen `gpt-5.6-sol` channel. The formal
  ten-question pilot completed with 7/10 Task Success, 100% Token-telemetry
  completeness, 68.631-second mean latency, 147.932-second P95 latency,
  556,860 Tokens, 102 LLM calls, and 65 tool-call attempts.
- All 50 questions were then rerun through the complete workflow in one
  sequential warm-cache batch. There were 49 normal answers and one bounded
  iteration-budget failure (`r17-qa-039`); no question was omitted.
- The independent Judge produced all 50 proposals. Manual audit changed five
  over-permissive success proposals to failures because they omitted reviewed
  core Answer Units, then finalized all 50 decisions.
- The reviewed formal result is 38/50, or 76.0% Task Success. Simple cases
  score 29/34 (85.3%); complex cases score 9/16 (56.2%). Eleven failed cases
  omit at least one core conclusion, one Run failed at runtime/budget, and two
  cases also contain an incorrect core citation.
- Formal efficiency is 79.183 seconds mean and 191.969 seconds P95 latency,
  450 LLM calls (9.00 mean, 19 P95), and 289 tool-call attempts (5.78 mean,
  15 P95). Tool outcomes are 271 completed, seven no-hit, and 11 rejected.
- Token telemetry is complete for 49/50 Runs and totals 2,652,796 Tokens over
  those complete Runs. One `r17-qa-046` call required five attempts; usage for
  four failed attempts is unavailable, so the exact 50-Run aggregate remains
  `null`. The observed lower bound is 2,761,525 Tokens and is not reported as
  an exact total.
- The concise checked-in result and failure analysis is recorded in
  `docs/end_to_end_evaluation.md`; generated Run and evaluation artifacts stay
  under ignored `artifacts/` paths.
- The canonical quality gate passes: structure check, 50-question QA dataset
  check, and all 219 tests.
