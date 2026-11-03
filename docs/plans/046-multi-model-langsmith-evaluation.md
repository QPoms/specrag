# 046 Multi-model and LangSmith evaluation

Status: implementation complete; live capability probes pending credential rotation

## Objective

Extend the frozen Release 17 final-answer evaluation into one reproducible
model-comparison experiment without redesigning the product Research Workflow
or duplicating the existing Benchmark, Workflow Runner, receipt capture, or
final-answer Rubric Judge.

The first experiment runs every selected Benchmark case once with each active
candidate model:

- required: `gpt-5.6-sol`
- required: `gpt-6-astra`
- required: `deepseek-v4.1-flash`
- optional: `claude-opus-5`

The fixed final-answer Judge is `deepseek-v4-pro`. The experiment records Judge
scores and execution statistics only. It does not calculate Task Success,
Pass@k, Pass^k, or a score threshold.

## Frozen decisions

1. Every model-backed Workflow stage uses the candidate model for that
   experiment cell. Router, Planner, Researcher, Sufficiency Judge, and
   Synthesizer therefore share one model. Deterministic Source Validator,
   scheduling, budget, and publication logic remain unchanged.
2. Candidate and Judge adapters have no cross-model fallback. Same-model
   bounded retries remain permitted and are recorded.
3. Every model is probed through the real structured-output boundary. Candidate
   models are also probed through the Researcher native-tool/replay boundary.
   A failed required probe prevents the formal batch from starting. A failed
   optional Claude probe is recorded as `skipped_optional`.
4. The fixed Benchmark remains `specrag-release17-final-answer` `1.0.0` with
   `trial_count=1`. Multi-model selection is experiment metadata and does not
   change the Benchmark version.
5. Only the Benchmark Workflow projection crosses into the Research Workflow.
   Reference Answer, Rubric, Gold Evidence, and Judge data remain evaluator-only.
6. LangSmith is an observability copy. Local Run artifacts, model receipts,
   evidence state, and Judge executions remain authoritative for evaluation.
7. Formal runs are sequential by model and question over the warm immutable
   Release 17 Retriever. Development can proceed in parallel, but the actual
   experiment must not introduce cache or provider-concurrency differences.
8. Generated probes, Runs, Judge results, and reports live below ignored
   `artifacts/`; no generated result is written under `src/`.

## Experiment contract

One frozen experiment manifest records:

- experiment ID and schema version;
- Benchmark ID, version, ordered case IDs, and source hashes;
- candidate model order and required/optional policy;
- fixed Judge model and Judge Prompt version;
- `trial_count_per_model=1`;
- exact Research budget and sequential execution policy;
- fallback policy;
- Git revision and dirty state;
- exact Release 17 document snapshot and index fingerprint;
- safe LangSmith project/correlation metadata;
- capability-probe results and active-model set.

Every candidate-model configuration is an immutable `AppConfig` copy that sets
the default, planning, and summary roles to the same model and clears
`api_fallback_models`. The Runner rejects any receipt whose final or attempted
model differs from the experiment cell.

## Artifact layout

```text
artifacts/evaluations/model-comparison/<experiment-id>/
|-- manifest.json
|-- probes/<model-slug>.json
|-- runs/<model-slug>/<question-id>.json
|-- judgments/<model-slug>/<question-id>.json
|-- model-summaries/<model-slug>.json
`-- comparison.json
```

Writes are atomic. Resume accepts an artifact only when its experiment,
Benchmark case, candidate model, trial, source manifest, and frozen inputs
still match. Truncated, stale, mismatched, missing, or extra artifacts fail
closed instead of silently changing the denominator.

## LangSmith boundary

Project `.env` values are loaded only through an allowlisted infrastructure
adapter and are never written to artifacts or logs. When tracing is enabled,
the LangGraph root run receives a stable correlation ID, run name, tags, and
metadata containing only:

- experiment and Benchmark identifiers;
- phase (`workflow` or `judge`);
- candidate model;
- question ID and trial index;
- Git revision;
- local thread/Run correlation ID;
- corpus/index fingerprint.

Workflow trace metadata must not contain Reference Answer, Rubric, Gold
Evidence, or Judge output. Judge execution is a separate trace boundary rather
than a child of the candidate Workflow trace.

## Implementation work packages

### A. Provider-neutral evaluation contract

- Add strict experiment, model-target, probe, cell, and comparison-summary
  models under `evaluation`.
- Add deterministic completeness and cross-model score aggregation.
- Report overall/simple/complex observed averages, complete-batch averages,
  score coverage, common-case averages, and distinct Workflow/Judge failure
  counts.
- Do not add success or Pass metrics.

### B. Provider and LangSmith integration

- Add secret-safe LangSmith configuration and activation in `infrastructure`.
- Add candidate structured-output and native-tool/replay probes plus the Judge
  structured-output probe.
- Extend LangGraph invocation config with correlation ID, run name, tags, and
  metadata without importing LangGraph outside `orchestration`.
- Preserve no-op behavior when tracing is disabled and safe failure projection
  when tracing is unavailable.
- Ensure all OpenAI-compatible model calls can be associated with the active
  trace without changing their business contracts.

### C. Experiment Runner and CLI

- Provide one supported entry point:

  ```text
  python scripts/evaluate_model_comparison.py probe EXPERIMENT_DIR
  python scripts/evaluate_model_comparison.py run EXPERIMENT_DIR --resume
  python scripts/evaluate_model_comparison.py judge EXPERIMENT_DIR --resume
  python scripts/evaluate_model_comparison.py summarize EXPERIMENT_DIR
  ```

- Extend the existing baseline and final-answer Judge services; do not copy the
  Workflow or scoring implementation.
- Freeze the manifest before the first formal Run.
- Validate the Benchmark Environment against the live immutable corpus before
  any question executes.
- Record optional Claude exclusion without adding fake question failures.
- Add an explicit Judge model override fixed to `deepseek-v4-pro`.

### D. Coordinator review and documentation

- Review package directions, Gold isolation, fallback enforcement, failure
  taxonomy, resume behavior, and artifact hygiene.
- Update the README and current project status with the single supported path.
- Run focused tests followed by `python scripts/check.py`.
- Run live probes only after the compromised provider and LangSmith credentials
  have been rotated.

## Acceptance criteria

1. Tests prove that all model-backed roles use the requested candidate model
   and every fallback list is empty.
2. Candidate receipts containing another model are rejected as contaminated.
3. Required probe failure blocks execution; optional Claude failure is an
   explicit skip.
4. Judge configuration and persisted Judge receipts identify only
   `deepseek-v4-pro`.
5. Workflow input and trace metadata contain no evaluator-only Benchmark data.
6. Resume rejects model, question, trial, Benchmark, Git, budget, corpus,
   manifest, or source-artifact drift.
7. Aggregation rejects missing or extra active-model cases and distinguishes
   no-answer, Workflow failure, and Judge failure.
8. The report contains no Task Success, threshold, Pass@k, or Pass^k field.
9. Disabled LangSmith tracing leaves Workflow behavior unchanged; enabled
   tracing carries searchable correlation metadata without replacing local
   audit data.
10. The canonical repository gate passes.

## Operational safety note

During the initial read-only audit on 2026-11-03, a command accidentally wrote
project `.env` values into an internal tool log. No credential value is stored
in this plan or should be repeated elsewhere. Rotate the OpenAI-compatible
proxy and LangSmith credentials before any live probe or formal experiment.
