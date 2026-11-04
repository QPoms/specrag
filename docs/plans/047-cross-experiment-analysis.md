# 047 Cross-Experiment Analysis

## Objective

Add the first offline analysis layer for completed one-trial model-comparison
experiments. The analysis compares compatible `comparison.json` artifacts and
produces reproducible machine-readable and human-readable reports without
rerunning the workflow or the Judge.

## Frozen decisions

- Extend the existing `evaluation` package and
  `scripts/evaluate_model_comparison.py`; do not create a second evaluation
  workflow.
- Require the same Benchmark ID/version, Judge model/prompt version, ordered
  question set, and difficulty labels across all inputs.
- Treat `(experiment_id, model_id)` as the model observation identity.
- Report descriptive one-trial statistics only: score coverage, paired score
  deltas, median/mean deltas, and higher/equal/lower counts.
- Preserve `workflow_status` and `judge_status` per question. A `no_answer`
  score remains zero but is not conflated with an ordinary scored answer.
- Do not calculate Task Success, thresholds, Pass@k/Pass^k, statistical
  significance, or stage-quality scores.
- Record the path and SHA-256 of every source artifact in the output.
- Keep generated reports under ignored `artifacts/`; source code and tests
  contain no generated experiment output.

## Work packages

1. Add strict cross-experiment contracts, validation, aggregation, and
   deterministic Markdown rendering under `specrag.evaluation`.
2. Add an `analyze` subcommand to the existing model-comparison CLI with
   atomic, non-destructive JSON/Markdown output.
3. Add unit and CLI tests for compatibility gates, paired statistics,
   no-answer handling, output determinism, and overwrite protection.
4. Generate and inspect the first report for the completed `gpt-5.6-sol` and
   `gpt-6-astra` experiments.
5. Update the supported workflow documentation and current status.

## Acceptance criteria

- Compatible summaries produce `cross-experiment.json` and
  `cross-experiment.md` deterministically.
- Incompatible Benchmark, Judge, question order, or difficulty inputs fail
  closed before any output is written.
- Pair deltas have an explicit `right - left` direction and are independently
  reported for overall, simple, and complex slices.
- The report lists all per-question scores and statuses and highlights the
  largest positive and negative score changes.
- Existing Workflow, provider, Judge, and LangSmith behavior is unchanged.
- `python scripts/check.py` passes.

## Deferred work

- Stage execution and trajectory telemetry from Run artifacts.
- Gold-evidence hit analysis and calibrated stage-quality evaluation.
- Repeated trials, Pass@k/Pass^k, confidence intervals, and significance.
- Failure attribution and causal claims across workflow stages.
