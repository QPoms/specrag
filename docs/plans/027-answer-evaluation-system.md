# Answer evaluation system

Status: implemented locally; legacy baseline artifact compatibility added

## Goal

Establish a reproducible offline evaluation path for final research answers.
This milestone evaluates answer content and source grounding only. Runtime
performance, token cost, Worker throughput, and Memory ablations remain out of
scope for the first benchmark.

## Evaluation layers

1. Deterministic answer alignment: map the final answer's cited evidence to the
   QA record's expected source targets using document hash, source path, and
   overlapping line locators. Historical claim-based artifacts remain readable.
2. Deterministic answer integrity: measure non-empty answers, claim citation
   validity, supported-claim rate, target coverage, target recall, and the
   fraction of questions whose annotated targets are all covered.
3. Optional structured review: retain the existing LLM rubric for factual
   completeness, evidence faithfulness, scope precision, and clarity. Its
   scores are reported separately from deterministic evidence metrics.

## Dataset contract

The existing `data/QA/questions.jsonl` remains the source of truth for the
first Release 17 development split. Each record must retain the question,
expected facts, expected answer, exact document hash, version, path, clause,
line range, and excerpt. A later benchmark expansion may add validation and
test splits without changing the evaluator contract.

## Completion gate

- Add a pure `specrag.evaluation` answer-alignment and summary implementation.
- Add `scripts/evaluate_answers.py` as the single deterministic offline runner.
- Emit one per-question JSON result and one aggregate report grouped by
  difficulty and reasoning scope.
- Cover complete answers, partial target coverage, unknown citations, skipped
  and failed artifacts, and duplicate/reordered citations in tests.
- Do not call an LLM from the deterministic runner.
- Run `python scripts/check.py` and preserve the existing CLI/runtime behavior.

The evaluator accepts both the current flat synthesis-artifact contract and
the retained `specrag.baseline-run/v1` envelope, where synthesis and evidence
are nested under `state`. This allows the first benchmark to use the existing
50-answer batch without rewriting generated artifacts.
