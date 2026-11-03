# Final-answer Judge

Status: active; prompt, scoring, execution policy, and saved-answer runner complete; calibration pending

## Goal

Evaluate only the final answer produced by one backend Research Workflow Run
against the frozen Benchmark Rubric. The model assigns one semantic band to
every Rubric Item; deterministic application code owns all arithmetic.

## Frozen input boundary

The Judge receives exactly:

1. Question;
2. Reference Answer;
3. Rubric; and
4. Candidate Answer.

Environment, Gold Evidence, difficulty, Workflow trajectory, retrieved
Evidence, citations, model identity, Tokens, cost, prior scores, and other
Judge outputs are excluded. Gold Evidence remains a Benchmark curation and
human-adjudication asset.

## Model output

For every input Rubric Item, in the same order, the model returns:

- exact `rubric_id`;
- `level` as `full`, `partial`, or `none`;
- optional Candidate Answer quotations or semantic summaries in
  `answer_support`; and
- a concise `reason`.

It also returns one `overall_reasoning` string. It must not return weights or
scores.

## Deterministic evaluation record

Application code validates the model output, attaches Run/Judge metadata, and
orders the formal record as follows:

```text
schema_version
question_id
run_id
score
rubric_assessments
overall_reasoning
judge_model
judge_prompt_version
```

`score` intentionally appears before `rubric_assessments`. For every item,
application code copies the Benchmark weight and calculates:

```text
full    = weight * 1.0
partial = weight * 0.5
none    = weight * 0.0
```

The final score is the item-score sum, ranges from 0 to 100, and is rendered
with one decimal place. The LLM never supplies or overrides it.

## Active Prompt

The canonical prompt is
`final-answer-rubric-judge/v1.1` in
`src/specrag/prompts/rubric_judge.py`. It defines semantic equivalence,
keyword rejection, item-local positive scoring, contradiction handling,
non-scoring answer-support audit notes, external-knowledge exclusion, citation exclusion,
and resistance to instructions embedded in evaluator inputs.

## Remaining work

1. Manually calibrate representative simple and complex cases before treating
   the automated scores as a reviewed baseline. The first complete API batch
   is diagnostic rather than human-approved.

## Frozen execution policy

- A missing or blank Candidate Answer bypasses the Judge and deterministically
  receives zero, with every Rubric Item recorded as `none`.
- A non-empty answer receives at most two top-level Judge calls using the same
  immutable four-field input.
- Invalid structured output, post-parse validation failure, timeout, provider
  failure, and unknown call failure are recorded as bounded attempt failures
  and retried once.
- A valid second response may produce the formal evaluation. If both attempts
  fail, the execution has `evaluation_failed`, a null score, and no formal
  evaluation. It is never converted to a zero.
- After deterministic output validation fails, the second call receives only
  bounded correction instructions derived from that failure. The immutable
  four-field semantic input is unchanged. `answer_support` is auxiliary audit
  text: it may quote or summarize the Candidate Answer, may be empty for any
  level, and is never substring-validated or used in scoring.
- Retrying judgment reuses the saved Candidate Answer and never reruns the
  Research Workflow.

## First complete API batch

The `final-answer-rubric-judge/v1.1` API run on 2026-11-03 scored all 50 saved
Workflow answers without a Judge execution failure. Forty-nine non-empty
answers were scored on their first model call; the one blank Workflow answer
was assigned the deterministic no-answer zero without a model call. The mean
score was 87.0 overall, 90.2 for simple cases, and 80.3 for complex cases.
These figures remain subject to the manual calibration above.
