# Offline synthesis quality evaluation

Status: complete

## Goal

Measure the semantic quality of final Synthesizer answers without changing the
runtime evidence boundary. Use the QA annotations only in the offline
evaluation layer.

## Scope

1. Add a versioned evaluator prompt and structured assessment model.
2. Compare each generated answer with the original question, annotated
   expected facts, and the exact Judge-supported evidence supplied to the
   Synthesizer.
3. Score factual completeness, evidence faithfulness, scope precision, and
   answer clarity independently on a 0--4 scale.
4. Persist per-question assessments and aggregate simple/complex summaries.
5. Keep insufficient Researcher cases excluded from answer-quality scoring.

## Acceptance

- Evaluation annotations never enter the Researcher or Synthesizer runtime
  payload.
- Every completed synthesis receives all four dimension scores and a verdict.
- Missing facts and unsupported claims are listed explicitly.
- Aggregate results distinguish structural generation success from semantic
  answer quality.

## Results

The 48 complete synthesis artifacts were evaluated with
`synthesis-quality-evaluator/v1.0`. Simple answers averaged 3.92/4.00 and
complex answers averaged 3.79/4.00. Across all 48 answers, 38 were rated
`pass`, 9 `review`, and 1 `fail`. The review/fail cases were retained with
explicit missing-fact or scope findings for follow-up rather than being
silently treated as successful.
