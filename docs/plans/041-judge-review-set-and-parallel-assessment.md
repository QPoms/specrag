# 041 - Judge review-set and parallel assessment

Status: complete

## Goal

Reduce repeated Sufficiency Judge work without weakening evidence provenance or
changing the Synthesizer contract. Keep the complete historical candidate pool
for audit, but judge only the evidence that is relevant to the latest task
attempt.

## Invariants

- `candidate_evidence` is the append-only, source-attributed candidate pool.
- Each ResearchTask records the final active evidence set from its latest
  TaskExecution.
- A Judge review set is the stable union of the previous Judge-selected
  evidence and the latest Researcher-active evidence for that task. Conflict
  evidence selected by the Judge is retained by the same rule.
- Only source-valid evidence can reach the Judge.
- An unchanged review-set fingerprint does not trigger another model call; the
  prior assessment, Gap, and evidence validation remain authoritative.
- Independent task assessments in one workflow wave may execute concurrently,
  but their state, receipts, and events are merged in ResearchTask order.
- Final synthesis continues to receive the original question, task questions,
  and Judge-selected source-valid evidence. No intermediate LLM summary is
  introduced.

## Implementation stages

1. Add durable latest-active and Judge-review records to runtime state and
   preserve the complete Researcher candidate pool.
2. Build deterministic per-task Judge review sets and retain audit validation
   records outside the current review set.
3. Skip duplicate Judge calls when the canonical review fingerprint is
   unchanged.
4. Run independent Judge calls concurrently with isolated receipt capture and
   deterministic fan-in.
5. Add regression tests, update project status, and run `python scripts/check.py`.

## Out of scope

- Changing Judge or Synthesizer prompts.
- Adding a whole-Run wall-clock limit.
- Replacing the existing TaskExecution, Worker, or LangGraph lifecycle.
- Running a live model/corpus validation before the deterministic quality gate.
