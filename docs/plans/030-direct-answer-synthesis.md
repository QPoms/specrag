# Direct final-answer synthesis

Status: complete

## Goal

Make the Synthesizer produce the complete user-facing answer directly from
Judge-selected evidence. Do not create a second claim representation and do
not derive the answer by concatenating claim text.

## Scope

1. Replace the runtime synthesis contract with `answer + evidence_ids`.
2. Require inline `[evidence_id]` citations in the answer and validate that
   every declared citation is eligible and that every sufficient ResearchTask
   is represented.
3. Update the prompt, API projection, browser result view, scripts, and tests.
4. Keep read compatibility for historical JSON artifacts that contain claims.
5. Stop automatically deriving SpecFact proposals from synthesis claims;
   SpecFact extraction is a separate reviewed workflow.

## Acceptance

- Simple and complex questions use the same direct-answer synthesis contract.
- The model returns one coherent answer rather than ordered claim fragments.
- Unknown, missing, or task-incomplete citations are rejected.
- Run inspection exposes only evidence cited by the final answer.
- Historical claim-based evaluation artifacts remain readable.
- `python scripts/check.py` passes.

## Verification

- Canonical quality gate: structure and 50-question dataset checks passed;
  174 tests passed.
- Frontend TypeScript project check and production Vite build passed.
- Live acceptance passed for simple question `r17-qa-001` and complex question
  `r17-qa-008`. Both Runs completed with sufficient evidence and no replans or
  errors; the complex case covered all three ResearchTasks. Their direct-answer
  artifacts contain only `answer + evidence_ids`, and deterministic evaluation
  reported target recall 1.0 and citation validity 1.0 for both questions.
