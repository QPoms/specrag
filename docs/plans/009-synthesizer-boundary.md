# Synthesizer evidence boundary

Status: complete

## Goal

Make final-answer synthesis a narrow, evidence-only composition step after
every ResearchTask has passed the Sufficiency Judge. The original question is
the authoritative answer scope; Planner metadata must not broaden it.

## Scope

1. Move the versioned synthesis instructions from `infrastructure` into
   `specrag.prompts`.
2. Send only the original question, each task ID and task question, and the
   Judge-supported evidence grouped by task. Do not send Planner objectives,
   dependencies, statuses, search history, gaps, or rejected evidence.
3. Treat structured evidence-backed claims as the canonical answer units and
   derive the displayed answer deterministically from those claims.
4. Require every ResearchTask to be represented by at least one cited evidence
   item owned by that task, while retaining the existing eligible-evidence-ID
   validation.
5. Preserve specification-specific actors, interfaces, messages, and
   terminology; do not merge analogous procedures from different systems.

## Acceptance

- Synthesis cannot run before every task is sufficient.
- Only source-valid, Judge-supported evidence is supplied to the generator.
- Planner objectives and runtime metadata are absent from the model payload.
- Every task contributes cited evidence to the final claim set.
- `final_answer` contains exactly the ordered claim texts, so its factual
  content cannot diverge from the auditable claims.
- `python scripts/check.py` passes.

## Verification

The canonical gate was run from the repository `.venv` with an isolated pytest
temporary directory because the historical repository temp directory has
restricted Windows ACLs. Structure check passed, all 50 QA records passed the
dataset check, and all 109 tests passed.

Live workflow regression completed for simple questions 003, 033, and 044 and
frozen-plan complex questions 008, 048, and 050. Every run reached synthesis,
and the multi-task cases satisfied the new per-task claim coverage guard.
