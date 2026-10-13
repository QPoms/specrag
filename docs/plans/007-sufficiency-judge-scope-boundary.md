# Sufficiency Judge task-scope boundary

Status: complete

## Objective

Make the Sufficiency Judge evaluate only the question assigned to a
`ResearchTask`. Planner metadata such as `objective` remains available to the
Planner, Researcher, and audit artifacts, but cannot add completeness
requirements at the evidence-judgment boundary.

## Implementation

1. Bump the Sufficiency Judge prompt contract and make the task question the
   authoritative scope.
2. Pass only `task_id` and `question` from `ResearchTask` to the Judge, along
   with source-valid evidence; keep the full task model in workflow state.
3. Add adapter and prompt regression tests proving that `objective` is not
   sent to the Judge and cannot be used as a sufficiency criterion.
4. Update the current-status documentation and run the canonical quality gate.

## Invariants

- Source validation remains deterministic and precedes the Judge.
- Evidence selection, gap creation, dependency context, and synthesis
  contracts remain unchanged.
- The Researcher and Planner continue to receive the full `ResearchTask`,
  including `objective`.
- The Judge still requires direct support for every proposition explicitly
  asked by `ResearchTask.question`, while allowing multiple passages to support
  a requested relation or description collectively.

## Acceptance

- The Sufficiency Judge payload contains no `objective`, `dependencies`, or
  task status fields.
- Prompt tests assert that scope comes only from `ResearchTask.question`.
- `python scripts/check.py` passes.
