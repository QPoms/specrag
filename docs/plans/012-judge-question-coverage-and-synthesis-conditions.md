# Judge question coverage and synthesis condition preservation

Status: complete

## Goal

Prevent a task-level sufficiency decision from overlooking a substantive part
of the original question, and prevent final synthesis from flattening
conditions, exceptions, or implication direction.

## Scope

1. Pass the original question to the Sufficiency Judge as a coverage guard;
   ResearchTask.question remains the task's authoritative local scope.
2. Require exact actor/action/object and requested message, state, branch, or
   return-path evidence instead of accepting broad topical paraphrases.
3. Strengthen synthesis instructions to preserve nested conditions, exceptions,
   and implication direction.
4. Re-run focused regressions 041, 049, 044, 003, 020, and 024.

## Acceptance

- Planner objectives and retrieval metadata remain absent from Judge and
  Synthesizer payloads.
- Judge marks an explicitly requested actor/action/message relation partial
  when evidence only states a broader related procedure.
- Synthesizer does not flatten conditional branches or reverse implications.
- Focused regressions retain valid direct evidence and improve semantic scores.

## Result

The Sufficiency Judge now receives the original question only as a coverage
guard while the ResearchTask remains its authoritative local scope. The
`evidence-sufficiency/v1.11` contract requires exact actor/action/message
relations and separately checks proposal, peer response, and initiator-side
handling or limits for requested negotiation procedures.

The `evidence-synthesis/v1.5` contract preserves nested antecedents,
exceptions, implication direction, and normative limits while rejecting
independent source facts that are not needed by the original question.

Focused live regressions produced the following semantic scores: 003 3.75,
020 3.50, 024 4.00, and 044 4.00. Questions 041 and 049 were correctly stopped
as partial because the supplied evidence did not directly establish the
requested Registration Request/load-rebalancing relation and selected PDU
Session Type return/handling relation, respectively. No unsupported final
answer was synthesized for either incomplete case.
