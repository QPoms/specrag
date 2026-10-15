# Gap-directed corrective retrieval

Status: in progress

## Goal

Replace mandatory Judge-gap query rewriting with evidence-gap-driven
corrective retrieval. Query generation remains one available retrieval
technique rather than the corrective phase itself.

## Scope

1. Extend Judge assessments and `ResearchGap` with a structured evidence need:
   the smallest required proposition, missing semantic dimensions, evidence
   need type, and source-grounded anchors.
2. Prepare corrective context deterministically from the current graph state,
   including Judge-supported evidence, all source-valid candidates, and prior
   tool-call outcomes for the affected task.
3. Seed the corrective Researcher ledger with those complete candidates so it
   can reselect existing evidence, rerank it, follow a grounded clause, or
   expand local context before deciding that a new search query is necessary.
4. Keep one Researcher model turn per round and the existing independent
   corrective tool-call allowance. Do not add a Query Rewriter model node.
5. Validate the behavior with focused unit tests and live regressions 041 and
   049.

## Acceptance

- The Judge describes what evidence is missing without prescribing a query or
  retrieval tool.
- Corrective Researcher input contains complete prior source-valid candidates
  and identifies the Judge-supported active evidence.
- Existing candidates can be selected or used as navigation seeds in a
  corrective round.
- Prior tool calls are visible so corrective retrieval does not blindly repeat
  the original search.
- Query rewriting is optional and chosen only when new search is the suitable
  evidence-acquisition strategy.

## Initial implementation result

The active runtime now uses `research-state/v1.7` and
`researcher-controller/v2.2`. Non-sufficient Judge assessments carry a
structured proposition, missing dimensions, evidence-need type, grounded
anchors, and the evidence IDs that already support the remaining task units.
The corrective orchestration context restores all source-valid candidates and
the Judge-supported active set into the Researcher ledger, and exposes prior
tool calls for strategy selection.

The first live check succeeded for 049: T2 recovered the direct V-SMF to H-SMF
request and H-SMF to V-SMF response evidence, and the complete question ended
with `evidence_sufficient=true`. Question 041 remained incomplete: its T2
received the structured gap and spent the corrective allowance, but still did
not find a direct statement connecting the AMF load-rebalancing initial NAS
message to Registration Request. This is a remaining retrieval-navigation
quality issue, not a state-transfer or budget-accounting failure.

The Sufficiency Judge contract is now `evidence-sufficiency/v1.13`. It keeps
the direct-support requirement but explicitly permits compositional closure
across multiple source passages when each link is direct and the passages align
on actor, message/object identity, stage, direction, and condition. A
source-backed identity mapping is required; topical similarity alone remains
insufficient.

## Dedicated corrective prompt

The corrective pass now uses a separate `researcher-controller-gap/v1.0`
prompt contract. The initial Researcher prompt remains
`researcher-controller/v2.3` and no longer contains the corrective workflow.
Both modes still share the same controller, native tool registry, evidence
ledger, validation policy, and budgets; only the model instructions and prompt
receipt are selected by `gap_retrieval_mode`. The corrective prompt enforces
the sequence review existing evidence -> test the smallest missing proposition
-> navigate grounded sources -> search only when necessary -> submit when no
new evidence is being produced.

Focused regression tests for the prompt split and corrective orchestration pass
(`14 passed`), and structure/QA checks pass. The 041 rerun used the dedicated
gap prompt for every corrective model turn. It reduced the run to 12 retrieval
actions and shifted the search toward Reroute NAS Request, clause navigation,
and CM-IDLE context, but the Judge still found no direct source statement that
the load-rebalancing initial NAS message is specifically a Registration
Request. The prompt split is therefore functioning; 041 remains a corpus
evidence-closure limitation rather than a prompt-routing failure.

The subsequent 041 rerun with `evidence-sufficiency/v1.13` completed
successfully. The Judge selected the load-rebalancing passage from TS 23.501
§5.19.4 together with the Registration Request/Initial NAS mapping from TS
23.502 §4.2.2.2.2, and marked both T1 and T2 sufficient. This confirms that
the earlier failure was the single-passage closure rule rather than missing
retrieval evidence.
