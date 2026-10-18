# Component evaluation review packet

Status: all six component reviews completed over the Release 17 development set
on 2026-10-17

This packet records the review gate for the first component-evaluation pilot.
The project owner explicitly delegated the content audit to Codex. The records
therefore identify the reviewer as
`project-owner-delegated:codex-manual-audit`; this is transparent internal
engineering review, not independent domain-expert or academic adjudication.

## Decision rule

For Router, use `decompose` only when the question requires multiple
planner-separable and independently source-verifiable normative conclusions.
Do not decompose merely because a question contains the word `and`, names several
fields, or needs more than one supporting fact from one coherent rule.

For semantic judgments, approve the meaning rather than exact wording. A
ResearchUnit is an independent research objective, an EvidenceUnit is a
proposition that evidence must establish, and an AnswerUnit is something the
user-facing answer must express.

## Gate A: ten-question Router pilot

| ID | Proposed route | Frozen v3.7 | Review focus |
| --- | --- | --- | --- |
| 001 | `single_task` | `single_task` | One coherent Extended Connected Time rule. |
| 002 | `single_task` | `single_task` | Storage and use concern one credential set. |
| 005 | `single_task` | `single_task` | One procedure taxonomy and initiator mapping. |
| 007 | `decompose` | `decompose` | TS 23.501 identity rules and TS 38.331 representation are separate objectives. |
| 008 | `decompose` | `decompose` | State rule, procedure entry, and their cross-specification relation require integration. |
| 019 | `single_task` | `decompose` | Decide whether UE QoS mapping and the resulting RAN N3 header are one coherent uplink rule. |
| 034 | `single_task` | `decompose` | Decide whether RIM purpose and transparent core-network handling form one procedure overview. |
| 035 | `decompose` | `single_task` | Decide whether the resume action and the cross-derivation freshness chain are independently researchable objectives. |
| 044 | `decompose` | `single_task` | Decide whether proposal, acceptance, and refusal feedback warrant separate research tasks. |
| 046 | `decompose` | `decompose` | Mutual TLS capability, TLS versions, and certificate requirements are separate requirement families. |

The reviewed pilot score is 0.60 accuracy and 0.60 Macro-F1. It has two
over-decompositions (019 and 034) and two under-decompositions (035 and 044).

## Gate B: full-set Router conflicts

The 50-question set contains 32 `single_task` and 18 `decompose` labels. All 50
routes were manually reviewed. Question 017 was adjudicated as `single_task`:

- Its 5GMM protection forms and related 5GSM piggybacking mechanism occur in
  one cohesive NAS security rule and one source locus. The earlier two-task
  Planner artifact is retained only as historical calibration output.

Questions 035 and 044 also differ from their original QA difficulty seeds.
The original difficulty field is only a seed and must not override this
review.

## Gate C: semantic units

All 50 records in `data/QA/component_evaluation_release17.jsonl` now contain
reviewed ResearchUnits, GoldResearchTasks, EvidenceUnits, and AnswerUnits.
The audit checked each layer against the question, expected facts, expected
answer, and exact local evidence targets. It also checked sequential IDs,
cross-layer references, route shape, and one AnswerUnit per expected fact.

| ID | ResearchUnits | EvidenceUnits | AnswerUnits | Main question for review |
| --- | ---: | ---: | ---: | --- |
| 001 | 1 | 2 | 2 | Do the units capture both the named duration and all inputs that determine it? |
| 007 | 2 | 4 | 4 | Is the PLMN-ID plus CAG-ID identity relationship an independent required unit? |
| 008 | 3 | 4 | 3 | Is the cross-specification connection a real research objective rather than duplication of the two source-specific objectives? |

For Planner, also review the ResearchUnits for 035, 044, and 046. The five
pilot task mappings are in
`data/QA/component_planner_pilot_judgments.jsonl`. Question 044 intentionally
uses separate ResearchUnits for acceptance and refusal so a plan that omits
one branch cannot receive full Task Coverage; the corresponding tasks are not
overlapping because they seek different normative outcomes.

The 017 Planner candidate remains outside the benchmark because the reviewed
route is `single_task`.

The other 13 decomposition judgments were drafted with
`component-planner-judge/v1.0` and then reviewed manually. Seven questions had
an overly permissive LLM mapping corrected: 010, 038, 045, 047, 048, 049, and
050. In each correction, a task covered part of a ResearchUnit but omitted a
required subcondition; partial coverage was not counted as full Task Coverage.
The final 18-question judgment set is
`data/QA/component_planner_release17_judgments.jsonl`.

The formal full-set reports are not calibration runs:

- Router, 50 questions: Route Accuracy 0.90 and Macro-F1 0.8927, with three
  over-decompositions and two under-decompositions.
- Planner, 18 decomposition questions: mean Task Coverage 0.8241, Full Task
  Coverage Rate 0.6111, Task Overlap Rate 0.0, and Out-of-Scope Task Rate 0.0.

## Gate D: evidence-path judgments

The following files were reviewed in this order:

1. `data/QA/component_retrieval_pilot_judgments.jsonl`: three ranked calls,
   including every relevance-2 target and every relevance-0/1 pooled result.
2. `data/QA/component_researcher_pilot_judgments.jsonl`: four task-level active
   evidence sets. The two surrounding RRC structure passages for 007/T2 are
   classified as background because the direct field definition already
   establishes both required units; this is the main precision decision to
   confirm.
3. `data/QA/component_final_evidence_pilot_judgments.jsonl`: three merged
   whole-question selections. For 007, three extra RRC passages are currently
   judged removable; for 008, three passages are treated as a necessary
   cross-specification chain.
4. `data/QA/component_synthesizer_pilot_judgments.jsonl`: three answers. The
   only non-full AnswerUnit is 007/AU4, currently `partial` because the answer
   implies rather than explicitly states that PLMN ID plus CAG ID identifies
   the PNI-NPN.

The full-set review then audited and accepted these files:

1. `data/QA/component_retrieval_release17_judgments.jsonl`: 78 fixed ranked
   calls. All 103 returned relevance-2 passages were checked against their
   task and source excerpt; every low-recall or low-nDCG case was inspected.
   Thirteen tasks genuinely missed at least one exact target, while some still
   retrieved an alternative directly supporting passage.
2. `data/QA/component_researcher_release17_judgments.jsonl`: 78 GoldTask
   evidence sets. Every task had submitted evidence and a successful ranked
   call. The only non-full coverage cases are 044/T2 and 044/T3, which both
   omit the shared response-before-execution and OAM-range constraint.
3. `data/QA/component_final_evidence_release17_judgments.jsonl`: 50 merged
   selections from independently gold-sufficient pools. The only non-full
   EvidenceUnits are 003/EU1 and 029/EU2; both selected passages require an
   unselected referenced clause to establish the complete proposition.
4. `data/QA/component_synthesizer_release17_judgments.jsonl`: 50 answers from
   Judge-independent fixed evidence. The audit retained six partial/missing
   AnswerUnits, two non-fully-supported claims, and sixteen claim-level
   citation failures. Citations are scored at their actual claim attachment;
   a later citation in the paragraph does not automatically cover an earlier
   uncited claim.

One draft-only defect was rejected during review: provenance deduplication had
removed valid task-bound evidence IDs from the Synthesizer judge input and
created false `unknown` citations. The judge input now preserves every eligible
task-bound ID, its regression test passes, and all Synthesizer judgments were
regenerated before promotion.

The formal full evidence-path reports have complete reviewed counts, zero
errors, and `calibration_only=false`: Retriever Recall@12 0.8761 and nDCG@5
0.8738; Researcher Coverage 0.9872 and Precision 0.6178; Judge Coverage 0.98,
Precision 0.908, and Overlap 0.102; Synthesizer Coverage 0.975, Groundedness
0.9937, and Citation Correctness 0.9567.

## Approval record

All four gates were accepted after these corrections:

- 044 acceptance and refusal became separate ResearchUnits;
- the two non-essential 007/T2 RRC structure passages became `background`;
- the secondary 001 procedure passage became `background` rather than
  `duplicate`;
- 017 became `single_task`.

The original pilot records use `2026-10-17T20:51:17+08:00`; the completed
full-set annotation and Planner review uses `2026-10-17T21:19:58+08:00`; the
full evidence-path review uses `2026-10-17T23:35:01+08:00`.
All accepted records contain the delegated reviewer identity. The six full
component reports were generated without proposed-input overrides.
