# R17 development question set

Updated: 2026-10-10

## Objective

Create a small, expandable set of 50 Release 17 development questions for
improving the real Specrag workflow. The set is implementation guidance, not a
formal benchmark. Router labels and evidence scope are maintained independently;
all 50 questions are development data for workflow optimization.

## Storage contract

- `data/QA/questions.jsonl` is the single curated question file.
- `data/QA/README.md` documents provenance, licenses, fields, and curation.
- Complete upstream snapshots may be downloaded to `data/QA/telco-dpr/` and
  `data/QA/mm-telcobench/`, but those directories remain ignored by Git.
- Every accepted question is re-verified against the highest local Release 17
  version of the cited specification. Upstream answers and passage IDs are
  candidate hints, never final evidence.
- Diagnostic facts and evidence are never supplied to the Research Agent.

## First batch

- [x] Download the two upstream datasets without nested Git repositories.
- [x] Inspect schemas, question types, licenses, and source metadata.
- [x] Select six clear single-clause questions.
- [x] Select four genuine cross-clause or cross-specification questions.
- [x] Bind every question to local R17 source path, hash, locator, and excerpt.
- [x] Initially reserve two of the ten questions from routine strategy tuning.
- [x] Validate the JSONL schema and all referenced local source artifacts.
- [x] Update repository status and run `python scripts/check.py`.

## Expansion gate

Expand from 10 to 50 only after the first batch proves that question wording,
R17 source identity, evidence boundaries, provenance, and split semantics are
usable by the end-to-end workflow.

The first-batch schema and source-binding checks passed. The project owner chose
to complete the remaining questions before running the workflow so the same
test run can cover the full set.

## Full-set expansion

- [x] Select 24 additional single-clause questions.
- [x] Select 16 additional cross-clause or cross-specification questions.
- [x] Initially keep 30 simple and 20 complex evidence-derived labels.
- [x] Review all 50 questions under the routing-only definition, resulting in
  31 `single_task/simple` and 19 `decompose/complex` labels.
- [x] Refine the boundary around question 035: one connected NH-handling chain
  remains simple even though its evidence is distributed across two clauses,
  resulting in 32 `single_task/simple` and 18 `decompose/complex` labels.
- [x] Relabel question 044 as simple because its request, acceptance/refusal,
  and feedback messages form one connected negotiation procedure, resulting in
  33 `single_task/simple` and 17 `decompose/complex` labels.
- [x] Relabel question 017 as simple because 5GMM and 5GSM protection are parts
  of one NAS-security research objective, resulting in 34
  `single_task/simple` and 16 `decompose/complex` labels.
- [x] Make all 50 questions development data; collect a fresh external test set
  after the full workflow has been optimized.
- [x] Rebind every added question to the highest local R17 source version.
- [x] Review expected answers and facts against the bound excerpts.
- [x] Validate all 50 records and run the canonical quality gate.
