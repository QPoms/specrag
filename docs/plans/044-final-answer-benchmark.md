# Final-answer benchmark

Status: complete; 50-case Benchmark V1 frozen and self-reviewed on 2026-11-02

## Goal

Build one versioned, auditable Benchmark for final-answer-only evaluation of
the complete backend Research Workflow over the existing 50 Release 17
questions. The Benchmark supplies Question, Reference Answer, positive Rubric,
Gold Evidence, frozen Environment, and simple/complex Task Metadata. It does
not define the LLM Judge implementation or score completed Runs.

## Decisions

- The system under test is the backend Research Workflow. Conversation routing
  and presentation are outside this Benchmark.
- The existing 50 development questions are the complete V1 case set. Every
  question is executed once by the later evaluation runner.
- Difficulty is the only reporting classification: `simple` or `complex`.
- A case has a maximum score of 100 and only positive Rubric items.
- Each Rubric item has a weight and predefined `full`, `partial`, and `none`
  bands. The default ratios are 1.0, 0.5, and 0.0. An indivisible item may
  disable the partial band.
- Correct semantic equivalents receive credit. Keyword presence alone does
  not.
- An absent, incorrect, or directly contradicted point receives no credit.
  Unrelated extra errors do not introduce deductions in V1.
- Gold Evidence is reviewed, known-sufficient, non-exclusive evidence. A later
  Judge uses it to understand the reference truth; the candidate is not
  required to retrieve or cite the same passage.
- Only Question plus Environment may cross into the Workflow. Reference
  Answer, Rubric, and Gold Evidence remain evaluator-only.
- The later Judge may return one direct 0--100 score. Judge prompting,
  calibration, and result schemas are a separate milestone.

## Canonical sources

Avoid a duplicate all-in-one data copy. The loaded BenchmarkCase is composed
from active, reviewed repository sources:

1. `data/QA/questions.jsonl` owns Question, Reference Answer, source metadata,
   and exact Gold Evidence locators.
2. `data/QA/component_evaluation_release17.jsonl` owns the reviewed semantic
   Answer Units used as Rubric criteria.
3. `data/QA/end_to_end_rubrics.jsonl` owns only the new per-item weights and
   partial-credit policy.
4. `data/QA/end_to_end_benchmark.json` freezes the case-set hashes and the
   exact Release 17 corpus environment.

For review convenience, `data/QA/end_to_end_benchmark_review.json` is a
deterministic materialization of the complete composed Benchmark. It is not a
second authority: the canonical check requires byte-for-byte agreement with
the loader's normalized output. The review layout is case-first and repeats
only each case's relevant specification documents; the full 34-document
execution Environment appears once at the end.

`specrag.evaluation.benchmark` validates and joins these sources into one
strict `BenchmarkCase` contract. This keeps one authority for every fact while
still exposing the six agreed Benchmark sections.

## Validation contract

- exactly `r17-qa-001` through `r17-qa-050` are present once in every source;
- difficulty counts remain 34 simple and 16 complex;
- every source hash matches the frozen manifest;
- every annotation is already reviewed and every Rubric is self-reviewed;
- Rubric items map one-to-one to reviewed Answer Units;
- every Rubric totals exactly 100;
- weights are positive multiples of five;
- the three bands have fixed ratios and non-empty guidance;
- every Gold Evidence excerpt matches its exact document hash and line range;
- every Gold Evidence document belongs to the frozen corpus environment;
- Workflow inputs contain no reference, Rubric, or Gold Evidence fields; and
- the environment contains the exact Release, version, path, and document hash
  of every frozen corpus document.

## Implementation sequence

1. Add strict Benchmark and Rubric models plus a composing loader.
2. Curate weights and partial-credit applicability for all 50 reviewed Answer
   Unit sets.
3. Freeze the Release 17 environment and source hashes in one manifest.
4. Add a standalone Benchmark checker and run it in `scripts/check.py`.
5. Add focused model, loader, leakage-boundary, and corruption tests.
6. Document the dataset and update repository status.
7. Audit every case, run the canonical quality gate, and correct every finding.

## Definition of done

- all 50 cases load into the six-section `BenchmarkCase` contract;
- no evaluator-only data can enter the Workflow input projection;
- every Rubric is weighted, banded, and self-reviewed;
- source and environment drift fail closed;
- documentation clearly states the final-answer-only and non-exclusive-evidence
  semantics; and
- `python scripts/check.py` passes.

## Completion record

- The composed loader returns all 50 cases with 129 weighted Rubric items and
  72 Gold Evidence passages.
- The exact 34-document Release 17 Environment and index fingerprint are
  frozen in the manifest. All three canonical source hashes and all corpus
  document hashes validate.
- Content self-review confirmed that every reviewed expected-fact set has a
  corresponding Answer Unit set, every Answer Unit appears once in its
  Rubric, every case totals 100, and all item weights use five-point steps.
- The leakage-boundary test confirms that only task ID, Question, and
  Environment appear in the Workflow projection.
- A single 50-case review file is checked in under `data/QA/`; its exporter and
  canonical stale-file check prevent it from drifting from the source data.
- `uv run python scripts/check.py` passes the structure check, original QA
  check, new Benchmark check, and all 342 tests.
