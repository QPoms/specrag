# Final-answer Benchmark V1

The active internal Benchmark is `specrag-release17-final-answer` version
`1.0.0`. It contains 50 answerable Release 17 development cases: 34 `simple`
and 16 `complex`. It is an engineering regression Benchmark, not a hidden or
independently adjudicated academic test set.

## Scope

The later evaluator will execute the complete backend Research Workflow once
per case. Conversation routing and presentation are excluded. Scoring is
final-answer-only: intermediate planning, retrieval behavior, and citation
matching are not quality dimensions in this Benchmark.

Each loaded case exposes exactly six sections:

1. Question;
2. Reference Answer;
3. positive, weighted Rubric;
4. known-sufficient Gold Evidence;
5. exact Release 17 Environment; and
6. Task Metadata.

Only Question, task ID, and Environment may enter the Workflow. Reference
Answer, Rubric, and Gold Evidence are evaluator-only data.

## Canonical files

- `data/QA/end_to_end_benchmark_review.json` is the materialized, single-file
  human-review view. It lists all 50 cases first. Every case keeps Task
  Metadata, Question, Reference Answer, Rubric, Gold Evidence, and its relevant
  specification documents and exact versions together. The complete frozen
  execution Environment appears once at the end of the file.
- `data/QA/end_to_end_benchmark.json` freezes Benchmark identity, source-file
  hashes, trial count, classification counts, retrieval index fingerprint, and
  all 34 exact corpus documents.
- `data/QA/end_to_end_rubrics.jsonl` contains self-reviewed weights for every
  reviewed Answer Unit.
- `data/QA/questions.jsonl` remains authoritative for questions, reference
  answers, source provenance, and Gold Evidence locators.
- `data/QA/component_evaluation_release17.jsonl` remains authoritative for the
  reviewed Answer Units used as Rubric criteria.

The loader in `specrag.evaluation.benchmark_loader` joins these sources into
strict `BenchmarkCase` values. The composed representation prevents duplicated
reference text and evidence from drifting across separate files.
`scripts/export_end_to_end_benchmark.py` deterministically materializes that
composition into the single review file. The Benchmark check rejects a stale
or manually divergent review file.

## Rubric semantics

Every case has a maximum score of 100. Simple cases normally have one to three
items; complex cases normally have three to four. Weights are positive
multiples of five and represent importance to the user's requested answer, not
retrieval difficulty or passage length.

Each item defines three bands:

- `full` (1.0): all required actors, conditions, actions, directions, values,
  and outcomes are semantically correct;
- `partial` (0.5): the central direction is correct, at least one explicitly
  required detail is missing, and no contradiction is introduced; and
- `none` (0.0): the point is absent, merely named, wrong, or contradicted.

Equivalent wording is accepted. Keyword presence is not sufficient. Errors
unrelated to a Rubric item do not introduce deductions in V1. The separate
Judge contract and remaining calibration work are tracked in
`docs/plans/045-final-answer-judge.md`.

## Gold Evidence semantics

Gold Evidence is reviewed, known-sufficient, and non-exclusive. It proves that
the Reference Answer and Rubric are grounded in the frozen corpus and supports
human curation and adjudication. It is deliberately excluded from the
final-answer Judge input. A candidate answer is not required to retrieve or
cite the same passage, and V1 does not score Gold Evidence overlap.

Every passage retains specification, exact version, clause, repository path,
document SHA-256, one-based line range, and verbatim excerpt. The loader checks
that each evidence document belongs to the exact frozen Environment.

## Environment

The Environment is the warm Release 17 snapshot used by the existing formal
baseline on 2026-10-18. It freezes 34 specification documents and retrieval
index fingerprint
`BD75704A2D409E0D71980E643DA6B91C20F10F6966C3D93904B57F707A93AF55`.
Model, Prompt, budget, retry, cache, and execution metadata belong to each
later Run manifest rather than to the Benchmark.

## Validation

Run the standalone check with:

```powershell
uv run python scripts/check_end_to_end_benchmark.py
```

Regenerate the single-file review view after an intentional canonical-source
change with:

```powershell
uv run python scripts/export_end_to_end_benchmark.py
```

The canonical `python scripts/check.py` gate also invokes it. Validation fails
on source drift, an incomplete or duplicate case set, an unreviewed component
annotation, a Rubric/Answer Unit mismatch, a weight total other than 100, a
nonstandard score band, corpus hash drift, Gold Evidence outside the frozen
Environment, or leakage in the tested Workflow input projection.

The Rubric weights have completed the requested Codex self-review. They remain
internal engineering annotations; later independent human/domain-expert
calibration should record new provenance and increment `benchmark_version`.
