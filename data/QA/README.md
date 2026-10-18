# Release 17 development questions

This directory contains the small, manually verified question set used to
improve Specrag's end-to-end research behavior. It is not a formal benchmark.

`questions.jsonl` is the curated development question set. The
`component_*_pilot*.jsonl` files are the reviewed component-evaluation
annotations and semantic judgments for the first internal baseline.
`component_evaluation_release17.jsonl` is the reviewed 50-question component
annotation set, including Router labels plus linked Research, task, Evidence,
and Answer Units. `component_planner_release17_judgments.jsonl` contains the
reviewed semantic mappings for all 18 decomposition cases. The reviewed full
evidence-path judgments are `component_retrieval_release17_judgments.jsonl`,
`component_researcher_release17_judgments.jsonl`,
`component_final_evidence_release17_judgments.jsonl`, and
`component_synthesizer_release17_judgments.jsonl`. The `*_pilot*.jsonl` files
retain the smaller reviewed calibration slice and evidence-path pilot.
Complete upstream snapshots may be downloaded into
`telco-dpr/` and `mm-telcobench/`; both directories are local inputs ignored by
Git.

A component record may be promoted to `reviewed` only when it also records a
non-empty `reviewed_by` value and a timezone-aware `reviewed_at` timestamp.
This prevents a bare status-string change from silently creating formal gold.
The first pilot review was performed by Codex under explicit project-owner
delegation and records that provenance in `reviewed_by`; it is an internal
engineering review, not independent domain-expert or academic adjudication.

Each accepted question must be verified against Specrag's own highest Release
17 specification version. External answers and passage identifiers are hints,
not trusted evidence. Diagnostic facts and evidence are withheld from the
Research Agent during a run.

The Router labels contain 34 `single_task/simple` and 16
`decompose/complex` questions. Evidence scope is tracked independently: 30
single-clause, 4 cross-clause, and 16 cross-specification cases. All 50 records
use the `development` split; a fresh external test set will be collected after
the complete workflow has been optimized.

## Record contract

Each line of `questions.jsonl` is one `specrag.qa/v2` record. The fields have
two deliberately separate roles:

- Agent inputs: only `id` and `question` cross the dataset-to-runtime boundary.
- Router diagnostics: `difficulty` is the reviewed routing label and is never
  supplied to Router. Evaluation derives `single_task` from `simple` and
  `decompose` from `complex`; the equivalent route is not stored twice.
- Evidence diagnostics: `reasoning_scope` records whether the bound evidence is
  single-clause, cross-clause, or cross-specification. It does not determine the
  Router label.
- Curation metadata: `source` records the upstream dataset, record kind and ID,
  license, original wording, and transformation applied by Specrag.
- Diagnostics: `expected_answer`, `expected_facts`, and `evidence_targets` are
  used only after a run to inspect its behavior.
- Split control: all current questions are development data. No internal
  holdout is reserved.

Every evidence target contains the specification number, exact R17 version,
clause, repository-relative source path, full-document SHA-256, one-based line
range, and the exact source excerpt. Run the dataset check with:

```text
uv run python scripts/check_qa_dataset.py
```

The check fails if a source is outside `data/3gpp/`, is not Release 17, has a
different hash, or no longer contains the exact excerpt at the recorded lines.
It validates evidence counts from `reasoning_scope`, independently from Router
difficulty, and requires every `cross_specification` question to bind at least
two distinct specifications.
The fixed set shape, sequential IDs, unique curated and upstream questions,
source dataset/license, and transformation labels are checked as well.

## Source decision

The complete Telco-DPR snapshot was inspected, but its 1,741 queries are tied
to TR 36.777, TR 38.811, TR 38.821, and TR 38.901. Those reports are not in the
current local R17 corpus, so using those questions now would violate the local
evidence-binding rule. The snapshot remains available locally for later use
after the corresponding source corpus is added.

The curated questions therefore come from MM-TelcoBench IR records. Their
upstream passages were used only to locate candidates; all expected answers and
evidence targets were re-verified against Specrag's highest local R17 versions.
All 50 questions use distinct upstream record IDs and distinct local question
text. An audit against the downloaded Parquet snapshot also verifies that every
stored `original_question` is the exact upstream wording and that the upstream
source specification appears in the local evidence targets.

Upstream sources:

- Telco-DPR RAG, Apache-2.0:
  <https://huggingface.co/datasets/DinoStackAI/telco-dpr-rag>
- MM-TelcoBench, CC-BY-4.0:
  <https://huggingface.co/datasets/Exploration-Lab/MM-TelecoBench>
