# End-to-end and efficiency baseline

Baseline date: 2026-10-18

Scope: 50 answerable Release 17 development questions

Quality metric: binary Task Success Rate

## Result

The formal baseline completed one fresh, sequential, warm-cache workflow Run
for every question. All 50 decisions were then proposed by the independent
LLM Judge and manually audited against the reviewed Answer Units, generated
answer, and attached evidence.

| Metric | Formal result |
| --- | ---: |
| Task Success | 38 / 50 |
| Task Success Rate | **76.0%** |
| Simple questions | 29 / 34 (85.3%) |
| Complex questions | 9 / 16 (56.2%) |
| Normal final answers | 49 / 50 |
| Runtime/budget failure | 1 / 50 |
| Manually reviewed decisions | 50 / 50 |

The LLM Judge initially proposed 43 successes. Manual review changed five
over-permissive success proposals (`003`, `006`, `023`, `042`, and `048`) to
failures because each omitted at least one reviewed core Answer Unit. It did
not reverse any proposed failure. The reviewed 38/50 result, rather than the
unreviewed 43/50 proposal, is the baseline.

## Success contract

A question succeeds only when the complete workflow:

1. produces a normal final answer;
2. answers the question's core conclusions;
3. contains no material error or unsupported core conclusion;
4. correctly cites evidence for its core conclusions; and
5. stays within the requested Release and frozen corpus scope.

Minor wording or stylistic defects do not fail a Run. A missing reviewed core
Answer Unit does. Runtime and budget-exhausted Runs remain in the denominator.

## Efficiency

| Metric | Formal result |
| --- | ---: |
| Per-question latency, mean | 79.183 s |
| Per-question latency, P95 | 191.969 s |
| Sum of per-question latency | 3,959.125 s (65.99 min) |
| LLM calls | 450 total; 9.00 mean; 19 P95 |
| Tool-call attempts | 289 total; 5.78 mean; 15 P95 |
| Tool outcomes | 271 completed; 7 no hits; 11 rejected |
| Complete Token telemetry | 49 / 50 Runs |
| Tokens for 49 complete Runs | 2,652,796 total; 54,138.7 mean; 191,787 P95 |
| Whole-batch exact Token total | unavailable |
| Observed whole-batch Token lower bound | 2,761,525 |

Tool calls count all recorded attempts, including rejected calls and calls that
returned no hits, rather than only successful retrieval executions.

The exact 50-Run Token total is deliberately not reported. In `r17-qa-046`,
one Sufficiency Judge logical call required five API attempts. The provider
reported 7,647 Tokens for the final successful response but did not report
usage for the four failed attempts. The Run therefore retains all call and
attempt metadata while marking aggregate Token telemetry incomplete. The
2,761,525 figure is only a lower bound: it combines the exact 49-Run total with
108,729 Tokens observed from the successful responses in the incomplete Run.
It must not be presented as the exact batch cost.

The same retry produced the 927.364-second maximum latency for `046`. P95 is
lower because it is not determined by this single extreme outlier.

## Failed cases

| Question | Reviewed failure | Notes |
| --- | --- | --- |
| `r17-qa-003` | Missing core conclusion | Did not explicitly release the SRB PDCP entity, RLC entity, and logical channel. |
| `r17-qa-006` | Missing core conclusion | Gave NIA0/NEA0 but omitted independence from announced algorithms and NAS Security Mode Command signalling. |
| `r17-qa-023` | Missing core conclusion | Explained KSI and `111` but omitted that KSI is three bits. |
| `r17-qa-035` | Missing core conclusion | Omitted the KgNB/previous-NH chaining and KAMF KDF input needed to explain freshness. |
| `r17-qa-039` | Runtime failure | Exhausted the iteration budget without a normal final answer. |
| `r17-qa-041` | Missing core conclusion | Omitted Registration Request rerouting from the initial AMF to the target AMF. |
| `r17-qa-042` | Missing core conclusion | Omitted ongoing-service, scheduling, and neighbour-cell information carried by MCCH. |
| `r17-qa-044` | Missing core conclusion | Omitted the cause, handover-trigger delta, response-before-apply behavior, and OAM limits. |
| `r17-qa-045` | Missing conclusion, material error, unsupported claim, incorrect citation | Replaced valid-context verification with DU admission and omitted the RRCSetup fallback. |
| `r17-qa-046` | Missing conclusion, incorrect citation | Omitted intra-PLMN TLS, DTLS, and RSA/ECDSA requirements; one normative certificate claim had a misattached citation. |
| `r17-qa-047` | Missing core conclusion | Omitted Control Plane Service Request state/actions and MME verification/decryption/decompression. |
| `r17-qa-048` | Missing core conclusion | Omitted Network Triggered Service Request when the UE is ECM-IDLE. |

The reason counts overlap because one question may fail more than one gate:
11 missing-core-conclusion flags, one runtime failure, one material error, one
material unsupported claim, and two incorrect-core-citation flags.

## Reproducibility record

- Question file SHA-256:
  `f2306f91066d64fbdf0f5e1d3c99c572a6c3027fd966cd985d23d24186ad933a`.
- Source revision recorded at execution:
  `66ea1c1b6567f39b2f4695c825550270143d4b75` with a dirty working tree.
- Workflow/Judge model: `gpt-5.6-sol`; every persisted call also records its
  actual model, prompt version, latency, attempts, outcome, and Token usage.
- Budget: three iterations, six regular and six gap-retrieval calls per task,
  and two replans.
- Retrieval scope: Release 17, highest version within the Release, 34 exact
  documents, index fingerprint
  `BD75704A2D409E0D71980E643DA6B91C20F10F6966C3D93904B57F707A93AF55`.
- Cache/execution condition: warm cache; questions executed sequentially;
  normal parallelism inside one workflow Run remained enabled.
- Local Run artifacts:
  `artifacts/runs/end-to-end/20261018-formal-50/`.
- Local reviewed evaluations:
  `artifacts/evaluations/end-to-end/20261018-formal-50/`.

The local artifact directories are intentionally ignored by Git because they
contain generated Runs and reports. The evaluator implementation, reviewed
metric contract, commands, and this concise result record are versioned.

## Commands

```powershell
uv run specrag baseline --all `
  --output-dir artifacts/runs/end-to-end/20261018-formal-50

uv run python scripts/evaluate_end_to_end.py draft `
  artifacts/runs/end-to-end/20261018-formal-50 `
  data/QA/questions.jsonl `
  data/QA/component_evaluation_release17.jsonl `
  artifacts/evaluations/end-to-end/20261018-formal-50

uv run python scripts/evaluate_end_to_end.py review `
  artifacts/evaluations/end-to-end/20261018-formal-50 `
  REVIEW_DECISIONS.jsonl

uv run python scripts/evaluate_end_to_end.py summarize `
  artifacts/evaluations/end-to-end/20261018-formal-50
```

The ten-question pilot that preceded the formal baseline scored 7/10 (70.0%)
with complete Token telemetry. Its mean/P95 latency was 68.631/147.932 seconds,
and it consumed 556,860 Tokens across 102 LLM calls and 65 tool-call attempts.
