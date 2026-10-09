# Specrag optimization roadmap

Updated: 2026-10-09

## Overall objective

Build Specrag into a clean, testable, internship-ready 3GPP Technical Research
Agent with version-aware research, auditable evidence, grounded answers, and
validated long-term memory.

## Progress summary

| Milestone | Status | Current evidence | Next completion gate |
| --- | --- | --- | --- |
| 1. Engineering baseline | Complete | Clean package/data layout, structural guardrails, canonical quality gate, 5 passing tests, Git baseline | Maintain the gate while starting characterization tests |
| 2. Core refactor | Not started | Mechanical package move completed; large historical modules remain | Characterization tests, then split without parallel versions |
| 3. Version-aware retrieval | Partial foundation | 146 versions retained; latest and Release-specific policies work | Exact, mixed-version, and cross-version query contracts |
| 4. Retrieval tool system | Not started | Retrieval implementations exist but have no stable tool registry | Unified tool protocol and independently tested tools |
| 5. Research Agent | Partial foundation | A bounded researcher loop was retained | Explicit state, budgets, stop reasons, router, replanning |
| 6. Evidence system | Not started | Historical evidence code was deliberately removed | Clean provenance, validation, sufficiency, gaps, synthesis |
| 7. Long-term Memory | Not started | Data boundary reserved at `data/memory/` | Validated-only write path, search, update, conflict and expiry |
| 8. Evaluation and demo | Not started | Structure checker and five tests only | Metrics, CLI demo, architecture diagram, examples, polished README |

## 1. Engineering baseline

- [x] Unify the repository and Python package name as `specrag`.
- [x] Reduce `data/` to the 3GPP corpus and future memory data.
- [x] Preserve all 146 local historical specification versions.
- [x] Add `AGENTS.md`, package boundaries, and automated structure checks.
- [x] Keep generated indexes outside project data under `.local/`.
- [x] Create the first recoverable Git baseline commit.
- [x] Define and run one canonical quality command for structure and tests.

## 2. Core refactor

- [ ] Add characterization tests for the active retrieval behavior.
- [ ] Split `retrieval/planner.py` by models, planning, execution, selection,
  and answer summarization.
- [ ] Split `retrieval/hypothesis.py` into proposal models and execution.
- [ ] Split `retrieval/hybrid.py` into corpus, index, lexical, semantic, and
  fusion responsibilities.
- [ ] Split `agents/researcher.py` into state, policy, and loop execution.
- [ ] Remove all temporary line-count exceptions from the structure checker.

## 3. Version-aware retrieval

- [x] Retain every historical version in a deterministic corpus layout.
- [x] Support latest-version-per-specification retrieval.
- [x] Support the highest available version within a selected Release.
- [ ] Support an exact specification version.
- [ ] Support different versions for different specifications in one task.
- [ ] Support cross-version comparison and change-oriented evidence.
- [ ] Include specification and version identity in every citation.

## 4. Retrieval tool system

- [ ] Define one stable agent-tool request/result contract.
- [ ] Add a tool registry without provider-specific logic.
- [ ] Expose keyword/BM25 retrieval as one tool.
- [ ] Expose semantic retrieval as one tool.
- [ ] Expose hybrid fusion and reranking as composable stages.
- [ ] Expose clause lookup and adjacent-section expansion.
- [ ] Test every tool independently from the agent loop.

## 5. Research Agent

- [ ] Define explicit research task, plan, state, action, and observation models.
- [ ] Add bounded turn, tool-call, and retry budgets.
- [ ] Record explicit finish and stop reasons.
- [ ] Implement Router/Planner boundaries.
- [ ] Implement Think → Tool → Observe → Re-plan.
- [ ] Keep single-Researcher execution stable before adding multi-Researcher
  supervision.

## 6. Evidence system

- [ ] Define candidate and validated evidence models.
- [ ] Verify source path, document hash, clause locator, and exact excerpt.
- [ ] Deduplicate overlapping evidence.
- [ ] Detect version and semantic conflicts.
- [ ] Evaluate evidence sufficiency against the research question.
- [ ] Produce explicit gap analysis and bounded follow-up retrieval.
- [ ] Generate answers that map claims to evidence.

## 7. Long-term Memory

- [ ] Define the knowledge and provenance schema.
- [ ] Permit writes only from validated evidence and grounded synthesis.
- [ ] Preserve specification version, source hash, and validation time.
- [ ] Implement search and reuse in later research tasks.
- [ ] Handle updates, conflicts, supersession, and expiry explicitly.
- [ ] Prevent raw model output from entering memory.

## 8. Evaluation and demo

- [ ] Define retrieval and evidence-set metrics.
- [ ] Measure citation correctness and answer grounding.
- [ ] Evaluate agent trajectories, budgets, and stop reasons.
- [ ] Add small, reviewable 3GPP example questions and expected evidence.
- [ ] Provide one supported `specrag` CLI demo.
- [ ] Add an architecture diagram and reproducible demo instructions.
- [ ] Polish the README around problem, design, results, and limitations.

## Global completion constraints

- Do not reintroduce attack-chain, attacker-capability, B-static, CR, or hidden-
  Gold research pipelines.
- Do not introduce versioned Python modules or parallel implementations.
- Do not treat model-generated content as evidence.
- Do not write unvalidated content to long-term memory.
- Keep every retained 3GPP historical version available for retrieval.
- Completion requires passing structure checks, automated tests, a runnable demo,
  and traceable claim-to-evidence output.
