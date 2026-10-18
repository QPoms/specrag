from __future__ import annotations

import json
from math import log2
from types import SimpleNamespace

import pytest

from specrag.evaluation.component_evidence_metrics import (
    evaluate_final_evidence_artifact,
    evaluate_researcher_artifact,
    evaluate_retrieval_artifact,
    evaluate_synthesizer_artifact,
)
from specrag.evaluation.component_evidence_summary import summarize_evaluations
from specrag.evaluation.component_evidence_models import (
    FinalEvidenceJudgmentDraft,
    FinalEvidenceJudgment,
    ResearcherQuestionJudgmentDraft,
    ResearcherEvidenceJudgment,
    RetrievalCallJudgment,
    RetrievalJudgmentDraft,
    SynthesizerJudgmentDraft,
    SynthesizerJudgment,
)
from specrag.evaluation.component_models import ComponentQuestionAnnotation
from specrag.models import ResearchTask
from scripts.draft_component_retrieval import _build_judgment, _first_ranked_calls
from scripts.run_component_evidence_pipeline import (
    _gold_candidates,
    _oracle_synthesis_inputs,
)
from scripts.draft_component_evidence import _eligible_synthesis_evidence


def _annotation() -> ComponentQuestionAnnotation:
    return ComponentQuestionAnnotation.model_validate({
        "question_id": "Q1",
        "review_status": "proposed",
        "expected_route": "single_task",
        "route_rationale": "One objective.",
        "research_units": [{"unit_id": "RU1", "description": "Objective."}],
        "gold_tasks": [{
            "task_id": "T1",
            "question": "Question?",
            "objective": "Find the rule.",
            "research_unit_ids": ["RU1"],
        }],
        "evidence_units": [
            {
                "unit_id": "EU1",
                "proposition": "First proposition.",
                "research_unit_ids": ["RU1"],
            },
            {
                "unit_id": "EU2",
                "proposition": "Second proposition.",
                "research_unit_ids": ["RU1"],
            },
        ],
        "answer_units": [
            {
                "unit_id": "AU1",
                "statement": "First answer fact.",
                "evidence_unit_ids": ["EU1"],
            },
            {
                "unit_id": "AU2",
                "statement": "Second answer fact.",
                "evidence_unit_ids": ["EU2"],
            },
        ],
    })


def _evidence(evidence_id: str, start: int, *, task_id: str = "T1") -> dict:
    return {
        "evidence_id": evidence_id,
        "task_id": task_id,
        "chunk_id": f"CH-{start}",
        "source_path": "data/spec.md",
        "document_sha256": "abc",
        "line_start": start,
        "line_end": start + 4,
        "excerpt": "Source text.",
    }


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_evidence_judge_draft_schemas_are_provider_compatible() -> None:
    for model in (
        ResearcherQuestionJudgmentDraft,
        RetrievalJudgmentDraft,
        FinalEvidenceJudgmentDraft,
        SynthesizerJudgmentDraft,
    ):
        assert "prefixItems" not in json.dumps(model.model_json_schema())


def test_retrieval_draft_uses_exact_targets_and_all_ranked_results() -> None:
    call = {
        "call_id": "CALL1",
        "task_id": "T1",
        "tool_name": "hybrid_search",
        "observed_evidence": [_evidence("E1", 8), _evidence("E2", 30)],
    }
    assert _first_ranked_calls({"state": {"tool_calls": [call]}}) == {"T1": call}
    draft = RetrievalJudgmentDraft.model_validate({
        "target_locator_ids": ["ET1"],
        "candidate_grades": [
            {"candidate_id": "R1", "relevance": 2, "rationale": "Direct."},
            {"candidate_id": "R2", "relevance": 0, "rationale": "Unrelated."},
        ],
    })
    target = {
        "locator_id": "ET1",
        "source_path": "data/spec.md",
        "document_sha256": "abc",
        "line_start": 10,
        "line_end": 10,
    }
    results = [
        {"candidate_id": "R1", **_evidence("E1", 8)},
        {"candidate_id": "R2", **_evidence("E2", 30)},
    ]

    judgment = _build_judgment(
        question_id="Q1",
        call=call,
        scope={"release": 17},
        targets=[target],
        results=results,
        draft=draft,
        judge_model="test-model",
    )

    assert len(judgment.qrels) == 3
    assert judgment.qrels[0].target is True
    assert judgment.qrels[1].relevance == 2
    assert judgment.qrels[2].relevance == 0


def test_gold_sufficient_pool_replicates_targets_per_task() -> None:
    tasks = (
        ResearchTask(task_id="T1", question="First?", objective="First."),
        ResearchTask(task_id="T2", question="Second?", objective="Second."),
    )
    candidates = _gold_candidates("Q1", tasks, [{
        "specification": "23.501",
        "version": "17.15.0",
        "source_path": "data/3gpp/spec.md",
        "document_sha256": "A" * 64,
        "line_start": 10,
        "line_end": 11,
        "excerpt": "Normative target.",
    }])

    assert len(candidates) == 2
    assert {item.task_id for item in candidates} == {"T1", "T2"}
    assert all(item.chunk_id == "GOLD-Q1-1" for item in candidates)


def test_oracle_synthesis_inputs_are_judge_independent() -> None:
    tasks = (
        ResearchTask(task_id="T1", question="First?", objective="First."),
        ResearchTask(task_id="T2", question="Second?", objective="Second."),
    )
    gold = _gold_candidates("Q1", tasks, [{
        "specification": "23.501",
        "version": "17.15.0",
        "source_path": "data/3gpp/spec.md",
        "document_sha256": "A" * 64,
        "line_start": 10,
        "line_end": 11,
        "excerpt": "Normative target.",
    }])

    class AlwaysValid:
        def validate(self, _tasks, candidates):
            return SimpleNamespace(source_validations=tuple(
                SimpleNamespace(evidence_id=item.evidence_id, valid=True)
                for item in candidates
            ))

    validated, assessments = _oracle_synthesis_inputs(tasks, gold, AlwaysValid())

    assert {item.evidence_id for item in validated} == {
        "E:T1:GOLD-1", "E:T2:GOLD-1"
    }
    assert all(item.validation_status == "supported" for item in validated)
    assert all(item.status == "sufficient" for item in assessments)


def test_synthesis_judge_keeps_task_bound_duplicate_evidence_ids() -> None:
    artifact = {
        "candidate_evidence": [
            _evidence("E:T1:GOLD-1", 10, task_id="T1"),
            _evidence("E:T2:GOLD-1", 10, task_id="T2"),
        ],
        "validated_evidence": [
            {
                "evidence_id": evidence_id,
                "source_valid": True,
                "supports_target": True,
                "validation_status": "supported",
            }
            for evidence_id in ("E:T1:GOLD-1", "E:T2:GOLD-1")
        ],
    }

    eligible = _eligible_synthesis_evidence(artifact)

    assert [item["evidence_id"] for item in eligible] == [
        "E:T1:GOLD-1", "E:T2:GOLD-1"
    ]


def test_formal_researcher_artifact_requires_exact_gold_task(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "schema_version": "specrag.component-researcher-run/v1",
        "question": {"id": "Q1"},
        "state": {
            "research_tasks": [{
                "task_id": "T1",
                "question": "Different question?",
                "objective": "Find the rule.",
                "dependencies": [],
            }],
            "candidate_evidence": [_evidence("E1", 10)],
        },
    })
    judgment = ResearcherEvidenceJudgment.model_validate({
        "question_id": "Q1",
        "task_id": "T1",
        "review_status": "proposed",
        "unit_assessments": [
            {"unit_id": "EU1", "status": "covered", "evidence_ids": ["E1"], "rationale": "Direct."},
            {"unit_id": "EU2", "status": "missing", "rationale": "Absent."},
        ],
        "item_assessments": [{
            "evidence_id": "E1",
            "contribution": "direct_support",
            "evidence_unit_ids": ["EU1"],
            "rationale": "Direct.",
        }],
    })

    with pytest.raises(ValueError, match="differs from reviewed GoldResearchTask"):
        evaluate_researcher_artifact(artifact, _annotation(), judgment)


def test_formal_final_evidence_requires_gold_sufficient_contract(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "schema_version": "specrag.component-final-evidence-run/v1",
        "question": {"id": "Q1"},
        "candidate_pool_contract": {"gold_sufficient": False},
        "state": {"candidate_evidence": [], "validated_evidence": []},
    })
    judgment = FinalEvidenceJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "proposed",
        "unit_assessments": [
            {"unit_id": "EU1", "status": "missing", "rationale": "Absent."},
            {"unit_id": "EU2", "status": "missing", "rationale": "Absent."},
        ],
    })

    with pytest.raises(ValueError, match="gold-sufficient candidate pool"):
        evaluate_final_evidence_artifact(artifact, _annotation(), judgment)


def test_retrieval_metrics_use_rank_and_exact_provenance(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    first = _evidence("E1", 30)
    target = _evidence("E2", 10)
    _write(artifact, {
        "question": {"id": "Q1"},
        "retrieval_scope": {"release": 17},
        "state": {"tool_calls": [{
            "call_id": "call-1",
            "tool_name": "hybrid_search",
            "observed_evidence": [first, target],
        }]},
    })
    judgment = RetrievalCallJudgment.model_validate({
        "question_id": "Q1",
        "call_id": "call-1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "corpus_scope": {"release": 17},
        "qrels": [
            {
                "qrel_id": "partial",
                "locator": {
                    "source_path": "data/spec.md",
                    "document_sha256": "abc",
                    "line_start": 30,
                    "line_end": 34,
                },
                "relevance": 1,
                "rationale": "Necessary context.",
            },
            {
                "qrel_id": "target",
                "locator": {
                    "source_path": "DATA\\SPEC.MD",
                    "document_sha256": "ABC",
                    "line_start": 11,
                    "line_end": 11,
                },
                "relevance": 2,
                "target": True,
                "rationale": "Direct support.",
            },
            {
                "qrel_id": "target-result-grade",
                "locator": {
                    "source_path": "data/spec.md",
                    "document_sha256": "abc",
                    "line_start": 10,
                    "line_end": 14,
                },
                "relevance": 2,
                "rationale": "Returned result directly supports the target.",
            },
        ],
    })

    result = evaluate_retrieval_artifact(artifact, judgment)

    expected_ndcg = (1 + 3 / log2(3)) / (3 + 1 / log2(3))
    assert result.target_recall_at_5 == 1.0
    assert result.target_recall_at_12 == 1.0
    assert result.ndcg_at_5 == pytest.approx(expected_ndcg)
    assert result.ranked_relevances == (1, 2)
    assert result.unjudged_result_count == 0


def test_retrieval_metric_rejects_navigation_tools(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "question_id": "Q1",
        "state": {"tool_calls": [{
            "call_id": "call-1",
            "tool_name": "clause_lookup",
            "observed_evidence": [_evidence("E1", 10)],
        }]},
    })
    judgment = RetrievalCallJudgment.model_validate({
        "question_id": "Q1",
        "call_id": "call-1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "corpus_scope": {"release": 17},
        "qrels": [{
            "qrel_id": "target",
            "locator": {
                "source_path": "data/spec.md",
                "document_sha256": "abc",
                "line_start": 10,
                "line_end": 14,
            },
            "relevance": 2,
            "target": True,
            "rationale": "Direct support.",
        }],
    })

    with pytest.raises(ValueError, match="not a ranked-search tool"):
        evaluate_retrieval_artifact(artifact, judgment)


def test_retrieval_metric_rejects_corpus_scope_drift(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "question_id": "Q1",
        "retrieval_scope": {"release": 17},
        "state": {"tool_calls": [{
            "call_id": "call-1",
            "tool_name": "hybrid_search",
            "observed_evidence": [_evidence("E1", 10)],
        }]},
    })
    judgment = RetrievalCallJudgment.model_validate({
        "question_id": "Q1",
        "call_id": "call-1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "corpus_scope": {"release": 18},
        "qrels": [{
            "qrel_id": "target",
            "locator": {
                "source_path": "data/spec.md",
                "document_sha256": "abc",
                "line_start": 10,
                "line_end": 14,
            },
            "relevance": 2,
            "target": True,
            "rationale": "Direct support.",
        }],
    })

    with pytest.raises(ValueError, match="corpus scope"):
        evaluate_retrieval_artifact(artifact, judgment)


def test_researcher_metrics_score_coverage_and_precision(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "question_id": "Q1",
        "state": {"candidate_evidence": [
            _evidence("E1", 10),
            _evidence("E2", 20),
            _evidence("E-other", 40, task_id="T2"),
        ]},
    })
    judgment = ResearcherEvidenceJudgment.model_validate({
        "question_id": "Q1",
        "task_id": "T1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "unit_assessments": [
            {
                "unit_id": "EU1",
                "status": "covered",
                "evidence_ids": ["E1"],
                "rationale": "Directly established.",
            },
            {
                "unit_id": "EU2",
                "status": "partial",
                "evidence_ids": ["E1"],
                "rationale": "Condition is absent.",
            },
        ],
        "item_assessments": [
            {
                "evidence_id": "E1",
                "contribution": "direct_support",
                "evidence_unit_ids": ["EU1"],
                "rationale": "Supports EU1.",
            },
            {
                "evidence_id": "E2",
                "contribution": "background",
                "rationale": "Context only.",
            },
        ],
    })

    result = evaluate_researcher_artifact(artifact, _annotation(), judgment)

    assert result.evidence_coverage == 0.5
    assert result.full_evidence_coverage is False
    assert result.evidence_precision == 0.5


def test_final_evidence_deduplicates_cross_task_provenance(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    duplicate = _evidence("E3", 10, task_id="T2")
    duplicate["chunk_id"] = "CH-other-id"
    _write(artifact, {
        "question_id": "Q1",
        "state": {
            "candidate_evidence": [
                _evidence("E1", 10),
                duplicate,
                _evidence("E2", 20),
            ],
            "validated_evidence": [
                {
                    "evidence_id": evidence_id,
                    "source_valid": True,
                    "supports_target": True,
                    "validation_status": "supported",
                }
                for evidence_id in ("E1", "E3", "E2")
            ],
        },
    })
    judgment = FinalEvidenceJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "unit_assessments": [
            {
                "unit_id": "EU1",
                "status": "covered",
                "evidence_ids": ["E1"],
                "rationale": "Covered.",
            },
            {
                "unit_id": "EU2",
                "status": "covered",
                "evidence_ids": ["E2"],
                "rationale": "Covered.",
            },
        ],
        "item_assessments": [
            {
                "evidence_id": "E1",
                "contribution": "direct_support",
                "evidence_unit_ids": ["EU1"],
                "rationale": "Supports EU1.",
            },
            {
                "evidence_id": "E2",
                "contribution": "duplicate",
                "rationale": "Removable support repetition.",
            },
        ],
        "redundant_evidence_ids": ["E2"],
    })

    result = evaluate_final_evidence_artifact(artifact, _annotation(), judgment)

    assert result.deduplicated_evidence_ids == ("E1", "E2")
    assert result.final_evidence_coverage == 1.0
    assert result.final_evidence_precision == 0.5
    assert result.evidence_overlap_rate == 0.5


def test_final_evidence_missing_selection_scores_zero(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "question_id": "Q1",
        "state": {
            "candidate_evidence": [_evidence("E1", 10)],
            "validated_evidence": [],
        },
    })
    judgment = FinalEvidenceJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "unit_assessments": [
            {"unit_id": "EU1", "status": "missing", "rationale": "Not selected."},
            {"unit_id": "EU2", "status": "missing", "rationale": "Not selected."},
        ],
    })

    result = evaluate_final_evidence_artifact(artifact, _annotation(), judgment)

    assert result.final_evidence_count == 0
    assert result.final_evidence_coverage == 0.0
    assert result.final_evidence_precision == 0.0
    assert result.evidence_overlap_rate == 0.0


def test_synthesizer_metrics_weight_partial_coverage(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "question_id": "Q1",
        "synthesis": {
            "answer": "A supported claim and an incomplete claim.",
            "claims": [{"claim_id": "C", "evidence_ids": ["E1", "E2"]}],
        },
    })
    judgment = SynthesizerJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "answer_unit_assessments": [
            {"unit_id": "AU1", "status": "covered", "rationale": "Complete."},
            {"unit_id": "AU2", "status": "partial", "rationale": "Condition absent."},
        ],
        "claim_assessments": [
            {
                "claim_id": "AC1",
                "text": "Supported claim.",
                "support_status": "supported",
                "citation_status": "correct",
                "cited_evidence_ids": ["E1"],
                "rationale": "Citation supports it.",
            },
            {
                "claim_id": "AC2",
                "text": "Partially supported claim.",
                "support_status": "partial",
                "citation_status": "incomplete",
                "cited_evidence_ids": ["E2"],
                "rationale": "A required passage is absent.",
            },
        ],
    })

    result = evaluate_synthesizer_artifact(artifact, _annotation(), judgment)

    assert result.answer_coverage == 0.75
    assert result.full_answer_coverage is False
    assert result.answer_groundedness == 0.5
    assert result.citation_correctness == 0.5
    summary = summarize_evaluations((result,), "synthesizer")
    assert summary["calibration_only"] is True
    assert summary["answer_coverage"] == 0.75


def test_formal_synthesizer_requires_isolated_input_contract(tmp_path) -> None:
    artifact = tmp_path / "Q1.json"
    _write(artifact, {
        "schema_version": "specrag.component-synthesis-run/v1",
        "question_id": "Q1",
        "input_contract": {"judge_independent": False, "gold_sufficient": True},
        "synthesis": {
            "answer": "Supported answer [E1]",
            "evidence_ids": ["E1"],
        },
    })
    judgment = SynthesizerJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "proposed",
        "answer_unit_assessments": [
            {"unit_id": "AU1", "status": "covered", "rationale": "Covered."},
            {"unit_id": "AU2", "status": "covered", "rationale": "Covered."},
        ],
        "claim_assessments": [{
            "claim_id": "AC1",
            "text": "Supported answer.",
            "support_status": "supported",
            "citation_status": "correct",
            "cited_evidence_ids": ["E1"],
            "rationale": "Supported.",
        }],
    })

    with pytest.raises(ValueError, match="judge-independent"):
        evaluate_synthesizer_artifact(artifact, _annotation(), judgment)
