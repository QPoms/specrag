from __future__ import annotations

from types import SimpleNamespace

import pytest

from specrag.infrastructure import (
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.models import CandidateEvidence, EvidenceSynthesis, ResearchTask
from specrag.prompts import SUFFICIENCY_INSTRUCTIONS
from specrag.prompts import SYNTHESIS_INSTRUCTIONS, synthesis_prompt


def _task() -> ResearchTask:
    return ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative registration trigger.",
    )


def _candidate(evidence_id: str = "E1") -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id=evidence_id,
        task_id="T1",
        chunk_id=f"C-{evidence_id}",
        specification="24.501",
        release=17,
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        line_start=1,
        line_end=1,
        excerpt="The UE shall initiate registration.",
    )


class _Agent:
    def __init__(self) -> None:
        self.payloads = []

    def parse(self, *, prompt_version, input_payload, output_type, **kwargs):
        self.payloads.append((prompt_version, input_payload))
        if prompt_version == "evidence-sufficiency/v1.13":
            value = {
                "task_id": "T1",
                "status": "sufficient",
                "supporting_evidence_ids": ["E1"],
                "rationale": "Complete support.",
            }
        else:
            value = {
                "answer": "The UE initiates registration. [E1]",
                "evidence_ids": ["E1"],
            }
        return SimpleNamespace(output=output_type.model_validate(value))


def test_structured_evidence_adapters_use_domain_output_contracts() -> None:
    agent = _Agent()
    task = _task()
    first = _candidate()
    sufficiency = StructuredTaskSufficiencyAssessor(agent).assess(
        task,
        (first,),
        "What starts registration?",
    )
    synthesis = StructuredEvidenceSynthesisGenerator(
        agent, answer_language="Chinese", detail_level="detailed"
    ).synthesize(
        task.question,
        (task,),
        (first,),
        mode="complete",
        covered_task_ids=("T1",),
        unresolved_task_ids=(),
        conflicting_task_ids=(),
    )

    assert sufficiency.status == "sufficient"
    assert synthesis.evidence_ids == ("E1",)
    assert [item[0] for item in agent.payloads] == [
        "evidence-sufficiency/v1.13",
        "evidence-synthesis/v1.10-full",
    ]
    sufficiency_payload = agent.payloads[0][1]
    assert sufficiency_payload["original_question"] == (
        "What starts registration?"
    )
    assert sufficiency_payload["task"] == {
        "task_id": "T1",
        "question": "What starts registration?",
    }
    synthesis_payload = agent.payloads[1][1]
    assert synthesis_payload["synthesis_mode"] == "complete"
    assert synthesis_payload["covered_task_ids"] == ("T1",)
    assert synthesis_payload["task_evidence"][0]["task_id"] == "T1"
    assert synthesis_payload["task_evidence"][0]["task_question"] == (
        "What starts registration?"
    )
    assert synthesis_payload["presentation"] == {
        "answer_language": "Chinese",
        "detail_level": "detailed",
    }
    assert "objective" not in synthesis_payload["task_evidence"][0]
    assert synthesis.answer == "The UE initiates registration. [E1]"


def test_synthesis_prompt_enforces_final_answer_boundary() -> None:
    full_version, full = synthesis_prompt("complete", has_conflicts=False)
    qualified_version, qualified = synthesis_prompt(
        "partial", has_conflicts=False
    )
    conflict_version, conflict = synthesis_prompt(
        "partial", has_conflicts=True
    )

    assert SYNTHESIS_INSTRUCTIONS == full
    assert full_version.endswith("-full")
    assert "original question is the authoritative answer scope" in full
    assert "coverage checkpoints only" in full
    assert "backend owns publication status" in full
    assert "Publication path: full synthesis" in full
    assert "qualified synthesis" not in full
    assert qualified_version.endswith("-qualified")
    assert "Publication path: qualified synthesis" in qualified
    assert "can appear in" in qualified
    assert "both `covered_task_ids` and `unresolved_task_ids`" in qualified
    assert "absent from `covered_task_ids`" in qualified
    assert "Cite evidence owned by every covered task" in qualified
    assert "turn missing evidence into a negative conclusion" in qualified
    assert "backend will append the mandatory unresolved-task disclosure" in qualified
    assert conflict_version.endswith("-qualified-conflict")
    assert "Do not reconcile" in conflict

    with pytest.raises(ValueError, match="cannot contain conflicts"):
        synthesis_prompt("complete", has_conflicts=True)


def test_synthesis_model_reads_legacy_claim_json_without_retaining_claims() -> None:
    synthesis = EvidenceSynthesis.model_validate({
        "answer": "The UE initiates registration.",
        "claims": [{
            "claim_id": "C1",
            "text": "The UE initiates registration.",
            "evidence_ids": ["E1"],
        }],
    })

    assert synthesis.answer == "The UE initiates registration."
    assert synthesis.evidence_ids == ("E1",)
    assert "claims" not in synthesis.model_dump()


def test_sufficiency_prompt_uses_question_as_authoritative_scope() -> None:
    assert "authoritative local scope" in SUFFICIENCY_INSTRUCTIONS
    assert "Identify Answer Units" in SUFFICIENCY_INSTRUCTIONS
    assert "Map Evidence" in SUFFICIENCY_INSTRUCTIONS
    assert "smallest non-redundant set" in SUFFICIENCY_INSTRUCTIONS
    assert "Minimality\n   must never override completeness" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "delegates an answer-bearing fact" in SUFFICIENCY_INSTRUCTIONS
    assert "every substantive answer unit" in SUFFICIENCY_INSTRUCTIONS
    assert "Do not introduce an unstated" in SUFFICIENCY_INSTRUCTIONS
    assert "normative relation" in SUFFICIENCY_INSTRUCTIONS
    assert "compositional evidence closure" in SUFFICIENCY_INSTRUCTIONS
    assert "explicit source-backed identity mapping" in SUFFICIENCY_INSTRUCTIONS
    assert "same actors, message or" in SUFFICIENCY_INSTRUCTIONS
    assert "matching target-routing passage" in SUFFICIENCY_INSTRUCTIONS
    assert "local\n   procedure step" in SUFFICIENCY_INSTRUCTIONS
    assert "Use partial when at least one requested answer unit" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "Do not promote background evidence" in SUFFICIENCY_INSTRUCTIONS
    assert "only the exact unsupported answer unit" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "Do not generate a task answer" in SUFFICIENCY_INSTRUCTIONS
