from __future__ import annotations

from types import SimpleNamespace

from specrag.infrastructure import (
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.models import CandidateEvidence, ResearchTask
from specrag.prompts import SUFFICIENCY_INSTRUCTIONS


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
        if prompt_version == "evidence-sufficiency/v1.7":
            value = {
                "task_id": "T1",
                "status": "sufficient",
                "supporting_evidence_ids": ["E1"],
                "rationale": "Complete support.",
            }
        else:
            value = {
                "answer": "The UE initiates registration.",
                "claims": [{
                    "claim_id": "C1",
                    "text": "The UE initiates registration.",
                    "evidence_ids": ["E1"],
                }],
            }
        return SimpleNamespace(output=output_type.model_validate(value))


def test_structured_evidence_adapters_use_domain_output_contracts() -> None:
    agent = _Agent()
    task = _task()
    first = _candidate()
    sufficiency = StructuredTaskSufficiencyAssessor(agent).assess(
        task,
        (first,),
    )
    synthesis = StructuredEvidenceSynthesisGenerator(agent).synthesize(
        task.question,
        (task,),
        (first,),
    )

    assert sufficiency.status == "sufficient"
    assert synthesis.claims[0].evidence_ids == ("E1",)
    assert [item[0] for item in agent.payloads] == [
        "evidence-sufficiency/v1.7",
        "evidence-synthesis/v1.0",
    ]
    sufficiency_payload = agent.payloads[0][1]
    assert sufficiency_payload["task"] == {
        "task_id": "T1",
        "question": "What starts registration?",
    }


def test_sufficiency_prompt_uses_question_as_authoritative_scope() -> None:
    assert "collectively establish that question" in SUFFICIENCY_INSTRUCTIONS
    assert "complete and authoritative scope" in SUFFICIENCY_INSTRUCTIONS
    assert "Do not use planner objectives" in SUFFICIENCY_INSTRUCTIONS
    assert "may collectively establish a description, mapping" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "explicitly states the overall criterion" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "establishes that branch, not by itself the complete success" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "only when the task question requests it" in (
        SUFFICIENCY_INSTRUCTIONS
    )
    assert "question explicitly requests all" in SUFFICIENCY_INSTRUCTIONS
    assert "planner objectives" in SUFFICIENCY_INSTRUCTIONS
