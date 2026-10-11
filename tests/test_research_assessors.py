from __future__ import annotations

from types import SimpleNamespace

from specrag.infrastructure import (
    StructuredEvidenceConflictAssessor,
    StructuredEvidenceSupportAssessor,
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.models import CandidateEvidence, ResearchTask, ValidatedEvidence


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
        if prompt_version == "evidence-support/v1.0":
            value = {
                "evidence_id": "E1",
                "status": "supported",
                "rationale": "Direct support.",
            }
        elif prompt_version == "evidence-sufficiency/v1.0":
            value = {
                "task_id": "T1",
                "status": "sufficient",
                "supporting_evidence_ids": ["E1"],
                "rationale": "Complete support.",
            }
        elif prompt_version == "evidence-conflict/v1.0":
            value = {"conflicts": [{
                "conflict_id": "X1",
                "task_id": "T1",
                "evidence_ids": ["E1", "E2"],
                "kind": "semantic_conflict",
                "description": "The passages disagree.",
            }]}
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
    second = _candidate("E2")
    validated = ValidatedEvidence(
        evidence_id="E1",
        task_id="T1",
        source_valid=True,
        supports_target=True,
        validation_status="supported",
        rationale="Direct support.",
    )

    support = StructuredEvidenceSupportAssessor(agent).assess(task, first)
    sufficiency = StructuredTaskSufficiencyAssessor(agent).assess(
        task,
        (first,),
        (validated,),
    )
    conflicts = StructuredEvidenceConflictAssessor(agent).detect(
        task,
        (first, second),
    )
    synthesis = StructuredEvidenceSynthesisGenerator(agent).synthesize(
        task.question,
        (task,),
        (first,),
    )

    assert support.status == "supported"
    assert sufficiency.status == "sufficient"
    assert conflicts[0].evidence_ids == ("E1", "E2")
    assert synthesis.claims[0].evidence_ids == ("E1",)
    assert [item[0] for item in agent.payloads] == [
        "evidence-support/v1.0",
        "evidence-sufficiency/v1.0",
        "evidence-conflict/v1.0",
        "evidence-synthesis/v1.0",
    ]
