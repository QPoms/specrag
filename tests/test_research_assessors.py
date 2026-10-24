from __future__ import annotations

from types import SimpleNamespace

from specrag.infrastructure import (
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.models import CandidateEvidence, EvidenceSynthesis, ResearchTask
from specrag.prompts import SUFFICIENCY_INSTRUCTIONS
from specrag.prompts import SYNTHESIS_INSTRUCTIONS


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
        "evidence-synthesis/v1.8",
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
    assert "complete and authoritative answer scope" in SYNTHESIS_INSTRUCTIONS
    assert "Task\nquestions are coverage checkpoints only" in (
        SYNTHESIS_INSTRUCTIONS
    )
    assert "one complete, coherent answer" in SYNTHESIS_INSTRUCTIONS
    assert "supported\nevidence item owned by each covered task" in (
        SYNTHESIS_INSTRUCTIONS
    )
    assert "Do not return a\nclaim list" in SYNTHESIS_INSTRUCTIONS
    assert "different specifications or radio" in SYNTHESIS_INSTRUCTIONS
    assert "systems into one generalized statement" in SYNTHESIS_INSTRUCTIONS
    assert "nested conditions" in SYNTHESIS_INSTRUCTIONS
    assert "implication direction" in SYNTHESIS_INSTRUCTIONS


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
