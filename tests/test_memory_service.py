from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from specrag.infrastructure import InMemoryKnowledgeStore, JsonlKnowledgeStore
from specrag.memory import ValidatedMemoryService
from specrag.orchestration import MemoryWriteHandler, new_research_state
from specrag.models import (
    CandidateEvidence,
    EvidenceBackedClaim,
    EvidenceSynthesis,
    KnowledgeProposal,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


def _candidate(version: str = "17.1.0", evidence_id: str = "E1") -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id=evidence_id,
        task_id="T1",
        chunk_id=f"C-{version}",
        specification="24.501",
        release=int(version.split(".", 1)[0]),
        version=version,
        source_path=f"data/3gpp/24.501/v{version}/source.md",
        document_sha256="A" * 64,
        line_start=2,
        line_end=2,
        excerpt="The UE shall initiate registration.",
    )


def _validated(evidence_id: str = "E1") -> ValidatedEvidence:
    return ValidatedEvidence(
        evidence_id=evidence_id,
        task_id="T1",
        source_valid=True,
        supports_target=True,
        validation_status="supported",
        rationale="Validated source support.",
    )


def _synthesis() -> EvidenceSynthesis:
    return EvidenceSynthesis(
        answer="The UE initiates registration.",
        claims=(EvidenceBackedClaim(
            claim_id="C1",
            text="The UE initiates registration.",
            evidence_ids=("E1",),
        ),),
    )


def _task() -> ResearchTask:
    return ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative registration trigger.",
    )


def _sufficient() -> tuple[TaskSufficiencyAssessment, ...]:
    return (TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The complete task is supported.",
    ),)


class _Distiller:
    def __init__(
        self,
        *,
        object_value: str = "initiates",
        expires_at: datetime | None = None,
    ) -> None:
        self.object_value = object_value
        self.expires_at = expires_at

    def propose(self, synthesis, evidence):
        return (KnowledgeProposal(
            claim_id="C1",
            subject="UE",
            predicate="initiates",
            object=self.object_value,
            conditions=("registration procedure",),
            expires_at=self.expires_at,
        ),)


def _store_record(
    tmp_path: Path,
    *,
    version: str = "17.1.0",
    object_value: str = "initiates",
):
    service = ValidatedMemoryService(InMemoryKnowledgeStore())
    service.distill_and_store(
        synthesis=_synthesis(),
        tasks=(_task(),),
        candidates=(_candidate(version),),
        validated=(_validated(),),
        sufficiency=_sufficient(),
        distiller=_Distiller(object_value=object_value),
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return service


def test_memory_write_requires_sufficient_validated_evidence() -> None:
    service = ValidatedMemoryService(InMemoryKnowledgeStore())
    partial = _sufficient()[0].model_copy(update={
        "status": "partial",
        "missing_relation": "missing relation",
    })

    with pytest.raises(ValueError, match="sufficient evidence"):
        service.distill_and_store(
            synthesis=_synthesis(),
            tasks=(_task(),),
            candidates=(_candidate(),),
            validated=(_validated(),),
            sufficiency=(partial,),
            distiller=_Distiller(),
        )


def test_memory_write_rejects_claim_with_unvalidated_source() -> None:
    service = ValidatedMemoryService(InMemoryKnowledgeStore())
    rejected = _validated().model_copy(update={
        "source_valid": False,
        "validation_status": "rejected",
    })

    with pytest.raises(ValueError, match="not fully validated"):
        service.distill_and_store(
            synthesis=_synthesis(),
            tasks=(_task(),),
            candidates=(_candidate(),),
            validated=(rejected,),
            sufficiency=_sufficient(),
            distiller=_Distiller(),
        )


def test_memory_search_returns_only_active_validated_knowledge() -> None:
    service = _store_record(Path("."))

    results = service.search("UE initiates registration")

    assert len(results) == 1
    assert results[0].knowledge.validation_status == "validated"
    assert results[0].knowledge.memory_status == "active"
    assert results[0].knowledge.sources[0].version == "17.1.0"


def test_new_version_supersedes_old_semantically_same_knowledge() -> None:
    store = InMemoryKnowledgeStore()
    service = ValidatedMemoryService(store)
    service.distill_and_store(
        synthesis=_synthesis(), tasks=(_task(),), candidates=(_candidate(),),
        validated=(_validated(),), sufficiency=_sufficient(),
        distiller=_Distiller(), now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    service.distill_and_store(
        synthesis=_synthesis(), tasks=(_task(),),
        candidates=(_candidate("18.1.0"),), validated=(_validated(),),
        sufficiency=_sufficient(), distiller=_Distiller(),
        now=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    records = store.all()
    assert {item.memory_status for item in records} == {"superseded", "active"}
    assert len(service.search("UE initiates")) == 1
    assert service.search("UE initiates")[0].knowledge.sources[0].version == "18.1.0"


def test_conflicting_object_marks_knowledge_conflicted_and_hides_it() -> None:
    store = InMemoryKnowledgeStore()
    service = ValidatedMemoryService(store)
    service.distill_and_store(
        synthesis=_synthesis(), tasks=(_task(),), candidates=(_candidate(),),
        validated=(_validated(),), sufficiency=_sufficient(),
        distiller=_Distiller(object_value="initiates"),
    )
    service.distill_and_store(
        synthesis=_synthesis(), tasks=(_task(),), candidates=(_candidate(),),
        validated=(_validated(),), sufficiency=_sufficient(),
        distiller=_Distiller(object_value="does-not-initiate"),
    )

    assert all(item.memory_status == "conflicted" for item in store.all())
    assert service.search("UE initiates") == ()


def test_invalidation_is_persisted_in_jsonl_store(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "knowledge.jsonl"
    store = JsonlKnowledgeStore(path)
    service = ValidatedMemoryService(store)
    records = service.distill_and_store(
        synthesis=_synthesis(), tasks=(_task(),), candidates=(_candidate(),),
        validated=(_validated(),), sufficiency=_sufficient(),
        distiller=_Distiller(),
    )

    invalidated = service.invalidate(records[0].knowledge_id, "superseded by review")
    reloaded = JsonlKnowledgeStore(path)

    assert invalidated.memory_status == "invalidated"
    assert reloaded.all()[0].invalidation_reason == "superseded by review"
    assert service.search("UE initiates") == ()


def test_memory_write_handler_returns_ids_without_copying_records_into_state() -> None:
    store = InMemoryKnowledgeStore()
    service = ValidatedMemoryService(store)
    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (_task().model_copy(update={"status": "completed"}),)
    state["candidate_evidence"] = (_candidate(),)
    state["validated_evidence"] = (_validated(),)
    state["sufficiency_assessments"] = _sufficient()
    state["synthesis"] = _synthesis()

    update = MemoryWriteHandler(service, _Distiller())(state)

    assert update["memory_knowledge_ids"]
    assert "memory_records" not in update
    assert len(store.all()) == 1


def test_expired_memory_is_invalidated_and_removed_from_search() -> None:
    store = InMemoryKnowledgeStore()
    service = ValidatedMemoryService(store)
    service.distill_and_store(
        synthesis=_synthesis(),
        tasks=(_task(),),
        candidates=(_candidate(),),
        validated=(_validated(),),
        sufficiency=_sufficient(),
        distiller=_Distiller(
            expires_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    expired = service.expire(now=datetime(2026, 1, 3, tzinfo=timezone.utc))

    assert len(expired) == 1
    assert expired[0].memory_status == "invalidated"
    assert service.search("UE initiates") == ()
