from datetime import datetime, timedelta, timezone

import pytest

from specrag.access import AuthorizationError
from specrag.infrastructure import InMemoryMemoryStore
from specrag.memory import (
    MemoryExtractionService,
    MemoryMaintenanceService,
    MemoryProposalService,
    MemoryRecallService,
    MemoryService,
    resolve_user_configuration,
)
from specrag.models import (
    ActorContext,
    MemoryExtractionSource,
    MemoryMutationDraft,
    ResearchRun,
)


NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


def _context(user="alice", workspace="WA", role="member"):
    return ActorContext(
        user_id=user,
        workspace_id=workspace,
        role=role,
        session_id="session",
        authenticated_at=NOW,
        expires_at=NOW + timedelta(hours=12),
    )


def _services():
    memory = MemoryService(InMemoryMemoryStore())
    proposals = MemoryProposalService(memory)
    return memory, proposals


def _run(*memory_ids: str) -> ResearchRun:
    return ResearchRun(
        run_id="RUN-" + "1" * 32,
        thread_id="THREAD-" + "1" * 32,
        workspace_id="WA",
        owner_user_id="alice",
        question="question",
        release=17,
        memory_ids=memory_ids,
        created_at=NOW,
    )


def test_user_core_is_private_and_project_core_is_workspace_shared() -> None:
    memory, _ = _services()
    private = memory.set(
        _context(), kind="preference", key="answer", value="Use concise Chinese", now=NOW
    )
    assert memory.list(_context()) == (private,)
    assert memory.list(_context("bob")) == ()
    with pytest.raises(AuthorizationError):
        memory.retract(_context("bob"), private.memory_id)

    reviewer = _context("reviewer", role="reviewer")
    project = memory.set_project_core(
        reviewer,
        kind="project_decision",
        key="judge-output",
        content="Judge returns evidence IDs only.",
        now=NOW,
    )
    assert memory.list_project_core(_context("bob")) == (project,)
    assert memory.list_project_core(_context(workspace="WB")) == ()
    with pytest.raises(AuthorizationError):
        memory.set_project_core(
            _context(), kind="project_decision", key="x", content="denied"
        )


def test_user_configuration_remains_typed_versioned_and_precedence_aware() -> None:
    memory, _ = _services()
    release = memory.set(
        _context(), kind="default_release", key="default", value="18", now=NOW
    )
    language = memory.set(
        _context(), kind="answer_language", key="default", value="Chinese", now=NOW
    )
    updated = memory.set(
        _context(),
        kind="answer_language",
        key="default",
        value="English",
        now=NOW + timedelta(minutes=1),
    )
    assert updated.revision == 2
    assert updated.supersedes_id == language.memory_id
    assert next(
        item for item in memory.store.all_entries() if item.memory_id == language.memory_id
    ).status == "superseded"
    resolved = resolve_user_configuration(
        memory.list(_context()), config_release=17, explicit_release=19
    )
    assert resolved.release == 19
    assert resolved.answer_language == "English"
    assert resolved.memory_ids == (updated.memory_id,)
    with pytest.raises(ValueError, match="integer"):
        memory.set(
            _context(), kind="default_release", key="default", value="R17", now=NOW
        )


def test_core_pinned_and_bm25_topic_recall_are_bounded() -> None:
    memory, _ = _services()
    reviewer = _context("reviewer", role="reviewer")
    user = memory.set(
        _context(), kind="research_habit", key="comparison", value="Compare baselines", now=NOW
    )
    project = memory.set_project_core(
        reviewer,
        kind="evaluation_convention",
        key="unit",
        content="Requirement is the evaluation unit.",
        now=NOW,
    )
    pinned = memory.create_topic(
        reviewer,
        slug="judge-design",
        description="Core Judge conventions",
        pinned=True,
        now=NOW,
    )
    pinned_entry = memory.add_topic_entry(
        reviewer,
        pinned.topic_id,
        kind="project_decision",
        key="selection",
        content="Judge selects evidence IDs.",
        now=NOW,
    )
    reranker = memory.create_topic(
        reviewer,
        slug="reranker-experiments",
        description="Reranker experiment design and failure lessons",
        now=NOW,
    )
    reranker_entry = memory.add_topic_entry(
        reviewer,
        reranker.topic_id,
        kind="experiment_lesson",
        key="baseline",
        content="Compare cross encoder reranker against hybrid retrieval.",
        now=NOW,
    )
    unrelated = memory.create_topic(
        reviewer,
        slug="agent-runtime",
        description="Checkpoint recovery and cancellation",
        now=NOW,
    )
    memory.add_topic_entry(
        reviewer,
        unrelated.topic_id,
        kind="project_decision",
        key="checkpoint",
        content="Persist checkpoints in SQLite.",
        now=NOW,
    )

    result = MemoryRecallService(memory).recall(
        _context(), "继续讨论 reranker 的实验设计"
    )
    assert {item.memory_id for item in result.entries} == {
        user.memory_id,
        project.memory_id,
        pinned_entry.memory_id,
        reranker_entry.memory_id,
    }
    assert result.pinned_topic_ids == (pinned.topic_id,)
    assert result.recalled_topic_ids == (reranker.topic_id,)

    bounded = MemoryRecallService(memory, entry_limit=2, character_limit=10000).recall(
        _context(), "reranker"
    )
    assert len(bounded.entries) == 2
    assert bounded.omitted_entry_count == 2


def test_run_snapshot_preserves_historical_user_and_project_revisions() -> None:
    memory, _ = _services()
    reviewer = _context("reviewer", role="reviewer")
    private = memory.set(
        _context(), kind="research_lead", key="lead", value="study X", now=NOW
    )
    shared = memory.set_project_core(
        reviewer,
        kind="project_decision",
        key="unit",
        content="Use Requirement units.",
        now=NOW,
    )
    memory.retract(_context(), private.memory_id, now=NOW + timedelta(minutes=1))
    admin = _context("admin", role="admin")
    assert memory.snapshot_for_run(admin, _run(private.memory_id, shared.memory_id)) == tuple(
        sorted((private.model_copy(update={"status": "retracted", "updated_at": NOW + timedelta(minutes=1)}), shared), key=lambda item: item.memory_id)
    )
    with pytest.raises(AuthorizationError):
        memory.snapshot_for_run(_context("bob"), _run(private.memory_id))


def test_extractor_creates_pending_proposals_and_owner_or_reviewer_approves() -> None:
    memory, proposals = _services()
    extraction = MemoryExtractionService(memory, proposals)

    class Extractor:
        def extract(self, source, **_):
            return (
                MemoryMutationDraft(
                    operation="add",
                    scope="user",
                    kind="preference",
                    key="judge-output",
                    content="Prefer Judge output to contain IDs only.",
                ),
                MemoryMutationDraft(
                    operation="add",
                    scope="workspace",
                    kind="project_decision",
                    key="evaluation-unit",
                    content="Use Requirement as the evaluation unit.",
                ),
            )

    records = extraction.propose_from_source(
        _context(),
        MemoryExtractionSource(
            source_ref="TURN-1",
            user_text="Remember these long-term decisions.",
        ),
        Extractor(),
    )
    assert all(item.status == "pending" for item in records)
    assert memory.list(_context()) == ()
    user_review = proposals.review(_context(), records[0].proposal_id, approve=True, now=NOW)
    assert user_review.status == "approved"
    with pytest.raises(AuthorizationError):
        proposals.review(_context(), records[1].proposal_id, approve=True, now=NOW)
    project_review = proposals.review(
        _context("reviewer", role="reviewer"),
        records[1].proposal_id,
        approve=True,
        now=NOW,
    )
    assert project_review.status == "approved"
    assert all(
        "TURN-1" in item.source_refs
        for item in (*memory.list(_context()), *memory.list_project_core(_context()))
    )


def test_consolidation_requires_both_thresholds_and_preserves_lineage() -> None:
    memory, proposals = _services()
    reviewer = _context("reviewer", role="reviewer")
    old = NOW - timedelta(hours=25)
    entries = tuple(
        memory.set_project_core(
            reviewer,
            kind="project_decision",
            key=f"decision-{index}",
            content=f"Decision text {index}",
            now=old + timedelta(minutes=index),
        )
        for index in range(10)
    )
    maintenance = MemoryMaintenanceService(memory, proposals)
    assert maintenance.eligibility(reviewer, scope="workspace", now=NOW).eligible

    class Consolidator:
        def consolidate(self, _entries):
            return (
                MemoryMutationDraft(
                    operation="merge",
                    scope="workspace",
                    kind="project_decision",
                    key="merged-decisions",
                    content="Decision text 0; Decision text 1",
                    target_memory_ids=(entries[0].memory_id, entries[1].memory_id),
                ),
            )

    proposal = maintenance.propose_consolidation(
        reviewer, scope="workspace", consolidator=Consolidator(), now=NOW
    )[0]
    reviewed = proposals.review(reviewer, proposal.proposal_id, approve=True, now=NOW)
    merged = next(
        item
        for item in memory.store.all_entries()
        if item.memory_id in reviewed.applied_memory_ids
    )
    assert merged.origin == "consolidation"
    assert set(merged.merged_from_ids) == {entries[0].memory_id, entries[1].memory_id}
    maintenance.mark_consolidated(reviewer, proposal.proposal_id, now=NOW)
    assert not maintenance.eligibility(
        reviewer, scope="workspace", now=NOW + timedelta(hours=25)
    ).eligible


def test_capacity_pruning_only_proposes_non_pinned_topic_archival() -> None:
    memory, proposals = _services()
    reviewer = _context("reviewer", role="reviewer")
    pinned = memory.create_topic(
        reviewer, slug="judge", description="Judge", pinned=True, now=NOW
    )
    normal = memory.create_topic(
        reviewer, slug="experiments", description="Experiments", now=NOW
    )
    protected = memory.add_topic_entry(
        reviewer,
        pinned.topic_id,
        kind="project_decision",
        key="protected",
        content="Keep this",
        now=NOW,
    )
    candidates = tuple(
        memory.add_topic_entry(
            reviewer,
            normal.topic_id,
            kind="experiment_lesson",
            key=f"lesson-{index}",
            content=f"Lesson {index}",
            now=NOW + timedelta(minutes=index),
        )
        for index in range(3)
    )
    proposal = MemoryMaintenanceService(
        memory, proposals, topic_capacity=1
    ).propose_pruning(reviewer, now=NOW + timedelta(hours=1))[0]
    assert proposal.draft.operation == "archive"
    assert set(proposal.draft.target_memory_ids) == {
        candidates[0].memory_id,
        candidates[1].memory_id,
    }
    assert protected.memory_id not in proposal.draft.target_memory_ids
    proposals.review(reviewer, proposal.proposal_id, approve=True, now=NOW)
    assert {
        item.memory_id
        for item in memory.store.all_entries()
        if item.status == "archived"
    } == set(proposal.draft.target_memory_ids)

    restore = MemoryMaintenanceService(
        memory, proposals, topic_capacity=1
    ).propose_restore(reviewer, candidates[0].memory_id, now=NOW)
    proposals.review(reviewer, restore.proposal_id, approve=True, now=NOW)
    restored = next(
        item
        for item in memory.store.all_entries()
        if item.memory_id == candidates[0].memory_id
    )
    assert restored.status == "active"


def test_memory_context_is_always_marked_untrusted() -> None:
    memory, _ = _services()
    item = memory.set(
        _context(), kind="research_lead", key="lead", value="study X", now=NOW
    )
    context = memory.as_research_context((item,))[0]
    assert context["trust"] == "untrusted_context"
    assert "Evidence" in context["instruction"]
