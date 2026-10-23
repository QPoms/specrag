from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from specrag.models import (
    Conversation,
    ConversationMessageContent,
    ConversationResearchLink,
    ConversationTurn,
    ConversationTurnBudget,
    require_conversation_status_transition,
    require_presentation_status_transition,
    require_tool_call_status_transition,
    require_turn_status_transition,
)


NOW = datetime(2026, 10, 22, tzinfo=UTC)
CONVERSATION_ID = "CONV-" + "A" * 32
TURN_ID = "TURN-" + "B" * 32
RUN_ID = "RUN-" + "C" * 32
RESULT_ID = "RESULT-" + "D" * 32
CALL_ID = "CALL-" + "E" * 32


def test_conversation_requires_coherent_message_watermarks() -> None:
    conversation = Conversation(
        conversation_id=CONVERSATION_ID,
        workspace_id="workspace-a",
        owner_user_id="alice",
        context_id="context-a",
        last_message_sequence=4,
        last_user_message_sequence=3,
        created_at=NOW,
        updated_at=NOW,
    )
    assert conversation.status == "active"

    with pytest.raises(ValidationError, match="user message sequence"):
        Conversation(
            conversation_id=CONVERSATION_ID,
            workspace_id="workspace-a",
            owner_user_id="alice",
            context_id="context-a",
            last_message_sequence=2,
            last_user_message_sequence=3,
            created_at=NOW,
            updated_at=NOW,
        )

    with pytest.raises(ValidationError, match="archived Conversation"):
        conversation.model_copy(update={
            "status": "archived",
            "active_turn_id": TURN_ID,
        }).model_validate(
            conversation.model_copy(update={
                "status": "archived",
                "active_turn_id": TURN_ID,
            }).model_dump()
        )


def test_message_source_kind_cannot_claim_unrelated_evidence() -> None:
    direct = ConversationMessageContent(
        text="Hello.", source_kind="direct_conversation"
    )
    assert direct.citations == ()

    with pytest.raises(ValidationError, match="requires a Result"):
        ConversationMessageContent(
            text="The procedure is mandatory.",
            source_kind="research_result",
            citations=("evidence-1",),
        )

    with pytest.raises(ValidationError, match="cannot claim research evidence"):
        ConversationMessageContent(
            text="Hello.",
            source_kind="direct_conversation",
            research_result_ids=(RESULT_ID,),
        )


def test_turn_trigger_and_terminal_state_are_explicit() -> None:
    automatic = ConversationTurn(
        turn_id=TURN_ID,
        conversation_id=CONVERSATION_ID,
        trigger="research_result_ready",
        trigger_result_id=RESULT_ID,
        trigger_run_id=RUN_ID,
        submitted_user_sequence=2,
        budget=ConversationTurnBudget(max_research_submissions=0),
        operation_key="present-result",
        created_at=NOW,
    )
    assert automatic.status == "queued"

    with pytest.raises(ValidationError, match="cannot submit research"):
        ConversationTurn(
            turn_id=TURN_ID,
            conversation_id=CONVERSATION_ID,
            trigger="research_result_ready",
            trigger_result_id=RESULT_ID,
            trigger_run_id=RUN_ID,
            submitted_user_sequence=2,
            operation_key="present-result",
            created_at=NOW,
        )

    with pytest.raises(ValidationError, match="reply message"):
        ConversationTurn(
            turn_id=TURN_ID,
            conversation_id=CONVERSATION_ID,
            trigger="user_message",
            trigger_message_id="message-1",
            submitted_user_sequence=2,
            status="completed",
            operation_key="reply",
            created_at=NOW,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
        )


def test_research_link_result_and_presentation_state_move_together() -> None:
    link = ConversationResearchLink(
        link_id="LINK-" + "F" * 32,
        conversation_id=CONVERSATION_ID,
        origin_message_id="message-1",
        origin_turn_id=TURN_ID,
        tool_call_id=CALL_ID,
        run_id=RUN_ID,
        submitted_user_sequence=1,
        task_status="running",
        created_at=NOW,
        updated_at=NOW,
    )
    assert link.presentation_status is None

    with pytest.raises(ValidationError, match="presentation state requires"):
        ConversationResearchLink.model_validate({
            **link.model_dump(),
            "presentation_status": "pending",
        })


@pytest.mark.parametrize(
    ("validator", "current", "target"),
    [
        (require_conversation_status_transition, "active", "archived"),
        (require_turn_status_transition, "queued", "running"),
        (require_turn_status_transition, "running", "queued"),
        (require_tool_call_status_transition, "proposed", "running"),
        (require_presentation_status_transition, "pending", "queued"),
        (require_presentation_status_transition, "queued", "presented"),
    ],
)
def test_legal_lifecycle_transitions(validator, current, target) -> None:
    assert validator(current, target) == target


@pytest.mark.parametrize(
    ("validator", "current", "target"),
    [
        (require_conversation_status_transition, "active", "active"),
        (require_turn_status_transition, "completed", "running"),
        (require_tool_call_status_transition, "completed", "running"),
        (require_presentation_status_transition, "presented", "pending"),
    ],
)
def test_terminal_and_duplicate_transitions_are_rejected(
    validator, current, target
) -> None:
    with pytest.raises(ValueError, match="invalid"):
        validator(current, target)
