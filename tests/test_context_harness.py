from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from specrag.context import (
    CompactionRejectedError,
    ContextHarness,
    InMemoryContextStore,
    ModelProfileResolver,
    TokenBudgetManager,
)
from specrag.infrastructure import SQLiteContextStore, SQLiteDatabase
from specrag.infrastructure.llm import OpenAIStructuredClient
from specrag.models import CompactionSummary, ModelContextProfile


def _budgets(
    *, context_window: int = 1_000, expected_output: int = 100,
) -> TokenBudgetManager:
    return TokenBudgetManager(
        ModelProfileResolver(ModelContextProfile(
            model_id="default",
            context_window_tokens=context_window,
            max_input_tokens=context_window,
            expected_output_tokens=expected_output,
        )),
        safety_ratio=0.15,
        large_window_threshold_tokens=100_000,
        large_window_minimum_reserve_tokens=16_000,
    )


class _Summary:
    def __init__(self, reference: str = "MSG-OLD") -> None:
        self.reference = reference
        self.calls = 0

    def summarize(self, request):
        self.calls += 1
        return CompactionSummary(
            objective="Continue the task",
            decisions=("Keep the durable result.",),
            next_moves=("Process the recent turn.",),
            references=(self.reference,),
        )


def _long_history(harness: ContextHarness, context_id: str) -> None:
    harness.create_context(context_id)
    harness.append_message(
        context_id, "user", "old objective " * 200,
        turn_id="TURN-1", message_id="MSG-OLD",
    )
    harness.append_message(
        context_id, "assistant", "old work " * 200,
        turn_id="TURN-1", message_id="MSG-WORK",
    )
    harness.append_message(
        context_id, "user", "recent request " * 20,
        turn_id="TURN-2", message_id="MSG-RECENT",
    )


def test_budget_reserve_and_model_override() -> None:
    default = ModelContextProfile(
        model_id="default", context_window_tokens=128_000,
        max_input_tokens=120_000, expected_output_tokens=8_000,
    )
    small = ModelContextProfile(
        model_id="small", context_window_tokens=20_000,
        max_input_tokens=18_000, expected_output_tokens=2_000,
    )
    manager = TokenBudgetManager(ModelProfileResolver(default, {"small": small}))
    large_budget = manager.inspect("large", {"input": "hello"})
    small_budget = manager.inspect("small", {"input": "hello"})
    assert large_budget.reserve_tokens == 19_200
    assert large_budget.input_ceiling_tokens == 108_800
    assert small_budget.reserve_tokens == 3_000
    assert small_budget.input_ceiling_tokens == 17_000


def test_tool_result_is_durable_and_pageable() -> None:
    harness = ContextHarness(
        InMemoryContextStore(), _budgets(), tool_result_soft_limit_tokens=12,
    )
    content = {"items": [f"result-{index}-" + "x" * 20 for index in range(20)]}
    first = harness.record_tool_result("CTX-1", "CALL-1", content)
    assert first.truncated is True
    assert first.next_cursor is not None

    fragments = [first.content["json_fragment"]]
    page = first
    while page.next_cursor is not None:
        page = harness.read_tool_result(
            first.tool_result_id, cursor=page.next_cursor, max_tokens=12,
        )
        fragments.append(page.content["json_fragment"])
    assert json.loads("".join(fragments)) == content

    repeated = harness.record_tool_result("CTX-1", "CALL-1", content)
    assert repeated.tool_result_id == first.tool_result_id


def test_compaction_activates_checkpoint_without_deleting_history() -> None:
    store = InMemoryContextStore()
    harness = ContextHarness(store, _budgets(), recent_history_tokens=150)
    _long_history(harness, "CTX-2")
    checkpoint = harness.compact("CTX-2", _Summary(), trigger="manual")

    assert checkpoint.covers_through_message_id == "MSG-WORK"
    assert checkpoint.recent_start_message_id == "MSG-RECENT"
    assert checkpoint.after_tokens < checkpoint.before_tokens
    active = harness.reconstruct("CTX-2")
    assert active.summary is not None
    assert tuple(item.message_id for item in active.messages) == ("MSG-RECENT",)
    assert len(store.messages("CTX-2")) == 3


def test_compaction_retries_invalid_references_then_rejects() -> None:
    harness = ContextHarness(
        InMemoryContextStore(), _budgets(), recent_history_tokens=150,
        summary_validation_attempts=2,
    )
    _long_history(harness, "CTX-3")
    generator = _Summary("MADE-UP-ID")
    with pytest.raises(CompactionRejectedError, match="absent"):
        harness.compact("CTX-3", generator, trigger="auto")
    assert generator.calls == 2
    assert harness.reconstruct("CTX-3").summary is None


def test_overflow_recovery_retries_only_once() -> None:
    harness = ContextHarness(
        InMemoryContextStore(), _budgets(), recent_history_tokens=150,
    )
    _long_history(harness, "CTX-4")
    calls = 0

    class Overflow(RuntimeError):
        failure_kind = "context_overflow"

    def invoke() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Overflow()
        return "ok"

    assert harness.call_with_overflow_recovery(
        "CTX-4", invoke, _Summary(), model_id="model-a",
    ) == "ok"
    assert calls == 2


def test_prepare_auto_compacts_with_the_same_harness() -> None:
    store = InMemoryContextStore()
    harness = ContextHarness(store, _budgets(), recent_history_tokens=150)
    _long_history(harness, "CTX-AUTO")

    active, budget = harness.prepare(
        "CTX-AUTO",
        "model-a",
        {"system": "bounded test"},
        generator=_Summary(),
    )

    assert budget.fits is True
    assert active.summary is not None
    context = store.get_context("CTX-AUTO")
    assert context is not None and context.active_checkpoint_id is not None
    checkpoint = store.get_checkpoint(context.active_checkpoint_id)
    assert checkpoint is not None and checkpoint.trigger == "auto"


def test_sqlite_context_round_trip(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "context.sqlite3")
    try:
        harness = ContextHarness(
            SQLiteContextStore(database), _budgets(), recent_history_tokens=150,
        )
        _long_history(harness, "CTX-SQL")
        page = harness.record_tool_result("CTX-SQL", "CALL-SQL", {"value": "x" * 500})
        checkpoint = harness.compact("CTX-SQL", _Summary(), trigger="manual")

        restored = ContextHarness(
            SQLiteContextStore(database), _budgets(), recent_history_tokens=150,
        )
        assert restored.read_tool_result(page.tool_result_id).content_sha256 == page.content_sha256
        assert restored.reconstruct("CTX-SQL").summary == checkpoint.summary
    finally:
        database.close()


class _Output(BaseModel):
    value: str


class _Responses:
    def __init__(self) -> None:
        self.models: list[str] = []

    def parse(self, **kwargs):
        self.models.append(kwargs["model"])
        return SimpleNamespace(
            status="completed",
            output_parsed={"value": "ok"},
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
            id="response-1",
        )


class _CorrectingResponses:
    def __init__(self) -> None:
        self.requests = 0

    def parse(self, **kwargs):
        self.requests += 1
        parsed = {} if self.requests == 1 else {"value": "corrected"}
        return SimpleNamespace(
            status="completed",
            output_parsed=parsed,
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
            id=f"response-{self.requests}",
        )


def test_structured_schema_failure_gets_bounded_correction_retry() -> None:
    responses = _CorrectingResponses()
    client = OpenAIStructuredClient(
        api_key="test",
        model="primary",
        max_retries=2,
        retry_initial_delay_seconds=0,
        retry_max_delay_seconds=0,
        client=SimpleNamespace(responses=responses),
    )

    response = client.parse(
        purpose="test",
        prompt_version="test/v1",
        instructions="Return the requested schema.",
        input_payload={"question": "q"},
        output_type=_Output,
    )

    assert response.output.value == "corrected"
    assert response.receipt.attempt_count == 2
    assert responses.requests == 2


def test_fallback_model_gets_its_own_preflight() -> None:
    responses = _Responses()
    profiles = ModelProfileResolver(
        ModelContextProfile(
            model_id="default", context_window_tokens=10_000,
            max_input_tokens=10_000, expected_output_tokens=100,
        ),
        {
            "primary": ModelContextProfile(
                model_id="primary", context_window_tokens=100,
                max_input_tokens=100, expected_output_tokens=50,
            ),
            "fallback": ModelContextProfile(
                model_id="fallback", context_window_tokens=10_000,
                max_input_tokens=10_000, expected_output_tokens=100,
            ),
        },
    )
    client = OpenAIStructuredClient(
        api_key="test", model="primary", fallback_models=("fallback",),
        max_output_tokens=50, max_retries=0,
        client=SimpleNamespace(responses=responses),
        budget_manager=TokenBudgetManager(profiles),
    )
    response = client.parse(
        purpose="test", prompt_version="test/v1", instructions="x" * 200,
        input_payload={"value": "y" * 200}, output_type=_Output,
    )
    assert responses.models == ["fallback"]
    assert response.receipt.model == "fallback"
    assert response.receipt.estimated_input_tokens is not None
