"""Characterize the Responses API native function-call boundary."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from specrag.infrastructure.llm import (
    LLMApiMode,
    OpenAIStructuredClient,
    StructuredLLMError,
    classify_provider_error,
)
from specrag.infrastructure.api_errors import (
    ProviderFailure,
    provider_public_message,
    safe_provider_field,
)
from specrag.infrastructure.model_capabilities import probe_native_tool_capability
from specrag.models import (
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
    RUN_FAILURE_KINDS,
)
from specrag.tools.research import conversation_research_tool_definitions


class _FunctionItem:
    def __init__(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.type = "function_call"
        self.call_id = call_id
        self.name = name
        self.arguments = json.dumps(arguments)

    def model_dump(self, **_: object) -> dict[str, Any]:
        return {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


class _TextItem:
    type = "message"

    def model_dump(self, **_: object) -> dict[str, Any]:
        return {"type": self.type, "content": "not a tool call"}


class _StructuredOutput(BaseModel):
    value: str


class _RefusingChatCompletions:
    def parse(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(refusal="policy refusal", parsed=None),
            )],
            usage=None,
        )


class _Responses:
    def __init__(self, outputs: list[list[object]]) -> None:
        self._outputs = outputs
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        output = self._outputs.pop(0)
        return SimpleNamespace(
            id=f"response-{len(self.requests)}",
            status="completed",
            output=output,
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


def _client(outputs: list[list[object]]) -> tuple[OpenAIStructuredClient, _Responses]:
    responses = _Responses(outputs)
    client = OpenAIStructuredClient(
        api_key="test-key",
        model="test-model",
        max_retries=0,
        client=SimpleNamespace(responses=responses),
    )
    return client, responses


def _tool() -> NativeFunctionDefinition:
    return NativeFunctionDefinition(
        name="semantic_search",
        description="Find semantically related source passages.",
        parameters={
            "type": "object",
            "properties": {"semantic_query": {"type": "string"}},
            "required": ["semantic_query"],
            "additionalProperties": False,
        },
    )


def test_native_call_rejects_expired_run_deadline_before_provider_request() -> None:
    responses = _Responses([[
        _FunctionItem("call-search", "semantic_search", {"semantic_query": "x"})
    ]])
    client = OpenAIStructuredClient(
        api_key="test-key",
        model="test-model",
        max_retries=0,
        deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        client=SimpleNamespace(responses=responses),
    )

    with pytest.raises(StructuredLLMError) as error:
        client.call_tools(
            purpose="researcher",
            prompt_version="test/v1",
            instructions="Use functions.",
            input_payload={"round": 1},
            tools=(_tool(),),
        )

    assert error.value.failure_kind == "timeout"
    assert responses.requests == []


def test_native_call_parses_multiple_calls_and_replays_outputs() -> None:
    client, responses = _client([
        [
            _FunctionItem("call-control", "update_research_state", {"x": 1}),
            _FunctionItem("call-search", "semantic_search", {"semantic_query": "x"}),
        ],
        [_FunctionItem("call-submit", "submit_for_validation", {"x": 2})],
    ])

    first = client.call_tools(
        purpose="researcher",
        prompt_version="test/v1",
        instructions="Use functions.",
        input_payload={"round": 1},
        tools=(_tool(),),
        max_tool_calls=2,
    )
    second = client.call_tools(
        purpose="researcher",
        prompt_version="test/v1",
        instructions="Use functions.",
        input_payload={"round": 2},
        tools=(_tool(),),
        prior_response_items=first.replay_items,
        prior_outputs=(
            NativeFunctionCallOutput(
                call_id="call-search",
                output={"status": "accepted", "evidence_ids": ["E1"]},
            ),
        ),
    )

    assert [call.name for call in first.calls] == [
        "update_research_state",
        "semantic_search",
    ]
    assert second.calls[0].name == "submit_for_validation"
    request = responses.requests[0]
    assert request["store"] is False
    assert request["parallel_tool_calls"] is True
    assert request["max_tool_calls"] == 2
    assert "include" not in request
    assert request["tools"][0]["strict"] is True
    replay_input = responses.requests[1]["input"]
    assert replay_input[0]["call_id"] == "call-control"
    assert replay_input[2]["type"] == "function_call_output"
    assert json.loads(replay_input[2]["output"])["evidence_ids"] == ["E1"]
    assert first.receipt.response_id_sha256
    assert "test-key" not in first.receipt.model_dump_json()


def test_native_call_rejects_response_without_function_calls() -> None:
    client, _ = _client([[_TextItem()]])

    with pytest.raises(StructuredLLMError, match="no function calls"):
        client.call_tools(
            purpose="researcher",
            prompt_version="test/v1",
            instructions="Use functions.",
            input_payload={"round": 1},
            tools=(_tool(),),
        )


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (SimpleNamespace(status_code=429), "rate_limited", True),
        (SimpleNamespace(status_code=503), "provider_unavailable", True),
        (SimpleNamespace(status_code=401), "authentication_or_configuration", False),
        (SimpleNamespace(status_code=400), "invalid_request", False),
        (SimpleNamespace(
            status_code=400, code="context_length_exceeded"
        ), "context_overflow", False),
        (SimpleNamespace(status_code=408), "timeout", True),
        (SimpleNamespace(status_code=None), "unknown", False),
    ],
)
def test_provider_error_classification_is_safe(error, kind, retryable) -> None:
    result = classify_provider_error(error)
    assert result.kind == kind
    assert result.retryable is retryable


def test_structured_failure_keeps_only_safe_retry_metadata() -> None:
    failure = StructuredLLMError(
        "temporary failure",
        failure_kind="rate_limited",
        retryable=True,
        status_code=429,
        attempt_count=4,
    )
    assert failure.failure_kind == "rate_limited"
    assert failure.retryable is True
    assert failure.status_code == 429
    assert failure.attempt_count == 4


def test_failure_vocabulary_is_shared_and_accepts_model_refusal() -> None:
    assert "model_refusal" in RUN_FAILURE_KINDS
    assert "context_overflow" in RUN_FAILURE_KINDS
    failure = StructuredLLMError(
        "refused",
        failure_kind="model_refusal",
        retryable=False,
    )
    assert failure.failure_kind == "model_refusal"


def test_chat_refusal_is_projected_as_model_refusal() -> None:
    client = OpenAIStructuredClient(
        api_key="test-key",
        model="test-model",
        api_mode=LLMApiMode.CHAT_COMPLETIONS,
        max_retries=0,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_RefusingChatCompletions())
        ),
    )

    with pytest.raises(StructuredLLMError) as caught:
        client.parse(
            purpose="test",
            prompt_version="test/v1",
            instructions="Return structured output.",
            input_payload={"question": "test"},
            output_type=_StructuredOutput,
        )

    assert caught.value.failure_kind == "model_refusal"
    assert caught.value.retryable is False


def test_conversation_tools_use_provider_valid_strict_object_schemas() -> None:
    def check(schema: dict[str, Any]) -> None:
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            assert set(schema.get("required", ())) == set(properties)
            assert schema.get("additionalProperties") is False
            for value in properties.values():
                check(value)
        if schema.get("type") == "array":
            check(schema["items"])
        for branch in schema.get("anyOf", ()):  # nullable object branches
            check(branch)
        assert "uniqueItems" not in schema

    tools = conversation_research_tool_definitions()
    assert {tool.name for tool in tools} == {
        "start_research", "get_research_status", "get_research_result", "respond",
    }
    for tool in tools:
        assert tool.strict is True
        check(tool.parameters)


def test_provider_diagnostic_exposes_only_allowlisted_metadata() -> None:
    error = SimpleNamespace(
        param="tools[0].parameters",
        code="invalid_function_parameters",
        body={"error": {"message": "secret provider prose must not escape"}},
    )
    message = provider_public_message(
        ProviderFailure("invalid_request", False, 400),
        operation="native_tool_calling",
        api_mode="responses",
        provider_code=safe_provider_field(error, "code"),
        provider_parameter=safe_provider_field(error, "param"),
    )
    assert "HTTP 400" in message
    assert "tools[0].parameters" in message
    assert "secret provider prose" not in message


def test_native_tool_probe_returns_ready_or_safe_failure() -> None:
    class Ready:
        def call_tools(self, **_: Any):
            return SimpleNamespace(receipt=SimpleNamespace(api_mode="responses"))

    class Rejected:
        def call_tools(self, **_: Any):
            error = RuntimeError("raw provider payload")
            error.failure_kind = "invalid_request"
            error.status_code = 400
            error.retryable = False
            error.api_mode = "responses"
            error.public_message = "Safe native tool diagnostic."
            raise error

    ready = probe_native_tool_capability(Ready(), conversation_research_tool_definitions())
    rejected = probe_native_tool_capability(
        Rejected(), conversation_research_tool_definitions()
    )
    assert ready.status == "ready" and ready.api_mode == "responses"
    assert rejected.status == "unavailable"
    assert rejected.status_code == 400
    assert rejected.message == "Safe native tool diagnostic."
    assert "raw provider payload" not in rejected.message
