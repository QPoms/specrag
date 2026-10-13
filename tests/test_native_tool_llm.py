"""Characterize the Responses API native function-call boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from specrag.infrastructure.llm import (
    OpenAIStructuredClient,
    StructuredLLMError,
)
from specrag.models import (
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
)


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
