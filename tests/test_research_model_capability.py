from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from specrag.infrastructure.model_capabilities import (
    probe_research_model_capability,
)
from specrag.models import NativeFunctionCall


class _ReadyModel:
    def __init__(self) -> None:
        self.tool_inputs: list[dict[str, Any]] = []

    def parse(self, **_: Any):
        return SimpleNamespace(receipt=SimpleNamespace(
            api_mode="responses", model="test-model"
        ))

    def call_tools(self, **kwargs: Any):
        self.tool_inputs.append(kwargs)
        turn = len(self.tool_inputs)
        return SimpleNamespace(
            calls=(NativeFunctionCall(
                call_id=f"call-{turn}",
                name="capability_probe",
                arguments={"marker": "ready"},
            ),),
            replay_items=({"type": "function_call", "call_id": f"call-{turn}"},),
            receipt=SimpleNamespace(api_mode="responses", model="test-model"),
        )


def test_research_model_probe_exercises_structured_and_replay_contracts() -> None:
    model = _ReadyModel()
    capability = probe_research_model_capability(model)

    assert capability.status == "ready"
    assert capability.structured_output.status == "ready"
    assert capability.native_tool_replay.status == "ready"
    assert capability.structured_output.observed_model == "test-model"
    assert capability.native_tool_replay.observed_model == "test-model"
    assert len(model.tool_inputs) == 2
    assert model.tool_inputs[0].get("prior_response_items", ()) == ()
    assert model.tool_inputs[1]["prior_response_items"]
    assert model.tool_inputs[1]["prior_outputs"][0].call_id == "call-1"


def test_research_model_probe_projects_safe_failures_only() -> None:
    class Rejected:
        def parse(self, **_: Any):
            raise RuntimeError("raw structured provider body")

        def call_tools(self, **_: Any):
            error = RuntimeError("raw native provider body")
            error.failure_kind = "invalid_request"
            error.status_code = 400
            error.api_mode = "responses"
            error.public_message = "Safe native-tool diagnostic."
            raise error

    capability = probe_research_model_capability(Rejected())

    assert capability.status == "unavailable"
    assert capability.structured_output.failure_kind == "unknown"
    assert "raw structured" not in capability.structured_output.message
    assert capability.native_tool_replay.failure_kind == "invalid_request"
    assert capability.native_tool_replay.status_code == 400
    assert capability.native_tool_replay.message == "Safe native-tool diagnostic."
    assert "raw native" not in capability.native_tool_replay.message
