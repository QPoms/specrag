"""Secret-safe startup probes for configured model capabilities."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from specrag.models import (
    RUN_FAILURE_KINDS,
    NativeFunctionDefinition,
    NativeFunctionCallOutput,
    NativeToolCapability,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _StructuredProbeOutput(_StrictModel):
    status: Literal["ready"]


class ProviderCapabilityCheck(_StrictModel):
    """One secret-safe provider capability result."""

    status: Literal["ready", "unavailable"]
    api_mode: Literal["responses", "chat_completions"] | None = None
    observed_model: str | None = Field(default=None, min_length=1, max_length=256)
    failure_kind: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    retryable: bool = False
    message: str = Field(min_length=1, max_length=1000)


class ResearchModelCapability(_StrictModel):
    """Structured-output plus replayable native-tool readiness."""

    status: Literal["ready", "unavailable"]
    structured_output: ProviderCapabilityCheck
    native_tool_replay: ProviderCapabilityCheck


class NativeToolModel(Protocol):
    def call_tools(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: Mapping[str, Any],
        tools: Sequence[NativeFunctionDefinition],
        prior_response_items: Sequence[Mapping[str, Any]] = (),
        prior_outputs: Sequence[NativeFunctionCallOutput] = (),
        max_tool_calls: int | None = None,
    ): ...


class ResearchProbeModel(NativeToolModel, Protocol):
    def parse(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: Mapping[str, Any],
        output_type: type[BaseModel],
    ): ...


def _api_mode(response: Any) -> Literal["responses", "chat_completions"]:
    value = getattr(getattr(response, "receipt", None), "api_mode", "responses")
    if hasattr(value, "value"):
        value = value.value
    return value if value in {"responses", "chat_completions"} else "responses"


def _observed_model(response: Any) -> str | None:
    value = getattr(getattr(response, "receipt", None), "model", None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_failure(exc: Exception, default_message: str) -> ProviderCapabilityCheck:
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None
    failure_kind = getattr(exc, "failure_kind", "unknown")
    if failure_kind not in RUN_FAILURE_KINDS:
        failure_kind = "unknown"
    api_mode = getattr(exc, "api_mode", None)
    message = getattr(exc, "public_message", None)
    if not isinstance(message, str) or not message.strip():
        message = default_message
    return ProviderCapabilityCheck(
        status="unavailable",
        api_mode=api_mode if api_mode in {"responses", "chat_completions"} else None,
        failure_kind=failure_kind,
        status_code=status_code,
        retryable=bool(getattr(exc, "retryable", False)),
        message=message[:1000],
    )


def probe_native_tool_capability(
    model: NativeToolModel,
    tools: Sequence[NativeFunctionDefinition],
) -> NativeToolCapability:
    """Exercise the real tool schema without executing any returned call."""

    try:
        response = model.call_tools(
            purpose="conversation_tool_capability_probe",
            prompt_version="conversation-tool-capability/v1",
            instructions=(
                "This is a transport capability probe. Call the respond function "
                "once with text 'ready', source_kind 'direct_conversation', and "
                "empty research_result_ids, run_ids, and citations. Do not call "
                "any research function."
            ),
            input_payload={"probe": "native_tool_calling"},
            tools=tools,
            max_tool_calls=1,
        )
    except Exception as exc:  # noqa: BLE001 - projection is allowlisted below
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int) or not 100 <= status_code <= 599:
            status_code = None
        failure_kind = getattr(exc, "failure_kind", "unknown")
        if failure_kind not in RUN_FAILURE_KINDS:
            failure_kind = "unknown"
        api_mode = getattr(exc, "api_mode", None)
        message = getattr(exc, "public_message", None)
        if not isinstance(message, str) or not message.strip():
            message = (
                "The configured model could not complete the native tool "
                "capability probe."
            )
        return NativeToolCapability(
            status="unavailable",
            api_mode=(
                api_mode
                if api_mode in {"responses", "chat_completions"}
                else None
            ),
            failure_kind=failure_kind,
            status_code=status_code,
            retryable=bool(getattr(exc, "retryable", False)),
            message=message[:1000],
        )
    api_mode = getattr(response.receipt, "api_mode", "responses")
    if hasattr(api_mode, "value"):
        api_mode = api_mode.value
    if api_mode not in {"responses", "chat_completions"}:
        api_mode = "responses"
    return NativeToolCapability(
        status="ready",
        api_mode=api_mode,
        message=(
            "The configured model accepted the Conversation Agent native tool "
            f"contract through {api_mode}."
        ),
    )


def probe_research_model_capability(
    model: ResearchProbeModel,
) -> ResearchModelCapability:
    """Probe the exact structured and two-turn tool contracts used by research.

    The synthetic function is never executed.  Only an opaque matched output is
    replayed, and provider exception text is excluded from the result.
    """

    try:
        structured_response = model.parse(
            purpose="research_model_structured_capability_probe",
            prompt_version="research-model-capability/v1",
            instructions=(
                "This is a transport capability probe. Return the required "
                "structured object with status set to ready."
            ),
            input_payload={"probe": "structured_output"},
            output_type=_StructuredProbeOutput,
        )
        structured = ProviderCapabilityCheck(
            status="ready",
            api_mode=_api_mode(structured_response),
            observed_model=_observed_model(structured_response),
            message="The model accepted the strict structured-output contract.",
        )
    except Exception as exc:  # noqa: BLE001 - safe projection below
        structured = _safe_failure(
            exc, "The model could not complete the structured-output probe."
        )

    tool = NativeFunctionDefinition(
        name="capability_probe",
        description="Return the fixed marker for a transport-only probe.",
        parameters={
            "type": "object",
            "properties": {"marker": {"type": "string", "enum": ["ready"]}},
            "required": ["marker"],
            "additionalProperties": False,
        },
    )
    try:
        first = model.call_tools(
            purpose="research_model_native_tool_capability_probe",
            prompt_version="research-model-capability/v1",
            instructions=(
                "This is a transport capability probe. Call capability_probe "
                "once with marker ready."
            ),
            input_payload={"probe": "native_tool_first_turn"},
            tools=(tool,),
            max_tool_calls=1,
        )
        calls = tuple(getattr(first, "calls", ()))
        replay_items = tuple(getattr(first, "replay_items", ()))
        if len(calls) != 1 or calls[0].name != tool.name or not replay_items:
            raise RuntimeError("provider returned an invalid probe call")
        second = model.call_tools(
            purpose="research_model_native_tool_replay_capability_probe",
            prompt_version="research-model-capability/v1",
            instructions=(
                "This is a transport capability probe. After the prior output, "
                "call capability_probe once more with marker ready."
            ),
            input_payload={"probe": "native_tool_replay_turn"},
            tools=(tool,),
            prior_response_items=replay_items,
            prior_outputs=(
                NativeFunctionCallOutput(
                    call_id=calls[0].call_id,
                    output={"status": "completed"},
                ),
            ),
            max_tool_calls=1,
        )
        second_calls = tuple(getattr(second, "calls", ()))
        if len(second_calls) != 1 or second_calls[0].name != tool.name:
            raise RuntimeError("provider returned an invalid replay probe call")
        native = ProviderCapabilityCheck(
            status="ready",
            api_mode=_api_mode(second),
            observed_model=_observed_model(second),
            message="The model accepted the two-turn native-tool replay contract.",
        )
    except Exception as exc:  # noqa: BLE001 - safe projection below
        native = _safe_failure(
            exc, "The model could not complete the native-tool replay probe."
        )

    return ResearchModelCapability(
        status=(
            "ready"
            if structured.status == "ready" and native.status == "ready"
            else "unavailable"
        ),
        structured_output=structured,
        native_tool_replay=native,
    )


__all__ = [
    "NativeToolModel",
    "ProviderCapabilityCheck",
    "ResearchModelCapability",
    "ResearchProbeModel",
    "probe_native_tool_capability",
    "probe_research_model_capability",
]
