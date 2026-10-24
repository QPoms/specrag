"""Secret-safe startup probes for configured model capabilities."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from specrag.models import (
    RUN_FAILURE_KINDS,
    NativeFunctionDefinition,
    NativeToolCapability,
)


class NativeToolModel(Protocol):
    def call_tools(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: Mapping[str, Any],
        tools: Sequence[NativeFunctionDefinition],
        max_tool_calls: int | None = None,
    ): ...


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


__all__ = ["NativeToolModel", "probe_native_tool_capability"]
