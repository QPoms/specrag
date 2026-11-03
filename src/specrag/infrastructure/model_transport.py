"""Explicit, serializable provider transport choices for one model role."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm_contracts import LLMApiMode


class ModelTransportProfile(BaseModel):
    """Frozen API-mode capabilities selected after a real provider probe.

    Profiles are caller-selected experiment inputs.  The adapter never infers
    them from a model name and never switches profiles after a failed call.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=256)
    structured_api_mode: LLMApiMode = LLMApiMode.RESPONSES
    reasoning_effort: str | None = Field(default="high", min_length=1, max_length=32)
    supports_native_tools: bool = True

    @model_validator(mode="after")
    def _coherent_profile(self) -> "ModelTransportProfile":
        if not self.model_id.strip() or self.model_id != self.model_id.strip():
            raise ValueError("transport model_id must be non-blank and trimmed")
        if self.reasoning_effort is not None and (
            not self.reasoning_effort.strip()
            or self.reasoning_effort != self.reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort must be null or non-blank and trimmed")
        if self.supports_native_tools and (
            self.structured_api_mode != LLMApiMode.RESPONSES
        ):
            raise ValueError("native tools require the Responses API mode")
        return self


def resolve_transport_profile(
    *,
    model_id: str,
    api_mode: LLMApiMode,
    reasoning_effort: str,
    explicit: ModelTransportProfile | None,
) -> ModelTransportProfile:
    """Resolve an explicit profile or preserve the existing default transport."""

    profile = explicit or ModelTransportProfile(
        model_id=model_id,
        structured_api_mode=api_mode,
        reasoning_effort=reasoning_effort.strip() or "high",
        supports_native_tools=(api_mode == LLMApiMode.RESPONSES),
    )
    if profile.model_id != model_id:
        raise ValueError("transport profile model does not match selected model")
    return profile


def transport_reasoning_options(
    reasoning_effort: str | None,
    *,
    api_mode: LLMApiMode,
) -> dict[str, Any]:
    """Return no provider field when the selected profile uses its default."""

    if reasoning_effort is None:
        return {}
    if api_mode == LLMApiMode.RESPONSES:
        return {"reasoning": {"effort": reasoning_effort}}
    return {"reasoning_effort": reasoning_effort}


__all__ = [
    "ModelTransportProfile",
    "resolve_transport_profile",
    "transport_reasoning_options",
]
