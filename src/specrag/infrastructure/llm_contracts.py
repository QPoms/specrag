"""Provider-neutral contracts returned by the structured LLM adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.models import NativeFunctionCall


OutputT = TypeVar("OutputT", bound=BaseModel)


class LLMApiMode(str, Enum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


class LLMCallReceipt(BaseModel):
    """Safe, reproducible metadata without prompts, endpoints, or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=256)
    api_mode: LLMApiMode
    purpose: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=256)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    attempt_count: int = Field(default=1, ge=1)
    attempted_models: tuple[str, ...] = Field(default_factory=tuple)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    input_ceiling_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _attempt_provenance(self) -> "LLMCallReceipt":
        if any(not model.strip() or model != model.strip() for model in self.attempted_models):
            raise ValueError("attempted model names must be non-blank and trimmed")
        if self.attempted_models and (
            len(self.attempted_models) != self.attempt_count
            or self.attempted_models[-1] != self.model
        ):
            raise ValueError("LLM receipt attempt provenance is inconsistent")
        return self


@dataclass(frozen=True)
class StructuredLLMResponse(Generic[OutputT]):
    output: OutputT
    receipt: LLMCallReceipt


@dataclass(frozen=True)
class NativeToolLLMResponse:
    calls: tuple[NativeFunctionCall, ...]
    replay_items: tuple[dict[str, Any], ...]
    receipt: LLMCallReceipt


class StructuredLLMClient(Protocol):
    def parse(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: BaseModel | Mapping[str, Any],
        output_type: type[OutputT],
    ) -> StructuredLLMResponse[OutputT]: ...


__all__ = [
    "LLMApiMode", "LLMCallReceipt", "NativeToolLLMResponse", "OutputT",
    "StructuredLLMClient", "StructuredLLMResponse",
]
