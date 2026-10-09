"""Narrow structured-output client used by agent components.

The client deliberately exposes neither credentials nor a general chat API.
Callers submit a purpose, fixed instructions, a JSON-compatible public task,
and a strict Pydantic output type.  Only a parsed model and a redacted receipt
cross back into the analyzer layer.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.config import AppConfig, load_config


OutputT = TypeVar("OutputT", bound=BaseModel)


class LLMApiMode(str, Enum):
    """Explicit API surface; fallback is a caller-owned experiment choice."""

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
    response_id_sha256: str = Field(
        default="",
        pattern=r"^(?:[0-9a-f]{64})?$",
    )
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    attempt_count: int = Field(default=1, ge=1)
    attempted_models: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _attempt_provenance(self) -> "LLMCallReceipt":
        if any(
            not model.strip() or model != model.strip()
            for model in self.attempted_models
        ):
            raise ValueError("attempted model names must be non-blank and trimmed")
        if self.attempted_models and (
            len(self.attempted_models) != self.attempt_count
            or self.attempted_models[-1] != self.model
        ):
            raise ValueError("LLM receipt attempt provenance is inconsistent")
        return self


@dataclass(frozen=True)
class StructuredLLMResponse(Generic[OutputT]):
    """One schema-validated response and its redacted call receipt."""

    output: OutputT
    receipt: LLMCallReceipt


class StructuredLLMClient(Protocol):
    """Small dependency boundary suitable for fake and sandboxed clients."""

    def parse(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: BaseModel | Mapping[str, Any],
        output_type: type[OutputT],
    ) -> StructuredLLMResponse[OutputT]: ...


class StructuredLLMError(RuntimeError):
    """Sanitized API/schema failure safe to persist as an analyzer gap."""


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_error(
    exc: Exception,
    *,
    attempt_count: int | None = None,
    exhausted_models: int | None = None,
) -> StructuredLLMError:
    """Do not persist provider bodies, headers, request objects, or prompts."""

    details = [type(exc).__name__]
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if isinstance(status_code, int):
        details.append(f"status={status_code}")
    if isinstance(request_id, str) and request_id:
        details.append(f"request_id_sha256={_sha256(request_id)[:16]}")
    if attempt_count is not None:
        details.append(f"attempts={attempt_count}")
    if exhausted_models is not None:
        details.append(f"models={exhausted_models}")
    return StructuredLLMError(
        "structured LLM request failed (" + ", ".join(details) + ")"
    )


def _is_retryable_provider_error(exc: Exception) -> bool:
    """Classify transient failures without retaining provider response bodies."""

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 425, 429} or status_code >= 500
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RateLimitError",
        "TimeoutError",
    }


class OpenAIStructuredClient:
    """Official OpenAI SDK adapter with an explicit, non-automatic API mode."""

    __slots__ = (
        "_api_mode",
        "_client",
        "_fallback_models",
        "_max_output_tokens",
        "_max_retries",
        "_model",
        "_reasoning_effort",
        "_retry_initial_delay_seconds",
        "_retry_max_delay_seconds",
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        api_mode: LLMApiMode = LLMApiMode.RESPONSES,
        reasoning_effort: str = "low",
        max_output_tokens: int = 8_000,
        timeout_seconds: float = 120.0,
        max_retries: int = 8,
        fallback_models: tuple[str, ...] = (),
        retry_initial_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 15.0,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for SPEC LLM mapping")
        if not model:
            raise ValueError("OPENAI_MODEL is required for SPEC LLM mapping")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_initial_delay_seconds < 0:
            raise ValueError("retry_initial_delay_seconds must be non-negative")
        if retry_max_delay_seconds < retry_initial_delay_seconds:
            raise ValueError("retry_max_delay_seconds is below initial delay")
        models = tuple(dict.fromkeys((model, *fallback_models)))
        if any(not item.strip() or item != item.strip() for item in models):
            raise ValueError("model names must be non-blank and trimmed")
        self._model = model
        self._fallback_models = models[1:]
        self._api_mode = LLMApiMode(api_mode)
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries
        self._retry_initial_delay_seconds = retry_initial_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        if client is None:
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout_seconds,
                # Retry explicitly below so attempt/model provenance is auditable
                # and SDK retries do not multiply the configured bound.
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
        self._client = client

    def __repr__(self) -> str:
        return (
            "OpenAIStructuredClient("
            f"model={self._model!r}, api_mode={self._api_mode.value!r})"
        )

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig | None = None,
        *,
        api_mode: LLMApiMode = LLMApiMode.RESPONSES,
        reasoning_effort: str = "low",
    ) -> "OpenAIStructuredClient":
        config = cfg or load_config()
        secrets = config.secrets
        return cls(
            api_key=secrets.openai_api_key,
            base_url=secrets.openai_base_url,
            model=secrets.openai_model,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            max_retries=config.llm.api_max_retries,
            fallback_models=config.llm.api_fallback_models,
            retry_initial_delay_seconds=(
                config.llm.api_retry_initial_delay_seconds
            ),
            retry_max_delay_seconds=config.llm.api_retry_max_delay_seconds,
        )

    def parse(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: BaseModel | Mapping[str, Any],
        output_type: type[OutputT],
    ) -> StructuredLLMResponse[OutputT]:
        for name, value in (
            ("purpose", purpose),
            ("prompt_version", prompt_version),
            ("instructions", instructions),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-blank")

        input_json = _canonical_json(input_payload)
        schema_json = _canonical_json(output_type.model_json_schema())
        started = time.perf_counter()
        attempted_models: list[str] = []
        raw: Any = None
        output: OutputT | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        selected_model = self._model
        models = (self._model, *self._fallback_models)
        for model_index, candidate_model in enumerate(models):
            for retry_index in range(self._max_retries + 1):
                attempted_models.append(candidate_model)
                try:
                    if self._api_mode == LLMApiMode.RESPONSES:
                        raw = self._client.responses.parse(
                            model=candidate_model,
                            instructions=instructions,
                            input=input_json,
                            text_format=output_type,
                            reasoning={"effort": self._reasoning_effort},
                            max_output_tokens=self._max_output_tokens,
                            store=False,
                        )
                        status = getattr(raw, "status", "completed")
                        if status != "completed":
                            incomplete = getattr(raw, "incomplete_details", None)
                            reason = getattr(incomplete, "reason", None)
                            suffix = f", reason={reason}" if reason else ""
                            raise StructuredLLMError(
                                "structured LLM response was not completed "
                                f"(status={status}{suffix})"
                            )
                        parsed = raw.output_parsed
                        usage = getattr(raw, "usage", None)
                        input_tokens = getattr(usage, "input_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                    else:
                        raw = self._client.chat.completions.parse(
                            model=candidate_model,
                            messages=(
                                {"role": "system", "content": instructions},
                                {"role": "user", "content": input_json},
                            ),
                            response_format=output_type,
                            reasoning_effort=self._reasoning_effort,
                            max_completion_tokens=self._max_output_tokens,
                            store=False,
                        )
                        choice = raw.choices[0]
                        finish_reason = getattr(choice, "finish_reason", "stop")
                        if finish_reason != "stop":
                            raise StructuredLLMError(
                                "structured chat completion did not finish normally "
                                f"(finish_reason={finish_reason})"
                            )
                        refusal = getattr(choice.message, "refusal", None)
                        if refusal:
                            raise StructuredLLMError(
                                "structured chat completion was refused"
                            )
                        parsed = choice.message.parsed
                        usage = getattr(raw, "usage", None)
                        input_tokens = getattr(usage, "prompt_tokens", None)
                        output_tokens = getattr(usage, "completion_tokens", None)
                    if parsed is None:
                        raise StructuredLLMError(
                            "structured LLM response contained no parsed output"
                        )
                    output = output_type.model_validate(parsed)
                    selected_model = candidate_model
                    break
                except StructuredLLMError:
                    raise
                except Exception as exc:  # noqa: BLE001 - sanitized below
                    if not _is_retryable_provider_error(exc):
                        raise _safe_error(
                            exc,
                            attempt_count=len(attempted_models),
                        ) from None
                    if retry_index < self._max_retries:
                        delay = min(
                            self._retry_initial_delay_seconds * (2 ** retry_index),
                            self._retry_max_delay_seconds,
                        )
                        if delay:
                            time.sleep(delay)
                        continue
                    if model_index + 1 < len(models):
                        break
                    raise _safe_error(
                        exc,
                        attempt_count=len(attempted_models),
                        exhausted_models=len(models),
                    ) from None
            if output is not None:
                break
        if output is None or raw is None:  # pragma: no cover - loop is exhaustive
            raise StructuredLLMError("structured LLM retry loop produced no result")

        request_sha = _sha256(_canonical_json({
            "model": selected_model,
            "api_mode": self._api_mode.value,
            "purpose": purpose,
            "prompt_version": prompt_version,
            "instructions": instructions,
            "input": json.loads(input_json),
            "output_schema": json.loads(schema_json),
        }))

        response_json = _canonical_json(output)
        receipt = LLMCallReceipt(
            model=selected_model,
            api_mode=self._api_mode,
            purpose=purpose,
            prompt_version=prompt_version,
            request_sha256=request_sha,
            response_sha256=_sha256(response_json),
            response_id_sha256=(
                _sha256(str(getattr(raw, "id", "")))
                if getattr(raw, "id", "") else ""
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=max(0, int((time.perf_counter() - started) * 1_000)),
            attempt_count=len(attempted_models),
            attempted_models=tuple(attempted_models),
        )
        return StructuredLLMResponse(output=output, receipt=receipt)
