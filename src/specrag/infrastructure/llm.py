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
from typing import Any, Mapping, Sequence

from openai import OpenAI
from pydantic import BaseModel

from specrag.config import AppConfig, load_config
from specrag.context import ContextBudgetExceededError, ModelProfileResolver, TokenBudgetManager
from specrag.models import (
    ContextBudget,
    ModelContextProfile,
    NativeFunctionCall,
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
)
from .llm_budget import build_token_budget_manager
from .llm_contracts import (
    LLMApiMode,
    LLMCallReceipt,
    NativeToolLLMResponse,
    OutputT,
    StructuredLLMClient,
    StructuredLLMResponse,
)

from .api_errors import (
    ProviderFailure,
    StructuredLLMError,
    classify_provider_error,
    is_retryable_provider_error,
)


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def _responses_tool_payload(definition: NativeFunctionDefinition) -> dict[str, Any]:
    return {
        "type": "function", "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
        "strict": definition.strict,
    }


def _responses_output_payload(output: NativeFunctionCallOutput) -> dict[str, Any]:
    return {
        "type": "function_call_output", "call_id": output.call_id,
        "output": _canonical_json(output.output),
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_error(
    exc: Exception,
    *,
    attempt_count: int | None = None,
    exhausted_models: int | None = None,
) -> StructuredLLMError:
    """Do not persist provider bodies, headers, request objects, or prompts."""

    failure = classify_provider_error(exc)
    details = [type(exc).__name__]
    request_id = getattr(exc, "request_id", None)
    for label, value in (
        ("status", getattr(exc, "status_code", None)),
        ("code", getattr(exc, "code", None)),
        ("param", getattr(exc, "param", None)),
    ):
        if isinstance(value, (str, int)) and str(value):
            details.append(f"{label}={str(value)[:80]}")
    if isinstance(request_id, str) and request_id:
        details.append(f"request_id_sha256={_sha256(request_id)[:16]}")
    if attempt_count is not None:
        details.append(f"attempts={attempt_count}")
    if exhausted_models is not None:
        details.append(f"models={exhausted_models}")
    return StructuredLLMError(
        "structured LLM request failed (" + ", ".join(details) + ")",
        failure_kind=failure.kind,
        retryable=failure.retryable,
        status_code=failure.status_code,
        attempt_count=attempt_count,
    )


def _is_retryable_provider_error(exc: Exception) -> bool:
    return is_retryable_provider_error(exc)


class OpenAIStructuredClient:
    """Official OpenAI SDK adapter with an explicit, non-automatic API mode."""

    __slots__ = (
        "_api_mode",
        "_client",
        "_budget_manager",
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
        budget_manager: TokenBudgetManager | None = None,
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
        self._budget_manager = budget_manager or TokenBudgetManager(
            ModelProfileResolver(ModelContextProfile(
                model_id="default",
                context_window_tokens=128_000,
                max_input_tokens=128_000,
                expected_output_tokens=max_output_tokens,
            ))
        )
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
            budget_manager=build_token_budget_manager(config),
        )

    def _preflight(
        self, model_id: str, components: Mapping[str, Any],
    ) -> ContextBudget:
        budget = self._budget_manager.inspect(
            model_id,
            components,
            expected_output_tokens=self._max_output_tokens,
        )
        if not budget.fits:
            raise ContextBudgetExceededError(budget)
        return budget

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
        selected_budget: ContextBudget | None = None
        models = (self._model, *self._fallback_models)
        for model_index, candidate_model in enumerate(models):
            try:
                candidate_budget = self._preflight(candidate_model, {
                    "instructions": instructions,
                    "input": input_json,
                    "output_schema": schema_json,
                })
            except ContextBudgetExceededError:
                if model_index + 1 < len(models):
                    continue
                raise
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
                                "structured chat completion was refused",
                                failure_kind="model_refusal",
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
                    selected_budget = candidate_budget
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
            estimated_input_tokens=(
                selected_budget.estimated_input_tokens if selected_budget else None
            ),
            input_ceiling_tokens=(
                selected_budget.input_ceiling_tokens if selected_budget else None
            ),
        )
        return StructuredLLMResponse(output=output, receipt=receipt)

    def call_tools(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: BaseModel | Mapping[str, Any],
        tools: Sequence[NativeFunctionDefinition],
        prior_response_items: Sequence[Mapping[str, Any]] = (),
        prior_outputs: Sequence[NativeFunctionCallOutput] = (),
        max_tool_calls: int | None = None,
    ) -> NativeToolLLMResponse:
        """Request strict native calls and retain only bounded replay state."""

        if self._api_mode != LLMApiMode.RESPONSES:
            raise ValueError("native Researcher tools require Responses API mode")
        for name, value in (
            ("purpose", purpose),
            ("prompt_version", prompt_version),
            ("instructions", instructions),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-blank")
        if not tools:
            raise ValueError("at least one native function definition is required")
        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive when supplied")

        input_json = _canonical_json(input_payload)
        input_items: list[Mapping[str, Any]] = [
            dict(item) for item in prior_response_items
        ]
        input_items.extend(_responses_output_payload(output) for output in prior_outputs)
        input_items.append({"role": "user", "content": input_json})
        tool_payloads = [_responses_tool_payload(tool) for tool in tools]
        started = time.perf_counter()
        attempted_models: list[str] = []
        raw: Any = None
        calls: tuple[NativeFunctionCall, ...] | None = None
        selected_model = self._model
        input_tokens: int | None = None
        output_tokens: int | None = None
        models = (self._model, *self._fallback_models)
        selected_budget: ContextBudget | None = None

        for model_index, candidate_model in enumerate(models):
            try:
                candidate_budget = self._preflight(candidate_model, {
                    "instructions": instructions,
                    "input": input_items,
                    "tools": tool_payloads,
                })
            except ContextBudgetExceededError:
                if model_index + 1 < len(models):
                    continue
                raise
            for retry_index in range(self._max_retries + 1):
                attempted_models.append(candidate_model)
                try:
                    kwargs: dict[str, Any] = {
                        "model": candidate_model,
                        "instructions": instructions,
                        "input": input_items,
                        "tools": tool_payloads,
                        "tool_choice": "required",
                        "parallel_tool_calls": True,
                        "reasoning": {"effort": self._reasoning_effort},
                        "max_output_tokens": self._max_output_tokens,
                        "store": False,
                    }
                    if max_tool_calls is not None:
                        kwargs["max_tool_calls"] = max_tool_calls
                    raw = self._client.responses.create(**kwargs)
                    status = getattr(raw, "status", "completed")
                    if status != "completed":
                        raise StructuredLLMError(
                            "native tool response was not completed "
                            f"(status={status})"
                        )
                    parsed_calls: list[NativeFunctionCall] = []
                    for item in raw.output:
                        if getattr(item, "type", "") != "function_call":
                            continue
                        try:
                            arguments = json.loads(item.arguments)
                        except (TypeError, json.JSONDecodeError) as exc:
                            raise StructuredLLMError(
                                "native function call contained invalid JSON"
                            ) from exc
                        if not isinstance(arguments, dict):
                            raise StructuredLLMError(
                                "native function arguments must be an object"
                            )
                        parsed_calls.append(NativeFunctionCall(
                            call_id=item.call_id,
                            name=item.name,
                            arguments=arguments,
                        ))
                    if not parsed_calls:
                        raise StructuredLLMError(
                            "native tool response contained no function calls"
                        )
                    calls = tuple(parsed_calls)
                    usage = getattr(raw, "usage", None)
                    input_tokens = getattr(usage, "input_tokens", None)
                    output_tokens = getattr(usage, "output_tokens", None)
                    selected_model = candidate_model
                    selected_budget = candidate_budget
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
            if calls is not None:
                break
        if calls is None or raw is None:  # pragma: no cover - exhaustive loop
            raise StructuredLLMError("native tool retry loop produced no result")

        request_sha = _sha256(_canonical_json({
            "model": selected_model,
            "api_mode": self._api_mode.value,
            "purpose": purpose,
            "prompt_version": prompt_version,
            "instructions": instructions,
            "input": input_items,
            "tools": tool_payloads,
            "max_tool_calls": max_tool_calls,
        }))
        response_json = _canonical_json([call.model_dump(mode="json") for call in calls])
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
            estimated_input_tokens=(
                selected_budget.estimated_input_tokens if selected_budget else None
            ),
            input_ceiling_tokens=(
                selected_budget.input_ceiling_tokens if selected_budget else None
            ),
        )
        replay_items = tuple({
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": _canonical_json(call.arguments),
        } for call in calls)
        return NativeToolLLMResponse(
            calls=calls,
            replay_items=replay_items,
            receipt=receipt,
        )


__all__ = [
    "LLMApiMode",
    "LLMCallReceipt",
    "NativeToolLLMResponse",
    "OpenAIStructuredClient",
    "ProviderFailure",
    "StructuredLLMClient",
    "StructuredLLMError",
    "StructuredLLMResponse",
    "build_token_budget_manager",
    "classify_provider_error",
]
