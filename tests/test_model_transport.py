from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from specrag.infrastructure.llm import LLMApiMode, OpenAIStructuredClient
from specrag.infrastructure.model_transport import (
    ModelTransportProfile,
    resolve_transport_profile,
    transport_reasoning_options,
)


class _Output(BaseModel):
    status: str


class _ChatCompletions:
    def __init__(self) -> None:
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            model="deepseek-v4-pro",
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(refusal=None, parsed={"status": "ready"}),
            )],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
            id="completion-1",
        )


def test_chat_structured_profile_can_explicitly_omit_reasoning() -> None:
    profile = ModelTransportProfile(
        model_id="deepseek-v4-pro",
        structured_api_mode=LLMApiMode.CHAT_COMPLETIONS,
        reasoning_effort=None,
        supports_native_tools=False,
    )

    assert profile.model_dump(mode="json") == {
        "model_id": "deepseek-v4-pro",
        "structured_api_mode": "chat_completions",
        "reasoning_effort": None,
        "supports_native_tools": False,
    }
    assert transport_reasoning_options(
        profile.reasoning_effort,
        api_mode=profile.structured_api_mode,
    ) == {}


def test_client_omits_reasoning_field_for_provider_default_chat_profile() -> None:
    completions = _ChatCompletions()
    client = OpenAIStructuredClient(
        api_key="test",
        model="deepseek-v4-pro",
        api_mode=LLMApiMode.CHAT_COMPLETIONS,
        reasoning_effort=None,
        max_retries=0,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    response = client.parse(
        purpose="judge",
        prompt_version="judge/v1",
        instructions="Return status.",
        input_payload={"probe": True},
        output_type=_Output,
    )

    assert response.output.status == "ready"
    assert response.receipt.api_mode == LLMApiMode.CHAT_COMPLETIONS
    assert "reasoning_effort" not in completions.request


def test_native_tools_cannot_be_claimed_for_chat_transport() -> None:
    with pytest.raises(ValidationError, match="native tools require"):
        ModelTransportProfile(
            model_id="deepseek-v4.1-flash",
            structured_api_mode=LLMApiMode.CHAT_COMPLETIONS,
            reasoning_effort=None,
            supports_native_tools=True,
        )


def test_transport_profile_is_explicit_and_model_bound() -> None:
    profile = ModelTransportProfile(
        model_id="deepseek-v4-pro",
        structured_api_mode=LLMApiMode.CHAT_COMPLETIONS,
        reasoning_effort=None,
        supports_native_tools=False,
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_transport_profile(
            model_id="gpt-6-astra",
            api_mode=LLMApiMode.RESPONSES,
            reasoning_effort="high",
            explicit=profile,
        )
