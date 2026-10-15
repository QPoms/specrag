"""Structured model adapter for bounded Researcher decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from specrag.prompts import (
    RESEARCHER_CONTROLLER_INSTRUCTIONS,
    RESEARCHER_CONTROLLER_PROMPT_VERSION,
    RESEARCHER_GAP_INSTRUCTIONS,
    RESEARCHER_GAP_PROMPT_VERSION,
)
from specrag.models import NativeFunctionCallOutput, NativeFunctionDefinition

from .structured_agent import CodexAgentToolResponse


class NativeToolAgent(Protocol):
    def call_tools(
        self,
        *,
        purpose: str,
        prompt_version: str,
        instructions: str,
        input_payload: Mapping[str, Any],
        tools: Sequence[NativeFunctionDefinition],
        prior_response_items: Sequence[Mapping[str, Any]],
        prior_outputs: Sequence[NativeFunctionCallOutput],
        max_tool_calls: int | None,
    ) -> CodexAgentToolResponse: ...


@dataclass(frozen=True)
class StructuredResearcherController:
    """Bind the Researcher prompt contract to a structured model adapter."""

    agent: NativeToolAgent

    def decide(
        self,
        *,
        input_payload: dict[str, Any],
        tools: Sequence[NativeFunctionDefinition],
        prompt_variant: Literal["initial", "evidence_gap"] = "initial",
        prior_response_items: Sequence[Mapping[str, Any]] = (),
        prior_outputs: Sequence[NativeFunctionCallOutput] = (),
        max_tool_calls: int | None = None,
    ) -> CodexAgentToolResponse:
        if prompt_variant == "evidence_gap":
            prompt_version = RESEARCHER_GAP_PROMPT_VERSION
            instructions = RESEARCHER_GAP_INSTRUCTIONS
        else:
            prompt_version = RESEARCHER_CONTROLLER_PROMPT_VERSION
            instructions = RESEARCHER_CONTROLLER_INSTRUCTIONS
        return self.agent.call_tools(
            purpose="review evidence and choose bounded retrieval tool calls",
            prompt_version=prompt_version,
            instructions=instructions,
            input_payload=input_payload,
            tools=tools,
            prior_response_items=prior_response_items,
            prior_outputs=prior_outputs,
            max_tool_calls=max_tool_calls,
        )


__all__ = ["StructuredResearcherController"]
