"""Structured model adapter for bounded Researcher decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from specrag.prompts import (
    RESEARCHER_CONTROLLER_INSTRUCTIONS,
    RESEARCHER_CONTROLLER_PROMPT_VERSION,
)

from .structured_agent import CodexAgentResponse, CodexStructuredAgent


OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredResearcherController:
    """Bind the Researcher prompt contract to a structured model adapter."""

    agent: CodexStructuredAgent

    def decide(
        self,
        *,
        input_payload: dict[str, Any],
        output_type: type[OutputT],
    ) -> CodexAgentResponse[OutputT]:
        return self.agent.parse(
            purpose="plan and choose the next bounded retrieval tool calls",
            prompt_version=RESEARCHER_CONTROLLER_PROMPT_VERSION,
            instructions=RESEARCHER_CONTROLLER_INSTRUCTIONS,
            input_payload=input_payload,
            output_type=output_type,
        )


__all__ = ["StructuredResearcherController"]
