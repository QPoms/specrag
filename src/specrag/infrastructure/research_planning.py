"""Structured question routing and bounded research-task decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.models import (
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
)
from specrag.prompts import ROUTER_INSTRUCTIONS, ROUTER_PROMPT_VERSION

from .structured_agent import (
    CodexAgentReceipt,
    CodexStructuredAgent,
)


class RouteGoalAssessment(BaseModel):
    """Provider-facing assessment used only to derive the public route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    independent_goal_count: int = Field(ge=1, le=4)
    goal_relationship: Literal["single_chain", "requires_synthesis"]
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _coherent_relationship(self) -> "RouteGoalAssessment":
        expected = (
            "single_chain"
            if self.independent_goal_count == 1
            else "requires_synthesis"
        )
        if self.goal_relationship != expected:
            raise ValueError(
                "one independent goal requires single_chain; two or more "
                "require requires_synthesis"
            )
        return self

    def to_decision(self) -> ResearchRouteDecision:
        complexity = (
            "simple" if self.independent_goal_count == 1 else "complex"
        )
        return ResearchRouteDecision(
            route="single_task" if complexity == "simple" else "decompose",
            complexity=complexity,
            rationale=self.rationale,
        )


@dataclass(frozen=True)
class StructuredRouteResponse:
    """Public decision plus an artifact-only internal assessment."""

    output: ResearchRouteDecision
    assessment: RouteGoalAssessment
    receipt: CodexAgentReceipt


@dataclass(frozen=True)
class StructuredQuestionRouter:
    agent: CodexStructuredAgent

    def route_with_receipt(
        self,
        question: str,
    ) -> StructuredRouteResponse:
        response = self.agent.parse(
            purpose="assess independent goals in one 3GPP question",
            prompt_version=ROUTER_PROMPT_VERSION,
            instructions=ROUTER_INSTRUCTIONS,
            input_payload={"question": question},
            output_type=RouteGoalAssessment,
        )
        assessment = response.output
        return StructuredRouteResponse(
            output=assessment.to_decision(),
            assessment=assessment,
            receipt=response.receipt,
        )

    def route(self, question: str) -> ResearchRouteDecision:
        return self.route_with_receipt(question).output


@dataclass(frozen=True)
class StructuredResearchPlanner:
    agent: CodexStructuredAgent
    max_complex_tasks: int = 4

    def plan(
        self,
        *,
        research_id: str,
        question: str,
        route: ResearchRouteDecision,
    ) -> ResearchPlan:
        if route.complexity == "simple":
            return ResearchPlan(
                plan_id=f"P:{research_id}",
                tasks=(ResearchTask(
                    task_id="T1",
                    question=question,
                    objective=(
                        "Locate the normative passage that completely answers "
                        "the question."
                    ),
                ),),
            )

        response = self.agent.parse(
            purpose="decompose one complex 3GPP research question",
            prompt_version="research-planner/v1.0",
            instructions=(
                "Create 2 to 4 non-overlapping ResearchTask items that together "
                "cover the complete user question. Use IDs T1, T2, and so on; "
                "use status pending. Each task must remain a retrieval objective, "
                "not an asserted answer. Independent source lookups have no "
                "dependencies and may run in parallel. Add dependencies only "
                "when a later relation cannot be researched until an earlier "
                "result is known. Include an explicit integration/relationship "
                "task when merely collecting separate facts would not answer the "
                "question. Do not invent requirements absent from the question."
            ),
            input_payload={
                "research_id": research_id,
                "question": question,
                "routing": route.model_dump(mode="json"),
            },
            output_type=ResearchPlan,
        )
        proposed = response.output
        if not 2 <= len(proposed.tasks) <= self.max_complex_tasks:
            raise ValueError(
                "complex research plans must contain 2 to "
                f"{self.max_complex_tasks} tasks"
            )
        if any(task.status != "pending" for task in proposed.tasks):
            raise ValueError("new research-plan tasks must be pending")
        return proposed.model_copy(update={"plan_id": f"P:{research_id}"})


__all__ = [
    "RouteGoalAssessment",
    "StructuredQuestionRouter",
    "StructuredResearchPlanner",
    "StructuredRouteResponse",
]
