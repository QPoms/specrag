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
from specrag.prompts import (
    PLANNER_INSTRUCTIONS,
    PLANNER_PROMPT_VERSION,
    ROUTER_INSTRUCTIONS,
    ROUTER_PROMPT_VERSION,
)

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
class StructuredPlanResponse:
    """Validated plan plus the execution metadata used to produce it."""

    output: ResearchPlan
    mode: Literal["deterministic", "model"]
    contract_version: str
    prompt_version: str | None
    receipt: CodexAgentReceipt | None


@dataclass(frozen=True)
class StructuredResearchPlanner:
    agent: CodexStructuredAgent
    max_complex_tasks: int = 4

    def plan_with_receipt(
        self,
        *,
        research_id: str,
        question: str,
        route: ResearchRouteDecision,
    ) -> StructuredPlanResponse:
        if route.complexity == "simple":
            return StructuredPlanResponse(
                output=ResearchPlan(
                    plan_id=f"P:{research_id}",
                    tasks=(ResearchTask(
                        task_id="T1",
                        question=question,
                        objective=(
                            "Locate the normative passage that completely answers "
                            "the question."
                        ),
                    ),),
                ),
                mode="deterministic",
                contract_version=PLANNER_PROMPT_VERSION,
                prompt_version=None,
                receipt=None,
            )

        response = self.agent.parse(
            purpose="decompose one complex 3GPP research question",
            prompt_version=PLANNER_PROMPT_VERSION,
            instructions=PLANNER_INSTRUCTIONS,
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
        expected_ids = tuple(
            f"T{index}" for index in range(1, len(proposed.tasks) + 1)
        )
        actual_ids = tuple(task.task_id for task in proposed.tasks)
        if actual_ids != expected_ids:
            raise ValueError(
                "complex research-plan task IDs must be ordered as "
                + ", ".join(expected_ids)
            )
        return StructuredPlanResponse(
            output=proposed.model_copy(update={"plan_id": f"P:{research_id}"}),
            mode="model",
            contract_version=PLANNER_PROMPT_VERSION,
            prompt_version=PLANNER_PROMPT_VERSION,
            receipt=response.receipt,
        )

    def plan(
        self,
        *,
        research_id: str,
        question: str,
        route: ResearchRouteDecision,
    ) -> ResearchPlan:
        """Return only the plan for runtime callers that do not persist audits."""

        return self.plan_with_receipt(
            research_id=research_id,
            question=question,
            route=route,
        ).output
__all__ = [
    "RouteGoalAssessment",
    "StructuredQuestionRouter",
    "StructuredPlanResponse",
    "StructuredResearchPlanner",
    "StructuredRouteResponse",
]
