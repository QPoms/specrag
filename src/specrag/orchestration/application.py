"""Product assembly for the real Release-scoped research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from specrag.agents.researcher import RegistryResearcherService
from specrag.config import AppConfig, REPO_ROOT, load_config
from specrag.evidence import (
    EvidenceConflictService,
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    EvidenceValidationService,
    SourceEvidenceValidator,
)
from specrag.infrastructure.research_assessors import (
    StructuredEvidenceConflictAssessor,
    StructuredEvidenceSupportAssessor,
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.infrastructure.research_planning import (
    StructuredPlanResponse,
    StructuredRouteResponse,
    StructuredQuestionRouter,
    StructuredResearchPlanner,
)
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.models import (
    ResearchBudget,
    ResearchPlan,
    ResearchRouteDecision,
)
from .adapters import (
    EvidenceSufficiencyHandler,
    EvidenceSynthesisHandler,
    EvidenceValidationHandler,
    SingleResearcherHandler,
)
from .state import ResearchState, new_research_state
from .workflow import (
    ResearchNodeHandlers,
    SingleResearchWorkflow,
    build_single_research_workflow,
)
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.tools import build_retrieval_tool_registry


@dataclass(frozen=True)
class QuestionRouterHandler:
    """Classify question complexity before planning or retrieval."""

    service: StructuredQuestionRouter

    def __call__(self, state: ResearchState) -> dict[str, object]:
        decision = self.service.route(state["user_question"])
        return {"route": "plan", "route_decision": decision}


@dataclass(frozen=True)
class ResearchPlannerHandler:
    """Create one simple task or a bounded complex-task decomposition."""

    service: StructuredResearchPlanner

    def __call__(self, state: ResearchState) -> dict[str, object]:
        decision = state["route_decision"]
        if decision is None:
            raise ValueError("research planning requires a routing decision")
        plan = self.service.plan(
            research_id=state["research_id"],
            question=state["user_question"],
            route=decision,
        )
        return {"research_plan": plan, "research_tasks": plan.tasks}


@dataclass(frozen=True)
class GapReplanner:
    """Continue incomplete tasks while exposing explicit gaps to Researcher."""

    def __call__(self, state: ResearchState) -> dict[str, object]:
        unresolved = tuple(gap for gap in state["gaps"] if not gap.resolved)
        if not unresolved:
            return {"stop_reason": "insufficient_evidence"}
        tasks = tuple(
            task.model_copy(update={"status": "running"})
            if task.status != "completed"
            else task
            for task in state["research_tasks"]
        )
        return {"research_tasks": tasks}


@dataclass(frozen=True)
class ResearchApplication:
    """One reusable, fully assembled Single Researcher application."""

    workflow: SingleResearchWorkflow
    release: int

    def run(
        self,
        *,
        research_id: str,
        question: str,
        thread_id: str,
        budget: ResearchBudget | None = None,
    ) -> ResearchState:
        state = new_research_state(
            research_id=research_id,
            user_question=question,
            budget=budget,
        )
        return self.workflow.invoke(state, thread_id=thread_id)


@dataclass(frozen=True)
class RoutingApplication:
    """Run only difficulty routing without planning or retrieval."""

    router: StructuredQuestionRouter

    def run(
        self,
        *,
        question: str,
    ) -> StructuredRouteResponse:
        return self.router.route_with_receipt(question)


@dataclass(frozen=True)
class PlanningApplication:
    """Run only question routing and task planning, without corpus retrieval."""

    router: StructuredQuestionRouter
    planner: StructuredResearchPlanner

    def run(
        self,
        *,
        research_id: str,
        question: str,
    ) -> tuple[ResearchRouteDecision, ResearchPlan]:
        route, response = self.run_with_receipt(
            research_id=research_id,
            question=question,
        )
        return route, response.output

    def run_with_receipt(
        self,
        *,
        research_id: str,
        question: str,
    ) -> tuple[ResearchRouteDecision, StructuredPlanResponse]:
        """Run live routing and retain Planner execution metadata."""

        route = self.router.route(question)
        return self.run_with_route_receipt(
            research_id=research_id,
            question=question,
            route=route,
        )

    def run_with_route(
        self,
        *,
        research_id: str,
        question: str,
        route: ResearchRouteDecision,
    ) -> tuple[ResearchRouteDecision, ResearchPlan]:
        """Plan from a fixed reviewed route without invoking Router again."""

        route, response = self.run_with_route_receipt(
            research_id=research_id,
            question=question,
            route=route,
        )
        return route, response.output

    def run_with_route_receipt(
        self,
        *,
        research_id: str,
        question: str,
        route: ResearchRouteDecision,
    ) -> tuple[ResearchRouteDecision, StructuredPlanResponse]:
        """Plan from a fixed route and retain Planner execution metadata."""

        response = self.planner.plan_with_receipt(
            research_id=research_id,
            question=question,
            route=route,
        )
        return route, response


def build_planning_application(
    config: AppConfig | None = None,
) -> PlanningApplication:
    """Build the first pipeline stage without loading a retrieval index."""

    app_config = config or load_config()
    planning_agent = LLMApiStructuredAgent(
        role="planning",
        reasoning_effort="high",
        config=app_config,
    )
    return PlanningApplication(
        router=StructuredQuestionRouter(planning_agent),
        planner=StructuredResearchPlanner(planning_agent),
    )


def build_routing_application(
    config: AppConfig | None = None,
) -> RoutingApplication:
    """Build Router alone; no Release scope or corpus is loaded."""

    app_config = config or load_config()
    planning_agent = LLMApiStructuredAgent(
        role="planning",
        reasoning_effort="high",
        config=app_config,
    )
    return RoutingApplication(router=StructuredQuestionRouter(planning_agent))


def build_research_application(
    config: AppConfig | None = None,
    *,
    release: int = 17,
    repository_root: Path = REPO_ROOT,
) -> ResearchApplication:
    """Wire existing retrieval, agent, evidence, and graph components."""

    if release < 1:
        raise ValueError("retrieval Release must be positive")
    app_config = config or load_config()
    retrieval_config = app_config.retrieval.model_copy(update={
        "version_policy": "highest_version_within_release",
        "target_release": release,
    })
    app_config = app_config.model_copy(update={"retrieval": retrieval_config})

    retriever = HybridMarkdownRetriever(app_config.retrieval)
    tool_registry = build_retrieval_tool_registry(retriever)
    planning_agent = LLMApiStructuredAgent(
        role="planning",
        reasoning_effort="high",
        config=app_config,
    )
    summary_agent = LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=app_config,
    )
    router = StructuredQuestionRouter(planning_agent)
    planner = StructuredResearchPlanner(planning_agent)
    researcher_service = RegistryResearcherService(
        controller_agent=planning_agent,
        planner_agent=planning_agent,
        retriever=retriever,
        app_config=app_config,
        tool_registry=tool_registry,
        max_rounds=3,
        max_search_actions=4,
        max_expansion_actions=2,
        hits_per_search=5,
        max_evidence_per_question=12,
    )

    validation = EvidenceValidationService(
        SourceEvidenceValidator(repository_root),
        StructuredEvidenceSupportAssessor(summary_agent),
    )
    conflicts = EvidenceConflictService(
        StructuredEvidenceConflictAssessor(summary_agent)
    )
    sufficiency = EvidenceSufficiencyService(
        StructuredTaskSufficiencyAssessor(summary_agent)
    )
    synthesis = EvidenceSynthesisService(
        StructuredEvidenceSynthesisGenerator(summary_agent)
    )
    handlers = ResearchNodeHandlers(
        router=QuestionRouterHandler(router),
        planner=ResearchPlannerHandler(planner),
        researcher=SingleResearcherHandler(researcher_service),
        validator=EvidenceValidationHandler(validation, conflicts),
        sufficiency_judge=EvidenceSufficiencyHandler(sufficiency),
        replanner=GapReplanner(),
        synthesizer=EvidenceSynthesisHandler(synthesis),
    )
    return ResearchApplication(
        workflow=build_single_research_workflow(handlers),
        release=release,
    )


__all__ = [
    "GapReplanner",
    "PlanningApplication",
    "QuestionRouterHandler",
    "ResearchApplication",
    "ResearchPlannerHandler",
    "RoutingApplication",
    "build_planning_application",
    "build_research_application",
    "build_routing_application",
]
