"""Product assembly for the real Release-scoped research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from specrag.agents.models import ResearcherActionPlan
from specrag.agents.planning import ResearcherPlanningService
from specrag.agents.researcher import RegistryResearcherService
from specrag.config import AppConfig, REPO_ROOT, load_config
from specrag.evidence import (
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    SourceEvidenceValidator,
    SourceValidationService,
)
from specrag.infrastructure.research_assessors import (
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.infrastructure.research_planning import (
    StructuredPlanResponse,
    StructuredRouteResponse,
    StructuredQuestionRouter,
    StructuredResearchPlanner,
)
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
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
    SourceValidationHandler,
    SingleResearcherHandler,
)
from .state import ResearchState, new_research_state
from .workflow import (
    ResearchNodeHandlers,
    SingleResearchWorkflow,
    build_single_research_workflow,
)
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.corpus import (
    corpus_scope_from_config,
    discover_source_documents,
)
from specrag.tools import (
    build_retrieval_tool_registry,
    retrieval_tool_definitions,
)


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
        if state["research_plan"] is not None:
            plan = state["research_plan"]
            return {"research_plan": plan, "research_tasks": plan.tasks}
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
        gap_task_ids = {gap.task_id for gap in unresolved}
        tasks = tuple(
            task.model_copy(update={"status": "running"})
            if task.task_id in gap_task_ids and task.status != "completed"
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
        plan: ResearchPlan | None = None,
    ) -> ResearchState:
        state = new_research_state(
            research_id=research_id,
            user_question=question,
            budget=budget,
        )
        if plan is not None:
            state["research_plan"] = plan
            state["research_tasks"] = plan.tasks
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


@dataclass(frozen=True)
class ResearcherPlanningResponse:
    """Planner output plus unexecuted first-turn Researcher proposals."""

    planning: StructuredPlanResponse
    first_turns: tuple[tuple[str, ResearcherActionPlan], ...]
    deferred_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResearcherPlanningApplication:
    """Generate inspectable Researcher actions without executing retrieval."""

    planner: StructuredResearchPlanner
    researcher: ResearcherPlanningService
    release: int

    def run_with_route(
        self,
        *,
        research_id: str,
        question: str,
        route: ResearchRouteDecision,
    ) -> ResearcherPlanningResponse:
        planning = self.planner.plan_with_receipt(
            research_id=research_id,
            question=question,
            route=route,
        )
        ready = tuple(
            task.model_copy(update={"status": "running"})
            for task in planning.output.tasks
            if not task.dependencies
        )
        deferred = tuple(
            task.task_id
            for task in planning.output.tasks
            if task.dependencies
        )
        first_turns = tuple(
            (
                task.task_id,
                self.researcher.propose_first_turn(
                    (task,),
                    shared_context={
                        "research_id": research_id,
                        "iteration": 0,
                        "unresolved_gaps": (),
                        "dependency_context": (),
                    },
                ),
            )
            for task in ready
        )
        return ResearcherPlanningResponse(
            planning=planning,
            first_turns=first_turns,
            deferred_task_ids=deferred,
        )


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


def build_researcher_planning_application(
    config: AppConfig | None = None,
    *,
    release: int = 17,
) -> ResearcherPlanningApplication:
    """Build planning-only Researcher inspection without retrieval indexes."""

    if release < 1:
        raise ValueError("retrieval Release must be positive")
    app_config = config or load_config()
    retrieval_config = app_config.retrieval.model_copy(update={
        "version_policy": "highest_version_within_release",
        "target_release": release,
    })
    app_config = app_config.model_copy(update={"retrieval": retrieval_config})
    corpus_scope = corpus_scope_from_config(app_config.retrieval)
    documents = discover_source_documents(app_config.retrieval, corpus_scope)
    available_specifications = tuple(sorted({
        document.specification
        for document in documents
        if document.specification is not None
    }))
    planning_agent = LLMApiStructuredAgent(
        role="planning",
        reasoning_effort="high",
        config=app_config,
    )
    return ResearcherPlanningApplication(
        planner=StructuredResearchPlanner(planning_agent),
        researcher=ResearcherPlanningService(
            controller_agent=StructuredResearcherController(planning_agent),
            corpus_scope=corpus_scope,
            available_specifications=available_specifications,
            available_tools=retrieval_tool_definitions(),
            max_rounds=3,
            max_total_actions=6,
            hits_per_search=5,
            max_evidence_per_question=12,
        ),
        release=release,
    )


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
        controller_agent=StructuredResearcherController(planning_agent),
        retriever=retriever,
        app_config=app_config,
        tool_registry=tool_registry,
        max_rounds=3,
        max_total_actions=6,
        hits_per_search=5,
        max_evidence_per_question=12,
    )

    validation = SourceValidationService(
        SourceEvidenceValidator(repository_root),
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
        source_validator=SourceValidationHandler(validation),
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
    "ResearcherPlanningApplication",
    "ResearcherPlanningResponse",
    "ResearchPlannerHandler",
    "RoutingApplication",
    "build_planning_application",
    "build_research_application",
    "build_researcher_planning_application",
    "build_routing_application",
]
