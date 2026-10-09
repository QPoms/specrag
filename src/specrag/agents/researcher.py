"""Bounded agent-controlled retrieval over the existing local toolchain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from specrag.config import AppConfig
from specrag.retrieval.hypothesis import (
    merge_parallel_retrieval_channels,
)
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from specrag.retrieval.planner import (
    DiscoveryRetrievalPlanV3,
    PlannedDiscoveryRetrieval,
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    SpecificationRoute,
    execute_planned_discovery_retrieval,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


AgentToolName = Literal["hybrid_search", "expand_same_section", "finish"]


class AgentRetrievalAction(_StrictModel):
    """One requested tool call; unused fields must be empty values."""

    action_id: str = Field(min_length=1)
    tool: AgentToolName
    query_id: str
    lexical_query: str
    semantic_query: str
    target_specifications: tuple[str, ...] = Field(max_length=6)
    seed_evidence_ids: tuple[str, ...] = Field(max_length=4)
    max_results: int = Field(ge=1, le=12)
    rationale: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)


class AgentRetrievalDecision(_StrictModel):
    assessment: str = Field(min_length=1)
    actions: tuple[AgentRetrievalAction, ...] = Field(
        min_length=1,
        max_length=3,
    )


class AgentActionOutcome(_StrictModel):
    round_number: int = Field(ge=1)
    action: AgentRetrievalAction
    status: Literal["completed", "no_hits", "rejected", "unavailable"]
    evidence_ids: tuple[str, ...] = ()
    new_evidence_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class AgentRetrievalTurn(_StrictModel):
    round_number: int = Field(ge=1)
    assessment: str = Field(min_length=1)
    receipt: CodexAgentReceipt
    outcomes: tuple[AgentActionOutcome, ...]


class AgenticRetrievalExecution(_StrictModel):
    schema_version: Literal["agentic-retrieval-execution/v0.1"] = (
        "agentic-retrieval-execution/v0.1"
    )
    status: Literal["completed", "partial", "unavailable"]
    stop_reason: Literal[
        "agent_finish",
        "round_budget",
        "action_budget",
        "agent_error",
    ]
    rounds_used: int = Field(ge=0)
    search_actions_used: int = Field(ge=0)
    expansion_actions_used: int = Field(ge=0)
    turns: tuple[AgentRetrievalTurn, ...] = ()
    evidence_execution: PlannedRetrievalExecution
    errors: tuple[str, ...] = ()


_CONTROLLER_INSTRUCTIONS = """\
You control a bounded source-retrieval loop. Your only goal is to collect enough
normative evidence to answer the supplied Discovery questions. You may call:

1. hybrid_search: lexical plus semantic search. Supply a query_id, both query
   strings, optional target specifications, no seed IDs, and a result limit.
2. expand_same_section: retrieve the opening/adjacent chunks of the exact
   subsection containing one to four seed evidence IDs. Supply a query_id and
   seed IDs; leave query strings and target specifications empty.
3. finish: stop retrieval. Use an empty query_id, empty query strings, no target
   specifications, no seed IDs, and max_results=1.

Inspect the evidence before choosing the next action. Search for missing facts
or missing relation edges, not paraphrases of evidence already present. Ordered
questions may form a semantic chain: terms found for an earlier question may be
used to formulate a later search, but they are search hints only. A later answer
still requires its own source evidence. Do not assume a candidate answer is true.
Do not repeat a prior tool call. Use only listed specification numbers. Finish
when the requested answer units are directly supportable or the remaining
budget cannot usefully improve them. Return one to three actions per round; a
finish action must be the only action.
"""


def _minimum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


class _EvidenceLedger:
    def __init__(
        self,
        query_ids: tuple[str, ...],
        *,
        max_per_question: int,
    ) -> None:
        self._hits = {query_id: [] for query_id in query_ids}
        self._indexes = {query_id: {} for query_id in query_ids}
        self._next = {query_id: 1 for query_id in query_ids}
        self._max_per_question = max_per_question

    def hits(self, query_id: str) -> tuple[PlannedRetrievalHit, ...]:
        return tuple(self._hits[query_id])

    def find(self, query_id: str, evidence_id: str) -> PlannedRetrievalHit | None:
        for hit in self._hits[query_id]:
            if hit.hit_id == evidence_id:
                return hit
        return None

    def add(
        self,
        query_id: str,
        hits: tuple[PlannedRetrievalHit, ...],
        *,
        action_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        observed: list[str] = []
        added: list[str] = []
        for hit in hits:
            existing_index = self._indexes[query_id].get(hit.chunk_id)
            source = f"agent_tool:{action_id}"
            if existing_index is not None:
                existing = self._hits[query_id][existing_index]
                merged = existing.model_copy(update={
                    "retrieval_sources": tuple(dict.fromkeys((
                        *existing.retrieval_sources,
                        *hit.retrieval_sources,
                        source,
                    ))),
                    "probe_ids": tuple(dict.fromkeys((
                        *existing.probe_ids,
                        *hit.probe_ids,
                    ))),
                    "matched_anchors": tuple(dict.fromkeys((
                        *existing.matched_anchors,
                        *hit.matched_anchors,
                    ))),
                    "reference_source_chunk_ids": tuple(dict.fromkeys((
                        *existing.reference_source_chunk_ids,
                        *hit.reference_source_chunk_ids,
                    ))),
                    "semantic_rank": _minimum_optional(
                        existing.semantic_rank,
                        hit.semantic_rank,
                    ),
                    "lexical_rank": _minimum_optional(
                        existing.lexical_rank,
                        hit.lexical_rank,
                    ),
                    "final_score": max(existing.final_score, hit.final_score),
                })
                self._hits[query_id][existing_index] = merged
                observed.append(existing.hit_id)
                continue
            if len(self._hits[query_id]) >= self._max_per_question:
                continue
            evidence_id = f"E-{query_id}:{self._next[query_id]}"
            self._next[query_id] += 1
            stored = hit.model_copy(update={
                "hit_id": evidence_id,
                "collection_rank": len(self._hits[query_id]) + 1,
                "retrieval_sources": tuple(dict.fromkeys((
                    *hit.retrieval_sources,
                    source,
                ))),
            })
            self._indexes[query_id][stored.chunk_id] = len(self._hits[query_id])
            self._hits[query_id].append(stored)
            observed.append(evidence_id)
            added.append(evidence_id)
        return tuple(observed), tuple(added)

    def payload(self, *, excerpt_characters: int = 1600) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for query_id, hits in self._hits.items():
            for hit in hits:
                excerpt = hit.excerpt
                if len(excerpt) > excerpt_characters:
                    excerpt = excerpt[:excerpt_characters].rstrip() + " ..."
                values.append({
                    "evidence_id": hit.hit_id,
                    "query_id": query_id,
                    "specification": hit.specification,
                    "version": hit.version,
                    "heading_path": list(hit.heading_path),
                    "line_start": hit.line_start,
                    "line_end": hit.line_end,
                    "retrieval_sources": list(hit.retrieval_sources),
                    "excerpt": excerpt,
                })
        return values


def _build_search_plan(
    base_item: PlannedDiscoveryRetrieval,
    action: AgentRetrievalAction,
    *,
    round_number: int,
) -> DiscoveryRetrievalPlanV3:
    """Build a one-probe search with exact-anchor recall explicitly disabled."""

    routes = tuple(
        SpecificationRoute(
            specification=specification,
            weight=1.0,
            reason=f"agent-selected target for {action.action_id}",
            source="agent_hypothesis",
        )
        for specification in action.target_specifications
    )
    probe = RetrievalProbe(
        probe_id=f"AG-R{round_number}-{action.action_id}",
        evidence_role=base_item.angle,
        lexical_query=action.lexical_query.strip(),
        semantic_query=action.semantic_query.strip(),
        routes=routes,
        # Empty anchors are intentional: this tool ranks only BM25/semantic
        # results and therefore cannot reintroduce Route-B exact-anchor bias.
        anchors=(),
        facets=(),
        stage=1,
    )
    item = base_item.model_copy(update={
        "retrieval_id": f"{base_item.query_id}:agent:r{round_number}:{action.action_id}",
        "planner_action": "planned",
        "anchors": (),
        "warnings": (*base_item.warnings, "agent-controlled hybrid search"),
        "reference_expansion": ReferenceExpansionPolicy(
            enabled=False,
            max_references=0,
        ),
        "retrieval_probes": (probe,),
    })
    return DiscoveryRetrievalPlanV3(
        anchor_extraction_method="deterministic_extractive",
        retrievals=(item,),
    )


def _validate_action(
    action: AgentRetrievalAction,
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    ledger: _EvidenceLedger,
) -> tuple[str, ...]:
    errors: list[str] = []
    if action.tool == "finish":
        if (
            action.query_id
            or action.lexical_query
            or action.semantic_query
            or action.target_specifications
            or action.seed_evidence_ids
            or action.max_results != 1
        ):
            errors.append("finish contains non-empty tool parameters")
        return tuple(errors)
    if action.query_id not in query_ids:
        errors.append(f"unknown query_id: {action.query_id}")
        return tuple(errors)
    unknown_specs = sorted(
        set(action.target_specifications) - available_specifications
    )
    if unknown_specs:
        errors.append("unknown specifications: " + ", ".join(unknown_specs))
    if action.tool == "hybrid_search":
        if not action.lexical_query.strip() or not action.semantic_query.strip():
            errors.append("hybrid_search requires both query strings")
        if action.seed_evidence_ids:
            errors.append("hybrid_search cannot contain seed evidence IDs")
    elif action.tool == "expand_same_section":
        if action.lexical_query or action.semantic_query:
            errors.append("expand_same_section cannot contain query strings")
        if action.target_specifications:
            errors.append("expand_same_section cannot contain target specifications")
        if not action.seed_evidence_ids:
            errors.append("expand_same_section requires seed evidence IDs")
        unknown_ids = [
            evidence_id
            for evidence_id in action.seed_evidence_ids
            if ledger.find(action.query_id, evidence_id) is None
        ]
        if unknown_ids:
            errors.append("unknown seed evidence IDs: " + ", ".join(unknown_ids))
    return tuple(errors)


def _action_signature(action: AgentRetrievalAction) -> tuple[Any, ...]:
    return (
        action.tool,
        action.query_id,
        action.lexical_query.casefold().strip(),
        action.semantic_query.casefold().strip(),
        tuple(sorted(action.target_specifications)),
        tuple(sorted(action.seed_evidence_ids)),
    )


def _expand_same_section(
    *,
    query_id: str,
    seeds: tuple[PlannedRetrievalHit, ...],
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    max_results: int,
) -> tuple[PlannedRetrievalHit, ...]:
    task = PlannedRetrievalTaskResult(
        retrieval_id=f"{query_id}:agent:adjacent",
        query_id=query_id,
        planner_action="planned",
        status="completed",
        hits=seeds,
    )
    literal = PlannedRetrievalExecution(
        status="completed",
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=app_config.retrieval.embedding_model,
        corpus_version_policy=app_config.retrieval.version_policy,
        corpus_release=app_config.retrieval.target_release,
        tasks=(task,),
    )
    expanded = merge_parallel_retrieval_channels(
        literal,
        None,
        retriever=retriever,
        route_a_quota=len(seeds),
        route_b_quota=0,
        max_hits_per_question=len(seeds) + max_results,
        adjacent_chunk_budget=max_results,
    )
    seed_chunks = {hit.chunk_id for hit in seeds}
    return tuple(
        hit for hit in expanded.tasks[0].hits
        if hit.chunk_id not in seed_chunks
    )


def _evidence_execution(
    *,
    base_plan: DiscoveryRetrievalPlanV3,
    ledger: _EvidenceLedger,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
) -> PlannedRetrievalExecution:
    tasks: list[PlannedRetrievalTaskResult] = []
    for item in base_plan.retrievals:
        hits = ledger.hits(item.query_id)
        tasks.append(PlannedRetrievalTaskResult(
            retrieval_id=item.retrieval_id,
            query_id=item.query_id,
            planner_action=item.planner_action,
            status="completed" if hits else "no_hits",
            hits=hits,
        ))
    has_hits = any(task.hits for task in tasks)
    return PlannedRetrievalExecution(
        status="completed" if has_hits else "unavailable",
        resolved_plan=base_plan,
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=app_config.retrieval.embedding_model,
        corpus_version_policy=app_config.retrieval.version_policy,
        corpus_release=app_config.retrieval.target_release,
        tasks=tuple(tasks),
    )


def run_agentic_retrieval_loop(
    base_plan: DiscoveryRetrievalPlanV3,
    *,
    controller_agent: CodexStructuredAgent,
    retriever: HybridMarkdownRetriever,
    app_config: AppConfig,
    shared_context: Mapping[str, Any] | None = None,
    max_rounds: int = 4,
    max_search_actions: int = 6,
    max_expansion_actions: int = 3,
    hits_per_search: int = 8,
    max_evidence_per_question: int = 28,
) -> AgenticRetrievalExecution:
    """Let an agent choose bounded calls to trusted local retrieval tools."""

    for name, value in (
        ("max_rounds", max_rounds),
        ("max_search_actions", max_search_actions),
        ("hits_per_search", hits_per_search),
        ("max_evidence_per_question", max_evidence_per_question),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if max_expansion_actions < 0:
        raise ValueError("max_expansion_actions cannot be negative")

    base_by_query = {item.query_id: item for item in base_plan.retrievals}
    query_ids = tuple(base_by_query)
    query_id_set = frozenset(query_ids)
    available_specs = frozenset(
        document.specification
        for document in retriever.documents
        if document.specification is not None
    )
    ledger = _EvidenceLedger(
        query_ids,
        max_per_question=max_evidence_per_question,
    )
    turns: list[AgentRetrievalTurn] = []
    errors: list[str] = []
    signatures: set[tuple[Any, ...]] = set()
    search_used = 0
    expansion_used = 0
    stop_reason: Literal[
        "agent_finish",
        "round_budget",
        "action_budget",
        "agent_error",
    ] = "round_budget"

    for round_number in range(1, max_rounds + 1):
        history = [
            {
                "round": outcome.round_number,
                "action_id": outcome.action.action_id,
                "tool": outcome.action.tool,
                "query_id": outcome.action.query_id,
                "status": outcome.status,
                "new_evidence_ids": list(outcome.new_evidence_ids),
                "errors": list(outcome.errors),
            }
            for turn in turns
            for outcome in turn.outcomes
        ]
        payload = {
            "questions": [
                {
                    "query_id": item.query_id,
                    "claim_id": item.claim_id,
                    "angle": item.angle,
                    "question": item.original_question,
                }
                for item in base_plan.retrievals
            ],
            "shared_context": dict(shared_context or {}),
            "available_specifications": sorted(available_specs),
            "budget": {
                "rounds_remaining_including_this": max_rounds - round_number + 1,
                "search_actions_remaining": max_search_actions - search_used,
                "expansion_actions_remaining": (
                    max_expansion_actions - expansion_used
                ),
                "hits_per_search_ceiling": hits_per_search,
                "max_evidence_per_question": max_evidence_per_question,
            },
            "history": history,
            "evidence_catalog": ledger.payload(),
        }
        try:
            response = controller_agent.parse(
                purpose="choose the next bounded Discovery retrieval tool calls",
                prompt_version="agentic-discovery-retrieval-controller/v0.1",
                instructions=_CONTROLLER_INSTRUCTIONS,
                input_payload=payload,
                output_type=AgentRetrievalDecision,
            )
        except Exception as exc:
            errors.append(f"controller: {type(exc).__name__}: {exc}")
            stop_reason = "agent_error"
            break

        actions = response.output.actions
        if any(action.tool == "finish" for action in actions):
            if len(actions) != 1 or actions[0].tool != "finish":
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=actions[0],
                    status="rejected",
                    errors=("finish must be the only action in a turn",),
                )
                turns.append(AgentRetrievalTurn(
                    round_number=round_number,
                    assessment=response.output.assessment,
                    receipt=response.receipt,
                    outcomes=(outcome,),
                ))
                errors.extend(outcome.errors)
                continue
            finish_errors = _validate_action(
                actions[0],
                query_ids=query_id_set,
                available_specifications=available_specs,
                ledger=ledger,
            )
            outcome = AgentActionOutcome(
                round_number=round_number,
                action=actions[0],
                status="rejected" if finish_errors else "completed",
                errors=finish_errors,
            )
            turns.append(AgentRetrievalTurn(
                round_number=round_number,
                assessment=response.output.assessment,
                receipt=response.receipt,
                outcomes=(outcome,),
            ))
            if finish_errors:
                errors.extend(finish_errors)
                continue
            stop_reason = "agent_finish"
            break

        outcomes: list[AgentActionOutcome] = []
        for action in actions:
            action_errors = list(_validate_action(
                action,
                query_ids=query_id_set,
                available_specifications=available_specs,
                ledger=ledger,
            ))
            signature = _action_signature(action)
            if signature in signatures:
                action_errors.append("duplicate prior tool call")
            if action.tool == "hybrid_search" and search_used >= max_search_actions:
                action_errors.append("search action budget exhausted")
            if (
                action.tool == "expand_same_section"
                and expansion_used >= max_expansion_actions
            ):
                action_errors.append("expansion action budget exhausted")
            if action_errors:
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="rejected",
                    errors=tuple(action_errors),
                )
                outcomes.append(outcome)
                errors.extend(action_errors)
                continue

            signatures.add(signature)
            if action.tool == "hybrid_search":
                search_used += 1
                search_plan = _build_search_plan(
                    base_by_query[action.query_id],
                    action,
                    round_number=round_number,
                )
                result = execute_planned_discovery_retrieval(
                    search_plan,
                    app_config=app_config,
                    final_hits=min(action.max_results, hits_per_search),
                    retriever=retriever,
                )
                task = result.tasks[0]
                observed, added = ledger.add(
                    action.query_id,
                    task.hits,
                    action_id=action.action_id,
                )
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status=(
                        "completed"
                        if observed
                        else "unavailable"
                        if task.status == "unavailable"
                        else "no_hits"
                    ),
                    evidence_ids=observed,
                    new_evidence_ids=added,
                    errors=task.errors,
                )
            else:
                expansion_used += 1
                seeds = tuple(
                    hit
                    for evidence_id in action.seed_evidence_ids
                    if (hit := ledger.find(action.query_id, evidence_id)) is not None
                )
                expanded_hits = _expand_same_section(
                    query_id=action.query_id,
                    seeds=seeds,
                    retriever=retriever,
                    app_config=app_config,
                    max_results=action.max_results,
                )
                observed, added = ledger.add(
                    action.query_id,
                    expanded_hits,
                    action_id=action.action_id,
                )
                outcome = AgentActionOutcome(
                    round_number=round_number,
                    action=action,
                    status="completed" if observed else "no_hits",
                    evidence_ids=observed,
                    new_evidence_ids=added,
                )
            outcomes.append(outcome)
            errors.extend(outcome.errors)

        turns.append(AgentRetrievalTurn(
            round_number=round_number,
            assessment=response.output.assessment,
            receipt=response.receipt,
            outcomes=tuple(outcomes),
        ))
        if search_used >= max_search_actions and expansion_used >= max_expansion_actions:
            stop_reason = "action_budget"
            break

    evidence = _evidence_execution(
        base_plan=base_plan,
        ledger=ledger,
        retriever=retriever,
        app_config=app_config,
    )
    has_hits = any(task.hits for task in evidence.tasks)
    every_question_has_hits = all(task.hits for task in evidence.tasks)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if not has_hits
        else "partial"
        if errors or not every_question_has_hits
        else "completed"
    )
    return AgenticRetrievalExecution(
        status=status,
        stop_reason=stop_reason,
        rounds_used=len(turns),
        search_actions_used=search_used,
        expansion_actions_used=expansion_used,
        turns=tuple(turns),
        evidence_execution=evidence,
        errors=tuple(errors),
    )


__all__ = [
    "AgentActionOutcome",
    "AgentRetrievalAction",
    "AgentRetrievalDecision",
    "AgentRetrievalTurn",
    "AgenticRetrievalExecution",
    "run_agentic_retrieval_loop",
]
