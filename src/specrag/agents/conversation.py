"""Bounded native-tool loop for the persistent Conversation Agent."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from specrag.models import (
    AssistantSourceKind,
    ConversationMessageContent,
    ConversationTurnBudget,
    NativeFunctionCall,
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
    ResearchResult,
    ResearchToolRequest,
)
from specrag.tools.research import conversation_research_tool_definitions


class ConversationModelResponse(Protocol):
    calls: tuple[NativeFunctionCall, ...]
    replay_items: tuple[dict[str, Any], ...]


class ConversationModel(Protocol):
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
    ) -> ConversationModelResponse: ...


class ConversationToolExecutor(Protocol):
    def execute(
        self, call: NativeFunctionCall, *, operation_key: str
    ) -> "ConversationToolExecution": ...


class _ResponseProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    source_kind: AssistantSourceKind
    research_result_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationAgentOutcome:
    message: ConversationMessageContent
    stop_reason: str
    model_steps: int
    tool_calls: int
    research_submissions: int


@dataclass(frozen=True)
class ConversationToolExecution:
    output: dict[str, Any]
    results: tuple[ResearchResult, ...] = ()


_NORMATIVE = re.compile(
    r"(?:\b3gpp\b|\bts\s*\d{2}\.|\brelease\s*\d+\b|\bshall\b|\bmust\b|"
    r"规范(?:规定|要求|定义)|协议(?:规定|要求|行为)|字段(?:含义|定义)|"
    r"消息(?:定义|流程)|状态(?:转换|变化)|技术要求)",
    re.IGNORECASE,
)


class ConversationAgent:
    def __init__(
        self,
        model: ConversationModel,
        *,
        instructions: str,
        prompt_version: str,
    ) -> None:
        if not instructions.strip() or not prompt_version.strip():
            raise ValueError("Conversation Agent prompt metadata cannot be blank")
        self.model = model
        self.instructions = instructions
        self.prompt_version = prompt_version
        self.tools = conversation_research_tool_definitions()

    def run(
        self,
        input_payload: Mapping[str, Any],
        *,
        budget: ConversationTurnBudget,
        executor: ConversationToolExecutor,
        historical_results: Mapping[str, ResearchResult] | None = None,
        trigger_result_id: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
        operation_prefix: str,
    ) -> ConversationAgentOutcome:
        started = time.monotonic()
        prior_items: tuple[dict[str, Any], ...] = ()
        prior_outputs: tuple[NativeFunctionCallOutput, ...] = ()
        available = dict(historical_results or {})
        current_result_ids: set[str] = set()
        if trigger_result_id is not None and trigger_result_id in available:
            current_result_ids.add(trigger_result_id)
        observed_run_ids: set[str] = set()
        tool_calls = 0
        research_submissions = 0

        for model_step in range(1, budget.max_model_steps + 1):
            if stop_requested is not None and stop_requested():
                raise RuntimeError("conversation turn cancelled")
            if time.monotonic() - started >= budget.elapsed_time_seconds:
                raise TimeoutError("conversation turn elapsed-time budget exhausted")
            remaining = budget.max_tool_calls - tool_calls
            response = self.model.call_tools(
                purpose="conversation_turn",
                prompt_version=self.prompt_version,
                instructions=self.instructions,
                input_payload={
                    **input_payload,
                    "execution": {
                        "model_step": model_step,
                        "remaining_tool_calls": remaining,
                        "remaining_research_submissions": (
                            budget.max_research_submissions - research_submissions
                        ),
                    },
                },
                tools=self.tools,
                prior_response_items=prior_items,
                prior_outputs=prior_outputs,
                max_tool_calls=max(1, remaining),
            )
            if not response.calls:
                raise RuntimeError("conversation model returned no native call")
            responses = tuple(call for call in response.calls if call.name == "respond")
            if responses:
                if len(response.calls) != 1 or len(responses) != 1:
                    raise RuntimeError("respond must be the only call in its model step")
                proposal = _ResponseProposal.model_validate(responses[0].arguments)
                message = self._validate_response(
                    proposal,
                    available=available,
                    current_result_ids=current_result_ids,
                    observed_run_ids=observed_run_ids,
                )
                return ConversationAgentOutcome(
                    message=message,
                    stop_reason="responded",
                    model_steps=model_step,
                    tool_calls=tool_calls,
                    research_submissions=research_submissions,
                )

            outputs: list[NativeFunctionCallOutput] = []
            for call_index, call in enumerate(response.calls, start=1):
                if call.name not in {
                    "start_research", "get_research_status", "get_research_result"
                }:
                    raise ValueError(f"unsupported Conversation tool: {call.name}")
                if tool_calls >= budget.max_tool_calls:
                    raise RuntimeError("conversation turn tool-call budget exhausted")
                if call.name == "start_research":
                    ResearchToolRequest.model_validate(call.arguments)
                    research_submissions += 1
                    if research_submissions > budget.max_research_submissions:
                        raise RuntimeError("research-submission budget exhausted")
                tool_calls += 1
                execution = executor.execute(
                    call,
                    operation_key=(
                        f"{operation_prefix}:step:{model_step}:call:{call_index}"
                    ),
                )
                output = execution.output
                self._observe(
                    output, execution.results, available,
                    current_result_ids, observed_run_ids,
                )
                outputs.append(NativeFunctionCallOutput(
                    call_id=call.call_id,
                    output=output,
                ))
            prior_items = response.replay_items
            prior_outputs = tuple(outputs)

        raise RuntimeError("conversation turn model-step budget exhausted")

    @staticmethod
    def _observe(
        output: Mapping[str, Any],
        trusted_results: tuple[ResearchResult, ...],
        available: dict[str, ResearchResult],
        current_result_ids: set[str],
        observed_run_ids: set[str],
    ) -> None:
        run_id = output.get("run_id")
        if isinstance(run_id, str):
            observed_run_ids.add(run_id)
        for result in trusted_results:
            available[result.result_id] = result
            current_result_ids.add(result.result_id)
            observed_run_ids.add(result.run_id)

    @staticmethod
    def _validate_response(
        proposal: _ResponseProposal,
        *,
        available: Mapping[str, ResearchResult],
        current_result_ids: set[str],
        observed_run_ids: set[str],
    ) -> ConversationMessageContent:
        result_ids = proposal.research_result_ids
        if proposal.source_kind == "direct_conversation":
            if result_ids or proposal.citations:
                raise ValueError("direct response cannot claim research evidence")
            if _NORMATIVE.search(proposal.text):
                raise ValueError("specification-backed claim requires a Research Result")
        else:
            if not result_ids:
                raise ValueError("research-backed response requires a Result")
            missing = set(result_ids) - set(available)
            if missing:
                raise ValueError("response references an unavailable Research Result")
            cited = set(proposal.citations)
            valid = {
                evidence_id
                for result_id in result_ids
                for evidence_id in available[result_id].evidence_ids
            }
            if cited - valid:
                raise ValueError("response cites evidence outside referenced Results")
            if valid and not cited:
                raise ValueError("research-backed response must retain Result citations")
            if proposal.source_kind == "research_result" and not (
                set(result_ids) & current_result_ids
            ):
                raise ValueError("current Result response requires a Result from this turn")
            if proposal.source_kind == "historical_result_readback" and (
                set(result_ids) & current_result_ids
            ):
                raise ValueError("historical readback cannot relabel a current Result")
        unobserved_runs = set(proposal.run_ids) - (
            observed_run_ids | {available[item].run_id for item in result_ids}
        )
        if unobserved_runs:
            raise ValueError("response references an unobserved Research Run")
        return ConversationMessageContent(
            text=proposal.text.strip(),
            source_kind=proposal.source_kind,
            research_result_ids=result_ids,
            run_ids=proposal.run_ids,
            citations=proposal.citations,
        )


__all__ = [
    "ConversationAgent",
    "ConversationAgentOutcome",
    "ConversationModel",
    "ConversationToolExecutor",
    "ConversationToolExecution",
]
