"""Audited execution of Conversation Agent calls across the MCP boundary."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.access import ServiceDelegationSigner
from specrag.agents import ConversationToolExecution
from specrag.context import ContextHarness
from specrag.models import (
    ActorContext,
    Conversation,
    ConversationToolCall,
    ConversationTurn,
    MCPResearchResultResponse,
    MCPResearchStartResponse,
    MCPResearchStatusResponse,
    NativeFunctionCall,
    ResearchResult,
    ResearchToolRequest,
)


class ConversationAgentAuditStore(Protocol):
    def start_tool_call(
        self, turn_id: str, *, call_id: str, tool_name: str,
        arguments: dict[str, object], operation_key: str, now=None,
    ) -> ConversationToolCall: ...
    def finish_tool_call(
        self, call_id: str, *, tool_result_id: str, run_id: str | None,
        result_id: str | None, now=None,
    ) -> ConversationToolCall: ...
    def fail_tool_call(
        self, call_id: str, *, safe_error: str, now=None,
    ) -> ConversationToolCall: ...


class ResearchMCPClient(Protocol):
    async def start_research(
        self, delegation_token: str, request: ResearchToolRequest
    ) -> MCPResearchStartResponse: ...
    async def get_research_status(
        self, delegation_token: str, run_id: str
    ) -> MCPResearchStatusResponse: ...
    async def get_research_result(
        self, delegation_token: str, *, result_id: str | None = None,
        run_id: str | None = None,
    ) -> MCPResearchResultResponse: ...


class _StatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")


class _ResultArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    run_id: str | None = Field(default=None, pattern=r"^RUN-[0-9A-F]{32}$")

    @model_validator(mode="after")
    def _one_selector(self) -> "_ResultArguments":
        if (self.result_id is None) == (self.run_id is None):
            raise ValueError("provide exactly one Result ID or Run ID")
        return self


class MCPConversationToolExecutor:
    def __init__(
        self,
        *,
        client: ResearchMCPClient,
        signer: ServiceDelegationSigner,
        actor: ActorContext,
        conversation: Conversation,
        turn: ConversationTurn,
        origin_message_id: str,
        origin_turn_id: str,
        submitted_user_sequence: int,
        audit: ConversationAgentAuditStore,
        context_harness: ContextHarness,
    ) -> None:
        self.client = client
        self.signer = signer
        self.actor = actor
        self.conversation = conversation
        self.turn = turn
        self.origin_message_id = origin_message_id
        self.origin_turn_id = origin_turn_id
        self.submitted_user_sequence = submitted_user_sequence
        self.audit = audit
        self.context_harness = context_harness

    def execute(
        self, call: NativeFunctionCall, *, operation_key: str
    ) -> ConversationToolExecution:
        call = self._validated_call(call)
        call_id = "CALL-" + hashlib.sha256(
            operation_key.encode("utf-8")
        ).hexdigest()[:32].upper()
        audited = self.audit.start_tool_call(
            self.turn.turn_id,
            call_id=call_id,
            tool_name=call.name,
            arguments=call.arguments,
            operation_key=operation_key,
        )
        if audited.status == "completed":
            return self._replay(audited)
        if audited.status == "failed":
            raise RuntimeError(audited.safe_error or "prior MCP tool call failed")
        token = self.signer.issue(
            self.actor,
            conversation_id=self.conversation.conversation_id,
            origin_message_id=self.origin_message_id,
            origin_turn_id=self.origin_turn_id,
            submitted_user_sequence=self.submitted_user_sequence,
            tool_call_id=call_id,
            operation_key=operation_key,
            lifetime=timedelta(seconds=60),
            now=datetime.now(timezone.utc),
        )
        try:
            response = asyncio.run(self._invoke(call, token))
            full = response.model_dump(mode="json")
            results = self._results(response)
            page = self.context_harness.record_tool_result(
                self.conversation.context_id, call_id, full
            )
            output = self._model_output(full, results, page.truncated)
            run_id, result_id = self._identifiers(response)
            self.audit.finish_tool_call(
                call_id,
                tool_result_id=page.tool_result_id,
                run_id=run_id,
                result_id=result_id,
            )
            return ConversationToolExecution(output=output, results=results)
        except Exception as exc:
            self.audit.fail_tool_call(
                call_id,
                safe_error=f"{type(exc).__name__}: research tool failed",
            )
            raise

    async def _invoke(self, call: NativeFunctionCall, token: str):
        if call.name == "start_research":
            return await self.client.start_research(
                token, ResearchToolRequest.model_validate(call.arguments)
            )
        if call.name == "get_research_status":
            arguments = _StatusArguments.model_validate(call.arguments)
            return await self.client.get_research_status(token, arguments.run_id)
        if call.name == "get_research_result":
            arguments = _ResultArguments.model_validate(call.arguments)
            return await self.client.get_research_result(
                token,
                result_id=arguments.result_id,
                run_id=arguments.run_id,
            )
        raise ValueError(f"unsupported Conversation tool: {call.name}")

    @staticmethod
    def _validated_call(call: NativeFunctionCall) -> NativeFunctionCall:
        """Validate before audit persistence so identity-like extras are not stored."""

        if call.name == "start_research":
            arguments = ResearchToolRequest.model_validate(
                call.arguments
            ).model_dump(mode="json", exclude_none=True)
        elif call.name == "get_research_status":
            arguments = _StatusArguments.model_validate(
                call.arguments
            ).model_dump(mode="json", exclude_none=True)
        elif call.name == "get_research_result":
            arguments = _ResultArguments.model_validate(
                call.arguments
            ).model_dump(mode="json", exclude_none=True)
        else:
            raise ValueError(f"unsupported Conversation tool: {call.name}")
        return call.model_copy(update={"arguments": arguments})

    def _replay(self, call: ConversationToolCall) -> ConversationToolExecution:
        if call.tool_result_id is None:
            raise RuntimeError("completed tool call has no stored result")
        stored = self.context_harness.store.get_tool_result(call.tool_result_id)
        if stored is None or not isinstance(stored.content, dict):
            raise RuntimeError("stored tool result is unavailable")
        results = self._results_from_dict(stored.content)
        return ConversationToolExecution(
            output=self._model_output(stored.content, results, False),
            results=results,
        )

    @staticmethod
    def _results(response: object) -> tuple[ResearchResult, ...]:
        result = getattr(response, "result", None)
        return (result,) if isinstance(result, ResearchResult) else ()

    @staticmethod
    def _results_from_dict(content: dict[str, object]) -> tuple[ResearchResult, ...]:
        raw = content.get("result")
        return (ResearchResult.model_validate(raw),) if isinstance(raw, dict) else ()

    @staticmethod
    def _identifiers(response: object) -> tuple[str | None, str | None]:
        run_id = getattr(response, "run_id", None)
        result_id = getattr(response, "result_id", None)
        result = getattr(response, "result", None)
        if isinstance(result, ResearchResult):
            run_id = result.run_id
            result_id = result.result_id
        return run_id, result_id

    @staticmethod
    def _model_output(
        full: dict[str, object],
        results: tuple[ResearchResult, ...],
        truncated: bool,
    ) -> dict[str, object]:
        if not truncated:
            return full
        output = {key: value for key, value in full.items() if key != "result"}
        if results:
            result = results[0]
            output["result"] = {
                "result_id": result.result_id,
                "run_id": result.run_id,
                "evidence_status": result.evidence_status,
                "answer": result.answer,
                "evidence_ids": list(result.evidence_ids),
                "conditions": list(result.conditions),
                "unresolved_questions": list(result.unresolved_questions),
            }
        output["full_tool_result_persisted"] = True
        return output


__all__ = [
    "ConversationAgentAuditStore",
    "MCPConversationToolExecutor",
    "ResearchMCPClient",
]
