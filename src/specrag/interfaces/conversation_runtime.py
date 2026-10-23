"""Composition of the Conversation Agent runtime at the application boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass

from specrag.access import ServiceDelegationSigner
from specrag.agents import ConversationAgent
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.orchestration import (
    ConversationTurnWorker,
    ResearchLifecycleWorker,
    RunWorker,
)
from specrag.prompts import (
    CONVERSATION_AGENT_INSTRUCTIONS,
    CONVERSATION_AGENT_PROMPT_VERSION,
)

from .mcp import SpecRAGMCPClient
from .workspace_services import WorkspaceServices


DELEGATION_SECRET_ENV = "SPECRAG_MCP_DELEGATION_SECRET"
MCP_URL_ENV = "SPECRAG_MCP_URL"


def build_conversation_worker(
    services: WorkspaceServices,
    *,
    mcp_client: SpecRAGMCPClient | None = None,
    delegation_secret: bytes | None = None,
    model: LLMApiStructuredAgent | None = None,
) -> ConversationTurnWorker:
    secret = delegation_secret
    if secret is None:
        raw = os.environ.get(DELEGATION_SECRET_ENV)
        if raw is None:
            raise RuntimeError(f"{DELEGATION_SECRET_ENV} is required")
        secret = raw.encode("utf-8")
    agent_model = model or LLMApiStructuredAgent(
        role="summary", reasoning_effort="high", timeout_seconds=60
    )
    agent = ConversationAgent(
        agent_model,
        instructions=CONVERSATION_AGENT_INSTRUCTIONS,
        prompt_version=CONVERSATION_AGENT_PROMPT_VERSION,
    )
    return ConversationTurnWorker(
        queue=services.conversation_turn_queue,
        store=services.conversation_agent_store,
        conversations=services.conversation_store,
        lifecycle=services.conversation_lifecycle_store,
        identity=services.identity,
        context_harness=services.context_harness,
        memory_recall=services.memory_recall,
        memory=services.memory,
        results=services.research_results,
        agent=agent,
        mcp_client=mcp_client or SpecRAGMCPClient(
            os.environ.get(MCP_URL_ENV, "http://127.0.0.1:8766/mcp")
        ),
        delegation_signer=ServiceDelegationSigner(secret),
    )


def optional_conversation_worker(
    services: WorkspaceServices,
) -> ConversationTurnWorker | None:
    """Enable embedded Turn execution only when trusted delegation is configured."""

    if not os.environ.get(DELEGATION_SECRET_ENV):
        return None
    return build_conversation_worker(services)


@dataclass
class ApplicationWorkers:
    run: RunWorker
    lifecycle: ResearchLifecycleWorker
    conversation: ConversationTurnWorker | None
    owned: tuple[bool, bool, bool]

    @classmethod
    def create(
        cls,
        services: WorkspaceServices,
        run: RunWorker | None,
        lifecycle: ResearchLifecycleWorker | None,
        conversation: ConversationTurnWorker | None,
    ) -> "ApplicationWorkers":
        return cls(
            run=run or RunWorker(services.harness, services.run_queue),
            lifecycle=lifecycle or ResearchLifecycleWorker(
                services.runs,
                services.research_results,
                services.conversation_lifecycle_store,
                services.completion_queue,
            ),
            conversation=conversation or optional_conversation_worker(services),
            owned=(run is None, lifecycle is None, conversation is None),
        )

    def start(self) -> None:
        self.run.start()
        self.lifecycle.start()
        if self.conversation is not None:
            self.conversation.start()

    def stop(self) -> None:
        if self.owned[0]:
            self.run.stop()
        if self.owned[1]:
            self.lifecycle.stop()
        if self.owned[2] and self.conversation is not None:
            self.conversation.stop()
__all__ = [
    "DELEGATION_SECRET_ENV",
    "MCP_URL_ENV",
    "ApplicationWorkers",
    "build_conversation_worker",
    "optional_conversation_worker",
]
