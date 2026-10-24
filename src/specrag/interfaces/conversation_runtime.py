"""Composition of the Conversation Agent runtime at the application boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
import logging

from specrag.access import ServiceDelegationSigner
from specrag.agents import ConversationAgent
from specrag.infrastructure import probe_native_tool_capability
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.models import NativeToolCapability
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
_LOGGER = logging.getLogger(__name__)


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
    capability = (
        probe_native_tool_capability(agent_model, agent.tools)
        if model is None
        else NativeToolCapability(
            status="ready",
            api_mode="responses",
            message="The injected Conversation model is available.",
        )
    )
    log = _LOGGER.info if capability.status == "ready" else _LOGGER.error
    log("Conversation model capability: %s", capability.message)
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
        model_capability=capability,
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

    @property
    def conversation_capability(self) -> NativeToolCapability:
        if self.conversation is None:
            return NativeToolCapability(
                status="disabled",
                message=(
                    "Conversation Agent is disabled because the MCP delegation "
                    "secret is not configured in the API process."
                ),
            )
        return self.conversation.model_capability or NativeToolCapability(
            status="ready",
            message="The injected Conversation model did not require a live probe.",
        )

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

    def start(
        self, *, research: bool = True, lifecycle: bool = True,
        conversation: bool = True,
    ) -> None:
        if research:
            self.run.start()
        if lifecycle:
            self.lifecycle.start()
        if conversation and self.conversation is not None:
            self.conversation.start()

    def stop(
        self, *, research: bool = True, lifecycle: bool = True,
        conversation: bool = True,
    ) -> None:
        if research and self.owned[0]:
            self.run.stop()
        if lifecycle and self.owned[1]:
            self.lifecycle.stop()
        if conversation and self.owned[2] and self.conversation is not None:
            self.conversation.stop()
__all__ = [
    "DELEGATION_SECRET_ENV",
    "MCP_URL_ENV",
    "ApplicationWorkers",
    "build_conversation_worker",
    "optional_conversation_worker",
]
