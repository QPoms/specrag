"""Redacted application health projection for model and research runtimes."""

from __future__ import annotations

from typing import Any

from specrag.models import NativeToolCapability


async def health_projection(
    capability: NativeToolCapability,
    conversation_worker: Any | None,
) -> dict[str, object]:
    if conversation_worker is None:
        research_service: dict[str, object] = {
            "status": "disabled",
            "release": 17,
            "message": "Research service probing is disabled with the Conversation Agent.",
            "stages": [],
            "total_ms": 0,
        }
    else:
        try:
            readiness = await conversation_worker.mcp_client.readiness()
        except Exception:
            research_service = {
                "status": "unavailable",
                "release": 17,
                "message": "The internal research service is not reachable.",
                "stages": [],
                "total_ms": 0,
            }
        else:
            research_service = readiness.model_dump(mode="json")
    return {
        "status": (
            "ok"
            if capability.status == "ready"
            and research_service["status"] == "ready"
            else "degraded"
        ),
        "conversation_agent": capability.model_dump(mode="json"),
        "research_service": research_service,
    }


__all__ = ["health_projection"]
