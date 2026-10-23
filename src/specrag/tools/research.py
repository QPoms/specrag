"""Provider-neutral native tools exposed to the Conversation Agent."""

from __future__ import annotations

from specrag.models import NativeFunctionDefinition, ResearchToolRequest


def conversation_research_tool_definitions() -> tuple[NativeFunctionDefinition, ...]:
    """Return strict research and final-response tools for one Agent Turn."""

    result_selector = {
        "type": "object",
        "properties": {
            "result_id": {"type": ["string", "null"]},
            "run_id": {"type": ["string", "null"]},
        },
        "required": ["result_id", "run_id"],
        "additionalProperties": False,
    }
    response = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "source_kind": {
                "type": "string",
                "enum": [
                    "direct_conversation",
                    "research_result",
                    "historical_result_readback",
                ],
            },
            "research_result_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "run_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        "required": [
            "text",
            "source_kind",
            "research_result_ids",
            "run_ids",
            "citations",
        ],
        "additionalProperties": False,
    }
    return (
        NativeFunctionDefinition(
            name="start_research",
            description=(
                "Create one durable evidence-backed 3GPP Research Run. Use this "
                "for specification facts, protocol behavior, or technical judgments."
            ),
            parameters=ResearchToolRequest.model_json_schema(),
        ),
        NativeFunctionDefinition(
            name="get_research_status",
            description="Read the lifecycle status of an already authorized Run.",
            parameters={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        ),
        NativeFunctionDefinition(
            name="get_research_result",
            description="Read one durable authorized Research Result.",
            parameters=result_selector,
        ),
        NativeFunctionDefinition(
            name="respond",
            description=(
                "Publish the final response for this turn. Research-backed claims "
                "must name the applicable Results and only their valid evidence IDs."
            ),
            parameters=response,
        ),
    )


__all__ = ["conversation_research_tool_definitions"]
