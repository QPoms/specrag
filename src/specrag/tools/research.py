"""Provider-neutral native tools exposed to the Conversation Agent."""

from __future__ import annotations

from specrag.models import NativeFunctionDefinition


def _research_request_schema() -> dict[str, object]:
    """Build a strict function schema rather than a default-oriented schema."""

    version_ref = {
        "type": "object",
        "properties": {
            "specification": {"type": "string"},
            "version": {"type": "string"},
            "release": {"type": ["integer", "null"]},
        },
        "required": ["specification", "version", "release"],
        "additionalProperties": False,
    }
    scope = {
        "type": "object",
        "properties": {
            "policy": {
                "type": "string",
                "enum": [
                    "latest_per_spec",
                    "highest_version_within_release",
                    "exact_version",
                    "mixed_versions",
                ],
            },
            "release": {"type": ["integer", "null"]},
            "versions": {"type": "array", "items": version_ref},
        },
        "required": ["policy", "release", "versions"],
        "additionalProperties": False,
    }
    prior_result = {
        "type": "object",
        "properties": {
            "result_id": {"type": "string"},
            "content_sha256": {"type": "string"},
            "relationship": {
                "type": "string",
                "enum": [
                    "follow_up",
                    "condition_change",
                    "revalidation",
                    "continuation",
                ],
            },
        },
        "required": ["result_id", "content_sha256", "relationship"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "objective": {"type": ["string", "null"]},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "requested_scope": {"anyOf": [scope, {"type": "null"}]},
            "prior_results": {"type": "array", "items": prior_result},
            "answer_language": {"type": "string"},
            "detail_level": {"type": "string"},
        },
        "required": [
            "question",
            "objective",
            "conditions",
            "requested_scope",
            "prior_results",
            "answer_language",
            "detail_level",
        ],
        "additionalProperties": False,
    }


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
            },
            "run_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
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
            parameters=_research_request_schema(),
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
