"""Canonical hashing and Responses payload helpers for LLM adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from specrag.models import NativeFunctionCallOutput, NativeFunctionDefinition


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def responses_tool_payload(
    definition: NativeFunctionDefinition,
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
        "strict": definition.strict,
    }


def responses_output_payload(
    output: NativeFunctionCallOutput,
) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": output.call_id,
        "output": canonical_json(output.output),
    }


__all__ = [
    "canonical_json",
    "responses_output_payload",
    "responses_tool_payload",
    "sha256_text",
]
