"""Provider-neutral native function-calling transport contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NativeFunctionDefinition(_StrictModel):
    """One strict function exposed to a model by an owning package."""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    strict: bool = True


class NativeFunctionCall(_StrictModel):
    """One model-requested function call with parsed JSON arguments."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]


class NativeFunctionCallOutput(_StrictModel):
    """One matched function result replayed on the next model round."""

    call_id: str = Field(min_length=1)
    output: dict[str, Any]


__all__ = [
    "NativeFunctionCall",
    "NativeFunctionCallOutput",
    "NativeFunctionDefinition",
]
