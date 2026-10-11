"""Validated contracts for speculative answer-hypothesis retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..infrastructure.structured_agent import CodexAgentReceipt
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


HypothesisScope = Literal[
    "same_layer",
    "adjacent_layer",
    "procedure_context",
    "security_context",
    "state_context",
    "other",
]


class AgentEvidenceQuery(_StrictModel):
    """One specification-local lookup for an evidence requirement."""

    specification: str = Field(min_length=1)
    lexical_query: str = Field(min_length=1)
    semantic_query: str = Field(min_length=1)
    required_terms: tuple[str, ...] = Field(min_length=1, max_length=3)

    @field_validator(
        "specification",
        "lexical_query",
        "semantic_query",
    )
    @classmethod
    def _non_blank_query_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence query text must be non-blank")
        return cleaned

    @field_validator("required_terms")
    @classmethod
    def _unique_query_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("evidence query terms must be non-blank")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("evidence query terms must be unique")
        return cleaned


class AgentEvidenceRequirement(_StrictModel):
    """One relation required by an answer hypothesis."""

    requirement_id: str = Field(min_length=1)
    relation_to_verify: str = Field(min_length=1)
    evidence_queries: tuple[AgentEvidenceQuery, ...] = Field(
        min_length=1,
        max_length=3,
    )

    @field_validator(
        "requirement_id",
        "relation_to_verify",
    )
    @classmethod
    def _non_blank_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("specification search text must be non-blank")
        return cleaned

    @model_validator(mode="after")
    def _unique_evidence_queries(self) -> "AgentEvidenceRequirement":
        keys = {
            (
                query.specification.casefold(),
                query.lexical_query.casefold(),
                query.semantic_query.casefold(),
            )
            for query in self.evidence_queries
        }
        if len(keys) != len(self.evidence_queries):
            raise ValueError("evidence queries must be unique")
        return self


class AgentAnswerHypothesis(_StrictModel):
    hypothesis_id: str = Field(min_length=1)
    possible_answer: str = Field(min_length=1)
    evidence_goal: str = Field(min_length=1)
    scope: HypothesisScope
    search_terms: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence_requirements: tuple[AgentEvidenceRequirement, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @field_validator("search_terms")
    @classmethod
    def _unique_non_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("hypothesis list values must be non-blank")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("hypothesis list values must be unique")
        return cleaned

    @field_validator("evidence_requirements")
    @classmethod
    def _unique_evidence_requirements(
        cls,
        values: tuple[AgentEvidenceRequirement, ...],
    ) -> tuple[AgentEvidenceRequirement, ...]:
        identifiers = tuple(value.requirement_id.casefold() for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evidence requirement IDs must be unique")
        return values

    @property
    def target_specifications(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            query.specification
            for requirement in self.evidence_requirements
            for query in requirement.evidence_queries
        ))


class AgentQuestionHypothesisProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    hypotheses: tuple[AgentAnswerHypothesis, ...] = Field(max_length=6)


class QuestionHypothesisPlan(_StrictModel):
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    status: Literal["planned", "no_expansion", "unavailable"]
    hypotheses: tuple[AgentAnswerHypothesis, ...] = ()
    warnings: tuple[str, ...] = ()


class AnswerHypothesisPlan(_StrictModel):
    schema_version: Literal["answer-hypothesis-plan/v1.3"] = (
        "answer-hypothesis-plan/v1.3"
    )
    release: int = Field(ge=1)
    agent_receipts: tuple[CodexAgentReceipt, ...] = ()
    questions: tuple[QuestionHypothesisPlan, ...]
    errors: tuple[str, ...] = ()



__all__ = [
    'AgentAnswerHypothesis',
    'AgentEvidenceQuery',
    'AgentEvidenceRequirement',
    'AgentQuestionHypothesisProposal',
    'AnswerHypothesisPlan',
    'HypothesisScope',
    'QuestionHypothesisPlan',
]
