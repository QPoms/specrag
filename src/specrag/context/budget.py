"""Deterministic, conservative token estimation and model budget policy."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from specrag.models import ContextBudget, ModelContextProfile


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: (
            item.model_dump(mode="json") if isinstance(item, BaseModel)
            else _raise_not_json(item)
        ),
    )


def _raise_not_json(value: Any) -> Any:
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ConservativeTokenEstimator:
    """Portable estimate that is conservative for English, CJK, and JSON."""

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        # Three UTF-8 bytes per token intentionally overestimates most English
        # and maps common CJK characters close to one token each.
        return max(1, math.ceil(len(text.encode("utf-8")) / 3))

    def estimate(self, value: Any) -> int:
        return self.estimate_text(value if isinstance(value, str) else canonical_json(value))


class ModelProfileResolver:
    def __init__(
        self,
        default: ModelContextProfile,
        overrides: Mapping[str, ModelContextProfile] | None = None,
    ) -> None:
        self.default = default
        self.overrides = dict(overrides or {})

    def resolve(self, model_id: str) -> ModelContextProfile:
        selected = self.overrides.get(model_id)
        if selected is not None:
            return selected
        return self.default.model_copy(update={"model_id": model_id})


class TokenBudgetManager:
    def __init__(
        self,
        profiles: ModelProfileResolver,
        *,
        estimator: ConservativeTokenEstimator | None = None,
        safety_ratio: float = 0.15,
        large_window_threshold_tokens: int = 100_000,
        large_window_minimum_reserve_tokens: int = 16_000,
    ) -> None:
        if not 0 < safety_ratio < 1:
            raise ValueError("safety_ratio must be between zero and one")
        if large_window_threshold_tokens < 1 or large_window_minimum_reserve_tokens < 1:
            raise ValueError("large-window reserve settings must be positive")
        self.profiles = profiles
        self.estimator = estimator or ConservativeTokenEstimator()
        self.safety_ratio = safety_ratio
        self.large_window_threshold_tokens = large_window_threshold_tokens
        self.large_window_minimum_reserve_tokens = large_window_minimum_reserve_tokens

    def inspect(
        self,
        model_id: str,
        components: Mapping[str, Any],
        *,
        expected_output_tokens: int | None = None,
    ) -> ContextBudget:
        profile = self.profiles.resolve(model_id)
        output_reserve = expected_output_tokens or profile.expected_output_tokens
        if output_reserve < 1:
            raise ValueError("expected output tokens must be positive")
        safety_reserve = math.ceil(profile.context_window_tokens * self.safety_ratio)
        conditional_minimum = (
            self.large_window_minimum_reserve_tokens
            if profile.context_window_tokens >= self.large_window_threshold_tokens
            else 0
        )
        reserve = max(output_reserve, safety_reserve, conditional_minimum)
        ceiling = max(0, min(
            profile.max_input_tokens,
            profile.context_window_tokens - reserve,
        ))
        component_tokens = {
            name: self.estimator.estimate(value) for name, value in components.items()
        }
        estimated = sum(component_tokens.values())
        return ContextBudget(
            model_id=model_id,
            estimated_input_tokens=estimated,
            reserve_tokens=reserve,
            input_ceiling_tokens=ceiling,
            fits=estimated <= ceiling,
            component_tokens=component_tokens,
        )


__all__ = [
    "ConservativeTokenEstimator", "ModelProfileResolver", "TokenBudgetManager",
    "canonical_json",
]
