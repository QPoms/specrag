"""Configuration adapter for the provider-independent token budget policy."""

from specrag.config import AppConfig
from specrag.context import ModelProfileResolver, TokenBudgetManager
from specrag.models import ModelContextProfile


def build_token_budget_manager(config: AppConfig) -> TokenBudgetManager:
    default_cfg = config.llm.context_default_profile
    default = ModelContextProfile(
        model_id="default",
        context_window_tokens=default_cfg.context_window_tokens,
        max_input_tokens=default_cfg.max_input_tokens,
        expected_output_tokens=default_cfg.expected_output_tokens,
    )
    overrides = {
        model_id: ModelContextProfile(
            model_id=model_id,
            context_window_tokens=profile.context_window_tokens,
            max_input_tokens=profile.max_input_tokens,
            expected_output_tokens=profile.expected_output_tokens,
        )
        for model_id, profile in config.llm.context_model_profiles.items()
    }
    return TokenBudgetManager(
        ModelProfileResolver(default, overrides),
        safety_ratio=config.llm.context_safety_ratio,
        large_window_threshold_tokens=config.llm.context_large_window_threshold_tokens,
        large_window_minimum_reserve_tokens=(
            config.llm.context_large_window_minimum_reserve_tokens
        ),
    )


__all__ = ["build_token_budget_manager"]
