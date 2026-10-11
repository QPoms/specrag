"""RAG 配置装载：密钥来自 .env，检索参数来自 config.yaml。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 仓库根 = 本文件上溯三级(src/specrag/config.py → 仓库根)
REPO_ROOT = Path(__file__).resolve().parents[2]


class Secrets(BaseSettings):
    """OpenAI-compatible credentials loaded from the project environment."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    # Optional role-specific models for the retrieval API backend.  When they
    # are empty, OPENAI_MODEL remains the shared fallback.
    openai_planning_model: str = ""
    openai_summary_model: str = ""

_PROJECT_SECRET_ENV_FIELDS = {
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_MODEL": "openai_model",
    "OPENAI_PLANNING_MODEL": "openai_planning_model",
    "OPENAI_SUMMARY_MODEL": "openai_summary_model",
}


def _load_project_secrets() -> Secrets:
    """Load the project's checked-in configuration without stale env leakage.

    ``BaseSettings`` gives inherited process variables precedence over the
    dotenv file.  That is normally useful, but it can silently route this
    project to an old proxy/key after the project ``.env`` has been updated.
    Values explicitly present in the project file therefore take precedence
    for the OpenAI-compatible backend; unrelated environment variables still
    retain the normal fallback behavior.
    """
    values = dotenv_values(REPO_ROOT / ".env")
    overrides: dict[str, str] = {}
    for env_name, field_name in _PROJECT_SECRET_ENV_FIELDS.items():
        value = values.get(env_name)
        if value is not None and str(value).strip():
            overrides[field_name] = str(value).strip()
    return Secrets(**overrides)


class RetrievalCfg(BaseModel):
    """Global mechanics for specification evidence retrieval.

    These values are deliberately configuration, not fields in a semantic
    Query or Retrieval Plan.
    """

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_dir: str = ".local/indexes/specifications"
    specification_root: str = "data/3gpp"
    version_policy: Literal[
        "latest_per_spec",
        "highest_version_within_release",
    ] = "highest_version_within_release"
    target_release: int | None = 17
    chunk_characters: int = 2400
    chunk_overlap_characters: int = 300
    semantic_candidates_per_collection: int = 20
    lexical_candidates_per_collection: int = 20
    final_hits_per_collection: int = 6
    rrf_k: int = 60

    @model_validator(mode="after")
    def _valid_retrieval_mechanics(self):
        if (
            self.version_policy == "highest_version_within_release"
            and self.target_release is None
        ):
            raise ValueError(
                "target_release is required for highest_version_within_release"
            )
        if self.version_policy == "latest_per_spec" and self.target_release is not None:
            raise ValueError("target_release is not valid for latest_per_spec")
        if self.target_release is not None and self.target_release < 1:
            raise ValueError("target_release must be positive")
        positive = {
            "chunk_characters": self.chunk_characters,
            "semantic_candidates_per_collection": (
                self.semantic_candidates_per_collection
            ),
            "lexical_candidates_per_collection": (
                self.lexical_candidates_per_collection
            ),
            "final_hits_per_collection": self.final_hits_per_collection,
            "rrf_k": self.rrf_k,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(
                "retrieval values must be positive: "
                + ", ".join(invalid)
            )
        if not 0 <= self.chunk_overlap_characters < self.chunk_characters:
            raise ValueError(
                "chunk overlap must be non-negative and below chunk size"
            )
        return self


class LlmCfg(BaseModel):
    temperature: float = 0.4
    max_retries: int = 3
    api_max_retries: int = 8
    api_retry_initial_delay_seconds: float = 1.0
    api_retry_max_delay_seconds: float = 15.0
    api_fallback_models: tuple[str, ...] = (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
    )

    @model_validator(mode="after")
    def _retry_policy(self):
        if self.max_retries < 1:
            raise ValueError("llm.max_retries must be positive")
        if self.api_max_retries < 0:
            raise ValueError("llm.api_max_retries must be non-negative")
        if self.api_retry_initial_delay_seconds < 0:
            raise ValueError("LLM API retry delay must be non-negative")
        if self.api_retry_max_delay_seconds < self.api_retry_initial_delay_seconds:
            raise ValueError("LLM API maximum retry delay is below initial delay")
        normalized = tuple(model.strip() for model in self.api_fallback_models)
        if any(not model for model in normalized):
            raise ValueError("LLM API fallback model names must be non-blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("LLM API fallback model names must be unique")
        self.api_fallback_models = normalized
        return self


class AppConfig(BaseModel):
    retrieval: RetrievalCfg = RetrievalCfg()
    llm: LlmCfg = LlmCfg()
    secrets: Secrets = Secrets()

    # —— 路径便捷方法(相对 config.yaml 一律解析为仓库根下的绝对路径)——
    def abspath(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (REPO_ROOT / p)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> AppConfig:
    """装载并缓存 AppConfig。config_path 默认仓库根 config.yaml。"""
    path = Path(config_path) if config_path else (REPO_ROOT / "config.yaml")
    raw = _load_yaml(path)
    raw["secrets"] = _load_project_secrets()
    return AppConfig(**raw)
