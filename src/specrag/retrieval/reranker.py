"""Optional cross-encoder reranking for specification candidates.

The dependency is loaded lazily so deterministic retrieval remains usable when
reranking is disabled or the local model is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class CrossEncoderReranker:
    """Score query/passage pairs with a lazily loaded CrossEncoder."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    deterministic_weight: float = 8.0
    batch_size: int = 16
    max_length: int = 512
    device: str = "cpu"
    _model: Any | None = field(default=None, init=False, repr=False)
    _load_error: RuntimeError | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.deterministic_weight < 0:
            raise ValueError("deterministic_weight must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.device.strip():
            raise ValueError("device must not be empty")

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise self._load_error
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        except Exception as exc:  # Keep deterministic retrieval available.
            self._load_error = RuntimeError(
                f"could not load reranker {self.model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise self._load_error from exc
        return self._model

    def predict(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        if not passages:
            return ()
        model = self._load()
        try:
            values = model.predict(
                [(query, passage) for passage in passages],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"reranker inference failed for {self.model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return tuple(float(value) for value in values)

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "sentence_transformers.CrossEncoder",
            "model": self.model_name,
            "deterministic_weight": self.deterministic_weight,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
        }


__all__ = ["CrossEncoderReranker"]
