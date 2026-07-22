"""Runtime data objects for foundation-model forecasters."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelHandle:
    """Loaded model state."""

    model: Any | None = None
    tokenizer: Any | None = None
    pipeline: Any | None = None


@dataclass(frozen=True)
class ForecastRequest:
    """Prediction request normalized from sktime forecasting inputs."""

    relative_fh: tuple[int, ...]
    absolute_index: Any
    alpha: tuple[float, ...] | None


@dataclass
class ForecastResult:
    """Model-family-neutral forecast output."""

    mean: Any | None = None
    median: Any | None = None
    quantiles: Mapping[float, Any] | None = None
