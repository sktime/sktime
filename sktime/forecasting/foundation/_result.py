"""Runtime data objects for foundation-model forecasters."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sktime.forecasting.foundation._config import InferenceConfig


@dataclass
class ModelHandle:
    """Loaded model state."""

    model: Any | None = None
    tokenizer: Any | None = None
    pipeline: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    shareable: bool = True
    mutable: bool = False


@dataclass
class ModelContext:
    """Model-native context plus metadata needed for sktime output."""

    values: Any
    past_covariates: Any | None = None
    future_covariates: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastRequest:
    """Prediction request normalized from sktime forecasting inputs."""

    relative_fh: tuple[int, ...]
    absolute_index: Any
    cutoff: Any
    alpha: tuple[float, ...] | None
    coverage: tuple[float, ...] | None
    inference: InferenceConfig


@dataclass
class ForecastResult:
    """Model-family-neutral forecast output."""

    mean: Any | None = None
    median: Any | None = None
    quantiles: Mapping[float, Any] | None = None
    samples: Any | None = None
    raw: Any | None = None

    @property
    def has_quantiles(self) -> bool:
        """Return whether quantile predictions are available."""
        return self.quantiles is not None

    @property
    def has_samples(self) -> bool:
        """Return whether sample predictions are available."""
        return self.samples is not None
