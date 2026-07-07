"""Configuration objects for foundation-model forecasters."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelLoadConfig:
    """Normalized model loading policy."""

    model_path: str | None
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    token: str | bool | None = None
    device_map: Any = "cpu"
    dtype: Any | None = None
    quantization_config: Any | None = None
    trust_remote_code: bool = False
    extra_load_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterEfficientTuneConfig:
    """Parameter-efficient tuning policy."""

    kind: Literal["none", "peft", "native"] = "none"
    peft_config: Any | None = None
    peft_model_id: str | None = None
    merge_after_train: bool = False


@dataclass(frozen=True)
class FineTuneConfig:
    """Normalized fine-tuning policy."""

    strategy: Literal["zero-shot", "minimal", "head-only", "peft", "full"] = "zero-shot"
    training_args: Mapping[str, Any] = field(default_factory=dict)
    parameter_efficient: ParameterEfficientTuneConfig = field(
        default_factory=ParameterEfficientTuneConfig
    )
    trainer_backend: Literal["none", "transformers", "lightning", "custom"] = "none"


@dataclass(frozen=True)
class InferenceConfig:
    """Normalized inference policy."""

    batch_size: int | None = None
    num_samples: int | None = None
    random_state: int | None = None
    forward_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoundationModelSpec:
    """Static metadata for a foundation-model forecaster family."""

    family: str
    default_model_path: str | None
    dependency_group: str
    supports_zero_shot: bool
    supports_pretrain: bool
    supports_fit_fine_tune: bool
    supports_peft: bool
    supports_quantization: bool
    supports_quantiles: bool
    supports_multivariate: bool
    supports_exogenous: bool
