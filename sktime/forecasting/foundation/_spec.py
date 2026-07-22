"""Foundation-model runtime specification."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class FoundationModelSpec:
    """Shared loading and runtime settings for a foundation-model forecaster.

    Concrete forecasters construct this specification from their public
    constructor parameters and pass it to ``BaseFoundationForecaster``.
    """

    model_path: str | None = None
    tokenizer_path: str | None = None
    revision: str | None = None
    config: Any | None = None
    device: Any | None = None
    dtype: Any | None = None
    quantization_config: Any | None = None
    random_state: Any | None = None
    ignore_deps: bool = False
    load_extra_kwargs: Mapping[str, Any] = field(default_factory=dict)
    predict_extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Keep model-specific keyword arguments isolated from caller mutation."""
        standard_fields = {
            item.name
            for item in fields(self)
            if not item.name.endswith("_extra_kwargs")
        }
        for name in ("load_extra_kwargs", "predict_extra_kwargs"):
            extra_kwargs = getattr(self, name)
            duplicates = standard_fields.intersection(extra_kwargs)
            if duplicates:
                duplicate_names = ", ".join(sorted(duplicates))
                raise ValueError(
                    f"{name} must contain only model-specific extras; "
                    f"use the explicit FoundationModelSpec field(s) instead: "
                    f"{duplicate_names}."
                )

        object.__setattr__(
            self, "load_extra_kwargs", deepcopy(dict(self.load_extra_kwargs))
        )
        object.__setattr__(
            self, "predict_extra_kwargs", deepcopy(dict(self.predict_extra_kwargs))
        )
