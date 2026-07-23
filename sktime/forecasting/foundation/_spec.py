"""Declarative model identity, loading, and inference settings."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class FoundationModelSpec:
    """Shared loading and runtime settings for a foundation-model forecaster.

    Concrete forecasters construct this specification from their public
    constructor parameters and pass it to ``BaseFoundationForecaster``. The
    specification is frozen so top-level settings cannot drift after they have
    been used to identify a cached model. Extra keyword mappings are deep-copied
    to isolate them from caller mutation.

    Parameters
    ----------
    model_path : str or None, default=None
        Repository identifier, checkpoint name, or local model path.
    tokenizer_path : str or None, default=None
        Separate tokenizer identifier or path, if the model family uses one.
    revision : str or None, default=None
        Model repository revision, branch, tag, or commit.
    config : Any or None, default=None
        Load-affecting backend configuration. The base class makes a deep
        runtime copy by default and includes it in the model cache key. Adapters
        may override ``_resolve_config`` only for configuration objects that
        cannot be deep-copied. Leave this field as ``None`` when a similarly named
        public constructor parameter contains prediction-only options; place
        those resolved options in ``predict_extra_kwargs`` instead. The base
        resolver normalizes ``None`` to an empty runtime dictionary.
    device : Any or None, default=None
        Device or backend selector. The string ``"auto"`` invokes the base Torch
        CUDA/MPS/CPU resolution policy.
    dtype : Any or None, default=None
        Native dtype object or supported serialized dtype name. The base class
        resolves any valid ``"torch.<dtype_name>"`` string, such as
        ``"torch.float32"`` or ``"torch.bfloat16"``.
    quantization_config : Any or None, default=None
        Backend quantization settings used while loading the model.
    random_state : int, RandomState, or None, default=None
        sklearn-compatible seed input. It is normalized to one integer and used
        inside the local Torch inference context; it does not identify cached
        model weights.
    ignore_deps : bool, default=False
        Whether to clear the estimator's soft-dependency tags. This is intended
        for controlled environments and tests where dependencies are handled
        externally.
    load_extra_kwargs : Mapping, default={}
        Model-family-specific options that affect ``_load_model``. Do not repeat
        standard fields such as ``device`` here. Loading extras participate in
        the default cache key.
    predict_extra_kwargs : Mapping, default={}
        Model-family-specific options used by ``_inference``. Prediction-only
        options do not participate in the default model cache key.

    Notes
    -----
    ``frozen=True`` prevents attribute reassignment but does not make objects
    stored in standard fields recursively immutable. Adapters should treat the
    specification as read-only and use ``BaseFoundationForecaster._update_model_spec``
    to replace the active specification when necessary.
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
        """Validate extras and isolate them from caller-owned mappings.

        Standard field names are rejected in the extra mappings to keep model
        identity and backend-only options unambiguous.
        """
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
