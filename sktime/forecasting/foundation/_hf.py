"""Private Hugging Face helpers for foundation-model forecasters."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from sktime.forecasting.foundation._cache import _stable_repr


def stable_quantization_key(quantization_config) -> tuple:
    """Return a stable cache-key component for quantization config."""
    if quantization_config is None:
        return ()
    to_dict = getattr(quantization_config, "to_dict", None)
    if callable(to_dict):
        return ("dict", _stable_repr(to_dict()))
    return ("repr", repr(quantization_config))


def stable_peft_key(peft) -> tuple:
    """Return a stable cache-key component for PEFT config."""
    if peft is None or peft.kind == "none":
        return ()
    peft_config = peft.peft_config
    to_dict = getattr(peft_config, "to_dict", None)
    if callable(to_dict):
        peft_config = to_dict()
    return (
        peft.kind,
        peft.peft_model_id,
        peft.merge_after_train,
        _stable_repr(peft_config),
    )


def load_hf_model(
    *,
    model_cls,
    config_cls=None,
    load,
    config_overrides: Mapping[str, Any] | None = None,
):
    """Load or initialize a Hugging Face-style model."""
    config = config_overrides
    if isinstance(config, dict) and config_cls is not None:
        config = config_cls.from_dict(config)
    elif config is None and load.model_path is None and config_cls is not None:
        config = config_cls()

    if load.model_path is not None:
        kwargs = {
            "revision": load.revision,
            "cache_dir": load.cache_dir,
            "local_files_only": load.local_files_only,
            "token": load.token,
            "device_map": load.device_map,
            "quantization_config": load.quantization_config,
            "trust_remote_code": load.trust_remote_code,
        }
        if load.dtype is not None:
            kwargs["dtype"] = load.dtype
        if config is not None:
            kwargs["config"] = config
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        kwargs.update(load.extra_load_kwargs)
        model = model_cls.from_pretrained(load.model_path, **kwargs)
        return model, getattr(model, "config", config)

    model = model_cls(config)
    model = move_torch_model(model, device_map=load.device_map, dtype=load.dtype)
    return model, config


def apply_peft_if_requested(model, tune):
    """Apply PEFT wrapping when requested."""
    peft = tune.parameter_efficient
    if peft.kind == "none":
        return model
    if peft.kind == "peft":
        if peft.peft_model_id is not None:
            from peft import PeftModel

            return PeftModel.from_pretrained(model, peft.peft_model_id)
        from peft import get_peft_model

        return get_peft_model(model, deepcopy(peft.peft_config))
    raise NotImplementedError(f"Unsupported PEFT strategy: {peft.kind}")


def move_torch_model(model, *, device_map=None, dtype=None):
    """Move torch-style models to device and dtype when supported."""
    if device_map is not None and hasattr(model, "to"):
        model = model.to(device_map)
    if dtype is not None and hasattr(model, "to"):
        model = model.to(dtype=dtype)
    return model
