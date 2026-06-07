"""Forecasting models."""

import warnings
from importlib import import_module


# alias dictionary to map old module names to new module names
# if an old module name is queried, imports work, but
# a deprecation warning is issued to update the import statement
# old_name -> new_name
_MODULE_ALIASES = {
    "boxcox_bias_adjusted_forecaster": "boxcox_biasadj",
    "conditional_invertible_neural_network": "cinn",
    "hf_momentfm_forecaster": "momentfm",
}


def __getattr__(name):
    if name in _MODULE_ALIASES:
        new_name = _MODULE_ALIASES[name]

        warnings.warn(
            f"sktime.forecasting.{name} is deprecated and has been renamed to "
            f"sktime.forecasting.{new_name}; please update your imports.",
            FutureWarning,
            stacklevel=2,
        )

        return import_module(f".{_MODULE_ALIASES[name]}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
