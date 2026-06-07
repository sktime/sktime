"""Forecasting models."""


# mypackage/__init__.py

from importlib import import_module
import warnings

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
            f"mypackage.{name} is deprecated and has been renamed to "
            f"mypackage.{new_name}; please update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )

        return import_module(f".{_MODULE_ALIASES[name]}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
