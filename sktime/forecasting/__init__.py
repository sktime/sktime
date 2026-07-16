"""Forecasting models."""

import sys
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
    "hf_transformers_forecaster": "hf_transformers",
    "hf_moirai_forecaster": "moirai",
    "pykan_forecaster": "pykan",
    "rbf_forecaster": "rbf",
    "timesfm_forecaster": "timesfm",
}
# TODO 2.0.0: remove deprecation and aliasing logic in 2.0 release
# imports do not need to be updated in the codebase


for _module, _new_name in _MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{_module}"] = import_module(f".{_new_name}", __name__)


def __getattr__(name):
    if name in _MODULE_ALIASES:
        new_name = _MODULE_ALIASES[name]

        warnings.warn(
            f"{__name__}.{name} is deprecated and has been renamed to "
            f"{__name__}.{new_name}; please update your imports.",
            FutureWarning,
            stacklevel=2,
        )

        return import_module(f".{_MODULE_ALIASES[name]}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
