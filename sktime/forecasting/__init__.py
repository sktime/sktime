"""Forecasting models."""

import sys
import warnings
from importlib import import_module
from types import ModuleType

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
    "timesfm2_forecaster": "timesfm2",
}
# TODO 2.0.0: remove deprecation and aliasing logic in 2.0 release
# imports do not need to be updated in the codebase


class _LazyAliasModule(ModuleType):
    """Lazy stand-in for a deprecated ``sktime.forecasting`` submodule alias.

    Registered in ``sys.modules`` so that ``import sktime.forecasting.<old_name>``
    keeps working, without eagerly importing the (possibly heavy, e.g. torch-based)
    module it aliases. The real module -- and its deprecation warning -- is only
    loaded on first attribute access.
    """

    def __init__(self, alias_name, target_name, package):
        super().__init__(f"{package}.{alias_name}")
        self._sktime_alias_name = alias_name
        self._sktime_target_name = target_name
        self._sktime_package = package

    def _sktime_load(self):
        warnings.warn(
            f"{self._sktime_package}.{self._sktime_alias_name} is deprecated and "
            f"has been renamed to "
            f"{self._sktime_package}.{self._sktime_target_name}; "
            "please update your imports.",
            FutureWarning,
            stacklevel=3,
        )
        real_module = import_module(
            f".{self._sktime_target_name}", self._sktime_package
        )
        sys.modules[self.__name__] = real_module
        return real_module

    def __getattr__(self, item):
        return getattr(self._sktime_load(), item)


for _module, _new_name in _MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{_module}"] = _LazyAliasModule(
        _module, _new_name, __name__
    )


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
