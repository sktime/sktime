"""Forecasting models."""

import sys
import warnings
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

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


class _AliasLoader(Loader):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def create_module(self, spec):
        warnings.warn(
            f"sktime.forecasting.{self.old_name} is deprecated and has been renamed to "
            f"sktime.forecasting.{self.new_name}; please update your imports.",
            FutureWarning,
            stacklevel=2,
        )
        mod = import_module(f".{self.new_name}", "sktime.forecasting")
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("sktime.forecasting."):
            submod = fullname.split(".")[-1]
            if submod in _MODULE_ALIASES:
                return ModuleSpec(
                    fullname, _AliasLoader(submod, _MODULE_ALIASES[submod])
                )
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


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
