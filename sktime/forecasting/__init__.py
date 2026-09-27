"""Forecasting models."""

import sys
import warnings
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader

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


def _alias_warning(old_name, new_name):
    """Warn that ``old_name`` is deprecated in favour of ``new_name``."""
    warnings.warn(
        f"{__name__}.{old_name} is deprecated and has been renamed to "
        f"{__name__}.{new_name}; please update your imports.",
        FutureWarning,
        stacklevel=3,
    )


class _AliasLoader(Loader):
    """Loader returning the renamed module for a deprecated alias."""

    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def create_module(self, spec):
        """Return the renamed module, so the alias is the very same object."""
        _alias_warning(self.old_name, self.new_name)
        return import_module(f".{self.new_name}", __name__)

    def exec_module(self, module):
        """Do nothing, the module was already executed on its real name."""


class _AliasFinder(MetaPathFinder):
    """Resolve deprecated ``sktime.forecasting`` submodule names on demand.

    Registering the aliases eagerly imports every renamed module - including
    the deep learning ones - at ``sktime.forecasting`` import time. Resolving
    them lazily instead keeps the aliases importable without that cost.
    """

    def find_spec(self, fullname, path=None, target=None):
        """Return a spec for a deprecated alias, or None if not an alias."""
        prefix = f"{__name__}."
        if not fullname.startswith(prefix):
            return None

        old_name = fullname[len(prefix) :]
        new_name = _MODULE_ALIASES.get(old_name)
        if new_name is None:
            return None

        return spec_from_loader(fullname, _AliasLoader(old_name, new_name))


sys.meta_path.append(_AliasFinder())


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
