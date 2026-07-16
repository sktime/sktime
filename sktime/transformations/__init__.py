"""Transformations."""

import sys
import warnings
from importlib import import_module

# alias dictionary to map old module names to new module names
# if an old module name is queried, imports work, but
# a deprecation warning is issued to update the import statement
# .deprecated_name.abc -> .abc
_MODULES_FLATTENED = ["series", "panel"]
# TODO 2.0.0: remove deprecation and aliasing logic in 2.0 release
# imports do not need to be updated in the codebase


for _module in _MODULES_FLATTENED:
    sys.modules[f"{__name__}.{_module}"] = import_module(__name__)

sys.modules[f"{__name__}._delegate"] = import_module(f"{__name__}.base._delegate")


def __getattr__(name):
    if name in _MODULES_FLATTENED:
        warnings.warn(
            f"{__name__}.{name} is deprecated, please import directly from "
            f"{__name__} instead. Same for deeper imports, e.g., "
            f"an import from {__name__}.{name}.abc should be replaced by import from "
            f"{__name__}.abc instead.",
            FutureWarning,
            stacklevel=2,
        )

        return import_module(__name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
