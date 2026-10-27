"""Time series anomaly, changepoint detection, segmentation."""
try:
    from .evaluation import evaluate_detection
except Exception:
    # Avoid breaking docs build due to optional dependencies
    evaluate_detection = None

import sys
import warnings
from importlib import import_module

# alias dictionary to map old module names to new module names
# if an old module name is queried, imports work, but
# a deprecation warning is issued to update the import statement
# .deprecated_name.abc -> .abc
_MODULES_FLATTENED = ["skchange_aseg", "skchange_cp"]
# TODO 2.0.0: remove deprecation and aliasing logic in 2.0 release
# imports do not need to be updated in the codebase


def __getattr__(name):
    if name in _MODULES_FLATTENED:
        warnings.warn(
            f"{__name__}.{name} is deprecated, please import directly from "
            f"{__name__} submodules instead, alternatively from {__name__}.all -"
            "please see the documentation for exact import locations of estimators",
            FutureWarning,
            stacklevel=2,
        )

        return import_module(__name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
