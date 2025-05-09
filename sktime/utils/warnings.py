"""Warning related utilities."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import warnings
from warnings import warn as _warn

__author__ = ["fkiraly"]


def warn(msg, category=None, obj=None, stacklevel=2):
    """Warn if obj has warnings turned on, otherwise not.

    Wraps ``warnings.warn`` in a conditional based on ``obj``.
    If ``obj.get_config()["warnings"] == "on"``, warns, otherwise not.

    Developer note: this is for configurable user warnings only.
    Deprecation warnings must always be raised.

    Parameters
    ----------
    msg : str
        warning message, passed on to ``warnings.warn``
    category : optional, warning class
        class of the warning to be raised
    obj : optional, any sktime object - cannot be class
    stacklevel : int
        stack level, passed on to ``warnings.warn``

    Returns
    -------
    is_sklearn_est : bool, whether obj is an sklearn estimator (class or instance)
    """
    if obj is None or not hasattr(obj, "get_config"):
        return _warn(msg, category=category, stacklevel=stacklevel)

    warn_on = obj.get_config()["warnings"] == "on"
    if warn_on:
        return _warn(msg, category=category, stacklevel=stacklevel)
    else:
        return None


class _SuppressWarningPattern:
    """Context manager to suppress warnings of a given type and message pattern.

    Parameters
    ----------
    warning_type : type, warning class, e.g., FutureWarning
        type of the warning
    message_pattern : str, regex pattern
        pattern to match the warning message
    """

    def __init__(self, warning_type, message_pattern):
        self.warning_type = warning_type
        self.message_pattern = message_pattern

    def __enter__(self):
        warnings.filterwarnings(
            "ignore", category=self.warning_type, message=self.message_pattern
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filterwarnings("default", category=self.warning_type)


def _suppress_pd22_warning():
    return _SuppressWarningPattern(
        FutureWarning,
        r"'[A-Z]+' is deprecated and will be removed in a future version, please use '[A-Z]+' instead",  # noqa
    )
