"""Warning related utilities."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import re
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
        self.message_pattern = re.compile(message_pattern)

    def __enter__(self):
        self.original_filters = warnings.filters[:]
        warnings.simplefilter("default", self.warning_type)
        self.original_showwarning = warnings.showwarning
        warnings.showwarning = self._custom_showwarning

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters = self.original_filters
        warnings.showwarning = self.original_showwarning

    def _custom_showwarning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        right_type = issubclass(category, self.warning_type)
        fits_pattern = self.message_pattern.search(str(message))
        if not (right_type and fits_pattern):
            self.original_showwarning(message, category, filename, lineno, file, line)


_suppress_pd22_warning = _SuppressWarningPattern(
    FutureWarning,
    r"'[A-Z]+' is deprecated and will be removed in a future version, please use '[A-Z]+' instead",  # noqa
)
