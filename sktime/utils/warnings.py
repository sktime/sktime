"""Warning related utilities."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

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
