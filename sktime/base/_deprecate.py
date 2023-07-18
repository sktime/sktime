"""Module for the current deprecations.

It contains the decorators for the upcoming deprecations.
"""


def _deprecate_util_loads(func):
    """Deprecate `load_from_serial()` and `load_from_path()` in favor of `load()`."""
    from functools import wraps
    from warnings import warn

    @wraps(func)
    def wrapper(*args, **kwargs):
        warn(
            f"{func.__name__}` will be deprecated in 0.20.2 in the favor of load() "
            "which is the standard way of deserializing estimators. "
            "This was called from a centralized decorator.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper
