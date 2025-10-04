"""Numba optional backend warning."""

from inspect import isclass

from sktime.utils.validation._dependencies import _check_soft_dependencies


def _check_numba_warning(obj=None, category=None, stacklevel=1):
    from sktime.utils.warnings import warn

    if obj is None:
        obj_name = "This feature"
    elif isclass(obj):
        obj_name = obj.__name__
    elif isclass(type(obj)):
        obj_name = type(obj).__name__
    else:
        obj_name = str(obj)

    numba_available = _check_soft_dependencies("numba", severity="none")

    if not numba_available:
        warn(
            f"{obj_name} uses numba as an optional backend, but numba was not found "
            "in the python environment. Without numba, performance will be suboptimal.",
            category=category,
            stacklevel=stacklevel,
            obj=obj,
        )
