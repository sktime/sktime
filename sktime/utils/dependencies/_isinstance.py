"""Utility to check isinstance by name rather than import, for dependency isolation."""

import inspect


def _isinstance_by_name(obj, cls_name):
    """Check if object is an instance of class by class name.

    This is useful as a pre-check before importing a class to isolate dependencies.

    A useful pattern is calling ``_isinstance_by_name`` before
    import and call of ``isinstance``, in a case where a ``True`` result
    is rare, to avoid importing the dependency when it is not needed.

    For instance, to check for ``polars.DataFrame`` without importing polars:

    .. code-block:: python
        def _is_polars_df(obj):

            if _isinstance_by_name(obj, "DataFrame"):
                import polars as pl

                return isinstance(obj, pl.DataFrame)
            else:
                return False

    Parameters
    ----------
    obj : object
        object to check
    cls_name : str
        name of class to check, e.g., "DataFrame"

    Returns
    -------
    bool
        whether object is an instance of class by name
    """
    if not isinstance(cls_name, str):
        raise TypeError(f"class_name must be a string, not {type(cls_name)}")

    cls = type(obj)

    return any(base.__name__ == cls_name for base in inspect.getmro(cls))
