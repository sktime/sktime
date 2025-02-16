"""Utilities to safely call functions with varying signature."""

__author__ = ["fkiraly"]


from inspect import signature


def _safe_call(method, args, kwargs):
    """Call a method with arguments and keyword arguments.

    Same as calling ``method(*args, **kwargs)`` but not passing keyword arguments
    in ``kwargs`` that are not in the signature of ``method``.

    Parameters
    ----------
    method : callable
        method to call
    args : tuple
        positional arguments to pass to the method
    kwargs : dict
        keyword arguments to pass to the method
    """
    safe_kwargs = {k: v for k, v in kwargs.items() if _method_has_arg(method, arg=k)}
    return method(*args, **safe_kwargs)


def _method_has_arg(method, arg="y"):
    """Return if method has a parameter.

    Parameters
    ----------
    method : callable
        method to check
    arg : str, optional, default="y"
        parameter name to check

    Returns
    -------
    has_param : bool
        whether the method ``method`` has a parameter with name ``arg``
    """
    method_params = list(signature(method).parameters.keys())
    return arg in method_params


def _method_has_param_and_default(method, arg="y"):
    """Return if transformer.method has a parameter, and whether it has a default.

    Parameters
    ----------
    method : callable
        method to check
    arg : str, optional, default="y"
        parameter name to check

    Returns
    -------
    has_param : bool
        whether the method ``method`` has a parameter with name ``arg``
    has_default : bool
        whether the parameter ``arg`` of method ``method`` has a default value
    """
    has_param = _method_has_arg(method=method, arg=arg)
    if has_param:
        param = signature(method).parameters[arg]
        default = param.default
        has_default = default is not param.empty
        return True, has_default
    else:
        return False, False
