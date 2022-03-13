# -*- coding: utf-8 -*-
"""Base utils and classes for Mock Estimators."""

from copy import deepcopy
from inspect import getfullargspec


class _MockEstimatorMixin:
    """Mixin class for constructing Mock estimators."""

    def __init__(self):
        self._log = []

    @property
    def log(self):
        """Log of the methods called and the parameters passed in each method."""
        return self._log


def _method_logger(method):
    """Log the method and it's arguments."""

    def wrapper(self, *args, **kwargs):
        arg_spec = getfullargspec(method)
        inputs_dict = dict(zip(arg_spec.args, deepcopy(args)))
        inputs_dict.update(deepcopy(kwargs))
        self._log.append((method.__name__, inputs_dict))
        return method(self, *args, **kwargs)

    return wrapper
