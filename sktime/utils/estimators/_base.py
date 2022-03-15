# -*- coding: utf-8 -*-
"""Base utils and classes for Mock Estimators."""

__author__ = ["ltsaprounis"]

from copy import deepcopy
from inspect import getcallargs


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
        args_dict = getcallargs(method, self, *args, **kwargs)
        if not isinstance(self, _MockEstimatorMixin):
            raise TypeError("method_logger requires a MockEstimator class")
        args_dict.pop("self")
        self._log.append((method.__name__, deepcopy(args_dict)))
        return method(self, *args, **kwargs)

    return wrapper
