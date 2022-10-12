# -*- coding: utf-8 -*-
"""Base utils and classes for Mock Estimators."""

__author__ = ["ltsaprounis"]

import re
from copy import deepcopy
from functools import wraps
from inspect import getcallargs, getfullargspec

from sktime.base import BaseEstimator


class _MockEstimatorMixin:
    """Mixin class for constructing Mock estimators."""

    @property
    def log(self):
        """Log of the methods called and the parameters passed in each method."""
        if not hasattr(self, "_MockEstimatorMixin__log"):
            return []
        else:
            return self._MockEstimatorMixin__log

    def add_log_item(self, value):
        """Append an item to the log.

        State change:
        self.log - `value` is appended to the list self.log

        Parameters
        ----------
        value : any object
        """
        if not hasattr(self, "_MockEstimatorMixin__log"):
            self._MockEstimatorMixin__log = [value]
        else:
            self._MockEstimatorMixin__log = self._MockEstimatorMixin__log + [value]


def _method_logger(method):
    """Log the method and it's arguments."""

    @wraps(wrapped=method)
    def wrapper(self, *args, **kwargs):
        args_dict = getcallargs(method, self, *args, **kwargs)
        if not isinstance(self, _MockEstimatorMixin):
            raise TypeError("method_logger requires a MockEstimator class")
        args_dict.pop("self")
        self.add_log_item((method.__name__, deepcopy(args_dict)))
        return method(self, *args, **kwargs)

    return wrapper


def make_mock_estimator(
    estimator_class: BaseEstimator, method_regex: str = ".*"
) -> BaseEstimator:
    r"""Transform any estimator class into a mock estimator class.

    The returned class will accept the original arguments passed in estimator_class
    __init__ as a dictionary of kwargs.

    Parameters
    ----------
    estimator_class : BaseEstimator
        any sktime estimator
    method_regex : str, optional
        regex to filter methods on, by default ".*"
        Useful regex examples:
            - everything: '.*'
            - private methods only: '^(?!^__\w+__$)^_\w'
            - public methods only: '(?!^_\w+)'

    Returns
    -------
    BaseEstimator
        input estimator class with logging feature enabled

    Examples
    --------
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.utils.estimators import make_mock_estimator
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> mock_estimator_class = make_mock_estimator(NaiveForecaster)
    >>> mock_estimator_instance = mock_estimator_class({"strategy": "last", "sp": 1})
    >>> mock_estimator_instance.fit(y)
    _MockEstimator(...)

    """
    dunder_methods_regex = r"^__\w+__$"

    class _MockEstimator(estimator_class, _MockEstimatorMixin):
        def __init__(self, estimator_kwargs=None):
            self.estimator_kwargs = estimator_kwargs
            if estimator_kwargs is not None:
                super().__init__(**estimator_kwargs)
            else:
                super().__init__()

    for attr_name in dir(estimator_class):
        attr = getattr(_MockEstimator, attr_name)
        # exclude dunder methods (e.g. __eq__, __class__ etc.) and non callables
        # from logging
        if not re.match(dunder_methods_regex, attr_name) and callable(attr):
            # match the given regex pattern
            # exclude static and class methods from logging
            if (
                re.match(method_regex, attr_name)
                and "self" in getfullargspec(attr).args
            ):
                setattr(_MockEstimator, attr_name, _method_logger(attr))

    return _MockEstimator
