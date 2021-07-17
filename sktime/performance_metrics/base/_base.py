#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ryan Kuhns"]
__all__ = ["BaseMetric"]

from sktime.base import BaseObject


class BaseMetric(BaseObject):
    """Base class for defining metrics in sktime.

    Parameters
    ----------
    func : function
        Function that implements the aggregate metric.
    name : str, default=None
        Optional name to pass to metric's name attribute. If None, then the
        function's name is used.
    """

    def __init__(self, func, sample_func=None, name=None):
        self._func = func
        self.name = name if name is not None else func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function."""
        raise NotImplementedError("abstract method")
