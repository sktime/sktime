#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base class for defining performance metric in sktime."""

__author__ = ["Ryan Kuhns"]
__all__ = ["BaseMetric"]

from sktime.base import BaseObject


class BaseMetric(BaseObject):
    """Base class for defining metrics in sktime.

    Extends sktime BaseObject.
    """

    def __init__(self, func, name=None):
        self.func = func
        self.name = name if name is not None else func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function."""
        NotImplementedError("abstract method")
