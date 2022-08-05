#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements polymorphic base class."""

__author__ = ["fkiraly", "benheid", "miraep8"]
__all__ = ["BasePolymorph"]

from sktime.base import BaseObject


class BasePolymorph(BaseObject):
    """Handles parameter management for estimators composed of named estimators.

    Partly adapted from sklearn utils.metaestimator.py.
    """

    def __new__(cls, *args, estimator_type="base", **kwargs):
        """Polymorphic dispatcher to all sktime base classes."""
        from sktime.registry._lookup import _check_estimator_types

        baseclass = _check_estimator_types(estimator_type)[0]
        obj = baseclass(*args, **kwargs)

        return obj

    def __init__(self, estimator_type="base"):

        self.estimator_type = estimator_type
        super().__init__()
