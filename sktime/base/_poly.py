#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements polymorphic base class."""

__author__ = ["fkiraly", "miraep8"]
__all__ = ["BasePolymorph"]

from sktime.base import BaseObject


class BasePolymorph(BaseObject):
    """Handles parameter management for estimators composed of named estimators.

    Partly adapted from sklearn utils.metaestimator.py.
    """

    def __init__(self, estimator_type="base"):

        from sktime.registry._lookup import _check_estimator_types

        self.estimator_type = estimator_type
        baseclass = _check_estimator_types(estimator_type)[0]

        super(BasePolymorph, self).__init__()

        attr_to_copy = set(dir(baseclass)).difference(dir(self))

        for attr in attr_to_copy:
            setattr(self, attr, getattr(baseclass, attr))

        baseclass.__init__(self)
