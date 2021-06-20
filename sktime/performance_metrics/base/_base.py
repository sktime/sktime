#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ryan Kuhns"]
__all__ = ["BaseMetric"]

import inspect
from sklearn.base import BaseEstimator


class BaseMetric(BaseEstimator):
    """Base class for defining metrics in sktime.

    Extends scikit-learn's BaseEstimator.
    """

    def __init__(self, func, name=None):
        self._func = func
        self.name = name if name is not None else func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function."""
        NotImplementedError("abstract method")

    # This is copied from sktime.base.BaseEstimator. Choice to copy was made to
    # Avoid the not applicable functionality from BaseEstimator that tripped
    # up unit tests (e.g. is_fitted, check_is_fitted).
    @classmethod
    def _all_tags(cls):
        """Get tags from estimator class and all its parent classes."""
        # We here create a separate estimator tag interface in addition to the one in
        # scikit-learn to make sure we do not interfere with scikit-learn's one
        # when we inherit from scikit-learn classes. We also make estimator tags a
        # class rather than object attribute.
        collected_tags = dict()

        # We exclude the last two parent classes; sklearn.base.BaseEstimator and
        # the basic Python object.
        for parent_class in reversed(inspect.getmro(cls)[:-2]):
            if hasattr(parent_class, "_tags"):
                # Need the if here because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = parent_class._tags
                collected_tags.update(more_tags)

        return collected_tags
