#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["BaseEstimator"]

import inspect

from sklearn import clone
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.ensemble._base import _set_random_states

from sktime.exceptions import NotFittedError


class BaseEstimator(_BaseEstimator):
    """Base class for defining estimators in sktime.

    Extends scikit-learn's BaseEstimator.
    """

    def __init__(self):
        self._is_fitted = False

    @property
    def is_fitted(self):
        """Whether `fit` has been called."""
        return self._is_fitted

    def check_is_fitted(self):
        """Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

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


def _clone_estimator(base_estimator, random_state=None):
    estimator = clone(base_estimator)

    if random_state is not None:
        _set_random_states(estimator, random_state)

    return estimator
