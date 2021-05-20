#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["BaseEstimator"]

import inspect

from sklearn.base import BaseEstimator as _BaseEstimator

from sktime.exceptions import NotFittedError


class BaseObject(_BaseEstimator):
    """Base class for defining other classes in sktime.

    Extends scikit-learn's BaseEstimator to include sktime interface for tags.
    """

    @classmethod
    def _all_tags(cls):
        """Get tags from estimator class and all its parent classes.

        Creates a separate sktime tag interface in addition to the one in
        scikit-learn to make sure it does not interfere with scikit-learn's tag
        interface when inheriting from scikit-learn classes. Sktime's
        estimator tags are class rather than object attribute as in scikit-learn.
        """
        collected_tags = dict()

        # We exclude the last two parent classes: sklearn.base.BaseEstimator and
        # the basic Python object.
        for parent_class in reversed(inspect.getmro(cls)[:-2]):
            if hasattr(parent_class, "_tags"):
                # Need the if here because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = parent_class._tags
                collected_tags.update(more_tags)

        return collected_tags


class BaseEstimator(BaseObject):
    """Base class for defining estimators in sktime.

    Extends sktime's BaseObject to include basic functionality for fitted estimators.
    """

    def __init__(self):
        self._is_fitted = False

    @property
    def is_fitted(self):
        """Wheter `fit` been called."""
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
