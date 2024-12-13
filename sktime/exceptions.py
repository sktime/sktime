#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Custom exceptions and warnings."""

__author__ = ["mloning"]
__all__ = ["NotEvaluatedError", "NotFittedError", "FitFailedWarning"]

from skbase._exceptions import NotFittedError


class NotEvaluatedError(ValueError, AttributeError):
    """NotEvaluatedError.

    Exception class to raise if evaluator is used before having evaluated any metric.
    """


class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    FitFailedWarning('Estimator fit failed. The score on this train-test
    partition for these parameters will be set to 0.000000').

    References
    ----------
    .. [1] Based on scikit-learn's FitFailedWarning
    """
