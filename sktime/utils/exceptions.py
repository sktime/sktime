#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"
__all__ = ["NotEvaluatedError", "NotFittedError"]


class NotEvaluatedError(ValueError, AttributeError):
    """Exception class to raise if evaluator is used before having
    evaluated any metric.
    """


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    References
    ----------
    ..[1] Based on scikit-learn's NotFittedError
    """