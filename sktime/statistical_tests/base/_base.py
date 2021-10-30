# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for statistical tests in sktime."""

import pandas as pd

from sktime.base import BaseEstimator

__author__ = ["RNKuhns"]
__all__ = ["BaseStatisticalTest"]


class BaseStatisticalTest(BaseEstimator):
    """Base class for diagnostic and post-hoc statistical tests."""

    def __init__(self, hypothesis=None, report_detail=True):
        self.hypothesis = hypothesis
        self.report_detail = report_detail
        self.p_value_ = None
        self.test_statistic_ = None
        self.reject_null_ = None

    def _fit(self, Y, X=None):
        """Logic to fit each test.

        Returns
        -------
        self :
            A reference to self.
        """
        raise NotImplementedError("Abstract method")

    def fit(self, Y, X=None):
        """Fit the statistical test.

        Returns
        -------
        self :
            A reference to self.
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False
        # Input checks, etc happen above

        # Call internal fitting logic
        self._fit(Y, X=X)

        # Mark test as being fitted if self._fit() completes with out error
        self._is_fitted = True
        return self

    def report(self):
        """Return the fitted test results.

        Returns
        -------
        results : bool or pd.DataFrame
            If `self.report_detail` is False then boolean of whether hypothesis
            was rejected or not is reported. If `self.report_detail` is True then a
            DataFrame with the following columns is returned:

                - hypothesis : str
                    Description of the test's hypothesis.
                - p_value : float or None
                    P-value associated with statistical test. If no p-value is
                    available for a test then will return None.
                - test_statistic : float or None
                    Test statistic from the statistical test. If no test
                    statistic is available for a test then will return None.
                - reject_null : bool
                    Whether the Test's Null Hypothesis was rejected.
        """
        self.check_is_fitted()
        if self.report_detail:
            # return self.p_value, self.test_statistic, self.reject_null
            result_dict = {
                "hypothesis": self.hypothesis,
                "p_value": self.p_value_,
                "test_statistic": self.test_statistic_,
                "reject_null": self.reject_null_,
            }
            results = pd.DataFrame(result_dict)
            return results
        else:
            return self.reject_null

    def fit_report(self, Y, X=None):
        """Fit the test and report the result."""
        return self.fit(Y, X=X).report()

    def print_results(self):
        """Print the results of the fitted test."""
        self.check_is_fitted()
        return None
