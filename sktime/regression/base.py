#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base class for time series regression estimators in sktime."""

__author__ = ["Markus Löning"]
__all__ = ["BaseRegressor"]

from sktime.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """Base class for regressors, for identification."""

    def fit(self, X, y):
        """Fit regressor to training data.

        Parameters
        ----------
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        y : pd.Series, pd.DataFrame, or np.array
            Target time series to which to fit the regressor.

        Returns
        -------
        self :
            Reference to self.
        """
        raise NotImplementedError("abstract method")

    def predict(self, X):
        """Predict time series.

        Parameters
        ----------
        X : pd.DataFrame, shape=[n_obs, n_vars]
            A2-d dataframe of exogenous variables.

        Returns
        -------
        y_pred : pd.Series
            Regression predictions.
        """
        raise NotImplementedError("abstract method")

    def score(self, X, y):
        """Scores regression against ground truth, R-squared.

        Parameters
        ----------
        X : pd.DataFrame, shape=[n_obs, n_vars]
            A2-d dataframe of exogenous variables.
        y : pd.Series
            Target time series to which to compare the predictions.

        Returns
        -------
        score : float
            R-squared score.
        """
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))
