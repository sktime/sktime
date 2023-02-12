# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements framework for applying online ensembling algorithms to forecasters."""

__author__ = ["magittan, mloning"]

import numpy as np
import pandas as pd

from sktime.forecasting.compose._ensemble import EnsembleForecaster


class OnlineEnsembleForecaster(EnsembleForecaster):
    """Online Updating Ensemble of forecasters.

    Parameters
    ----------
    ensemble_algorithm : ensemble algorithm
    forecasters : list of (str, estimator) tuples
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    """

    _tags = {
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "y_inner_mtype": ["pd.Series"],
        "scitype:y": "univariate",
    }

    def __init__(self, forecasters, ensemble_algorithm=None, n_jobs=None):

        self.n_jobs = n_jobs
        self.ensemble_algorithm = ensemble_algorithm

        super(EnsembleForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        names, forecasters = self._check_forecasters()
        self.weights = np.ones(len(forecasters)) / len(forecasters)
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _fit_ensemble(self, y, X=None):
        """Fit the ensemble.

        This makes predictions with individual forecasters and compares the
        results to actual values. This is then used to update ensemble
        weights.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        """
        fh = np.arange(len(y)) + 1
        estimator_predictions = np.column_stack(self._predict_forecasters(fh, X))
        y = np.array(y)

        self.ensemble_algorithm.update(estimator_predictions.T, y)

    def _update(self, y, X=None, update_params=False):
        """Update fitted paramters and performs a new ensemble fit.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        if len(y) >= 1 and self.ensemble_algorithm is not None:
            self._fit_ensemble(y, X)

        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)

        return self

    def _predict(self, fh=None, X=None):
        if self.ensemble_algorithm is not None:
            self.weights = self.ensemble_algorithm.weights
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1) * self.weights
        y_pred = y_pred.sum(axis=1)
        y_pred.name = self._y.name
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}
        return params
