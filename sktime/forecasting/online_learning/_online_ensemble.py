# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements framework for applying online ensembling algorithms to forecasters."""

__author__ = ["magittan", "mloning"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.compose._ensemble import EnsembleForecaster


class OnlineEnsembleForecaster(EnsembleForecaster):
    """Online Updating Ensemble of forecasters.

    Parameters
    ----------
    ensemble_algorithm : ensemble algorithm

    forecasters : list of estimator, (str, estimator), or (str, estimator, count) tuples
        Estimators to apply to the input series.

        * (str, estimator) tuples: the string is a name for the estimator.
        * estimator without string will be assigned unique name based on class name
        * (str, estimator, count) tuples: the estimator will be replicated count times.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.

    Attributes
    ----------
    ensemble_algorithm_ : ensemble algorithm or None
        copy of ``ensemble_algorithm`` made in ``fit``, whose weights are
        updated in ``update``; ``ensemble_algorithm`` itself is not changed.
    weights : np.ndarray
        current ensemble weights of the forecasters, uniform after ``fit``
        (or the initial weights of ``ensemble_algorithm``), updated in ``update``

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.online_learning import (
    ...     NormalHedgeEnsemble,
    ...     OnlineEnsembleForecaster,
    ... )
    >>> y = load_airline()
    >>> forecaster = OnlineEnsembleForecaster(
    ...     [
    ...         ("last", NaiveForecaster()),
    ...         ("mean", NaiveForecaster(strategy="mean", window_length=12)),
    ...     ],
    ...     ensemble_algorithm=NormalHedgeEnsemble(
    ...         n_estimators=2, loss_func=mean_squared_error
    ...     ),
    ... )
    >>> forecaster.fit(y[:120], fh=[1, 2, 3])
    OnlineEnsembleForecaster(...)
    >>> forecaster.update(y[120:132])
    OnlineEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["magittan", "mloning"],
        "maintainers": ["magittan"],
        # estimator type
        # --------------
        "category": ["composition"],
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "y_inner_mtype": ["pd.Series"],
        "capability:multivariate": False,
        "capability:update": True,
    }

    def __init__(self, forecasters, ensemble_algorithm=None, n_jobs=None):
        self.n_jobs = n_jobs
        self.ensemble_algorithm = ensemble_algorithm

        super().__init__(forecasters=forecasters, n_jobs=n_jobs)

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self._cur_y = y
        self._cur_X = X
        forecasters = [x[1] for x in self.forecasters_]
        self.weights = np.ones(len(forecasters)) / len(forecasters)

        # the ensemble algorithm is stateful, its weights are updated in update,
        # so we work on a copy, to leave the parameter unchanged,
        # and to start from the initial weights in every fit
        if self.ensemble_algorithm is not None:
            self.ensemble_algorithm_ = deepcopy(self.ensemble_algorithm)
            self.weights = np.array(self.ensemble_algorithm_.weights, copy=True)
        else:
            self.ensemble_algorithm_ = None

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

        self.ensemble_algorithm_.update(estimator_predictions.T, y)
        self.weights = np.array(self.ensemble_algorithm_.weights, copy=True)

    def _update(self, y, X=None, update_params=False):
        """Update fitted parameters and performs a new ensemble fit.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        if len(y) >= 1 and self.ensemble_algorithm_ is not None:
            self._fit_ensemble(y, X)

        for forecaster in self._get_forecaster_list():
            forecaster.update(y, X, update_params=update_params)

        return self

    def _predict(self, fh=None, X=None):
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1) * self.weights
        y_pred = y_pred.sum(axis=1)
        y_pred.name = self._cur_y.name
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
        from sklearn.metrics import mean_squared_error

        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.online_learning._prediction_weighted_ensembler import (
            NormalHedgeEnsemble,
        )

        FORECASTER = NaiveForecaster()
        params0 = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}
        params1 = {
            "forecasters": [
                ("last", NaiveForecaster()),
                ("mean", NaiveForecaster(strategy="mean", window_length=3)),
            ],
            "ensemble_algorithm": NormalHedgeEnsemble(
                n_estimators=2, loss_func=mean_squared_error
            ),
        }
        return [params0, params1]
