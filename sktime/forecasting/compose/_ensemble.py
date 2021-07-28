#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["EnsembleForecaster"]

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.model_selection import SingleWindowSplitter


class AutoEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Automatically find best weights for the ensemble.

    The AutoEnsembleForecaster uses a meta-model (regressor) to calculate the optimal
    weights for ensemble aggregation with mean. The regressor has to be sklearn-like
    and needs to have either a param "feature_importances_" or "coef_".

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    regressor : sklearn-like regressor, optional, default=None
        Used to infer optimal weights from coefficients (linear models) or from
        feature importance scores (decision tree-based models). If None, then
        a GradientBoostingRegressor() is used. The regressor can also be a
        sklearn.Pipeline() object.
    cv : Splitter (e.g. ExpandingWindowSplitter), optional, default=None
        Splitter is used for cross-validation of the weights. For each split,
        there is a training of the regressor on the test data and the weights
        are in the end averaged over all fitted regressors. Default is a
        SingleWindowSplitter()
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.

    Attributes
    ----------
    weights_ : np.array
        The weights based on either regressor.feature_importances_ or
        regressor.coef_ values.
    cv_ : Splitter (e.g. ExpandingWindowSplitter)
        This is exposed here in case cv=None was given, then cv_ is assigned
        by default to cv_=SingleWindowSplitter().
    """

    def __init__(self, forecasters, regressor=None, cv=None, n_jobs=None):
        super(AutoEnsembleForecaster, self).__init__(
            forecasters=forecasters, n_jobs=n_jobs
        )
        self.regressor = regressor
        self.cv = cv

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        # use GradientBoostingRegressor() as default regressor
        self._regressor = (
            self.regressor
            if self.regressor is not None
            else GradientBoostingRegressor()
        )
        # use SingleWindowSplitter() as default cv
        self.cv_ = (
            self.cv
            if self.cv is not None
            else SingleWindowSplitter(fh.to_relative(cutoff=y.index[-1]))
        )
        self.weights_ = None

        names, forecasters = self._check_forecasters()

        # # get training data for meta-model
        # if X is not None:
        #     y_train, y_test, X_train, X_test = temporal_train_test_split(
        # y, X, test_size=self.test_size)
        # else:
        #     y_train, y_test = temporal_train_test_split(y, test_size=self.test_size)
        #     X_train, X_test = None, None

        weight_list = []
        for idx_train, idx_test in self.cv_.split(y):
            # fit ensemble models
            if X is not None:
                self._fit_forecasters(
                    forecasters, y.iloc[idx_train], X.iloc[idx_train], fh
                )
                y_pred = pd.concat(
                    self._predict_forecasters(fh, X.iloc[idx_test]), axis=1
                )
            else:
                self._fit_forecasters(forecasters, y.iloc[idx_train], None, fh)
                y_pred = pd.concat(self._predict_forecasters(fh), axis=1)
            # fit meta-model (regressor) on predictions of ensemble models with
            # y_test as target
            self._regressor.fit(X=y_pred, y=y.iloc[idx_test])
            # check if regressor is a sklearn.Pipeline
            if isinstance(self._regressor, Pipeline):
                # extract regressor from pipeline to access its attributes
                weights = _get_weights(self._regressor.steps[-1][1])
            else:
                weights = _get_weights(self._regressor)
            weight_list.append(weights)

        # use average weights from each cv split
        self.weights_ = np.mean(weight_list, axis=0)

        # fit forecasters with all data
        self._fit_forecasters(forecasters, y, X, fh)

        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame
        return_pred_int : boolean, optional, default=False
        alpha : fh : float, default=DEFAULT_ALPHA

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        if return_pred_int:
            raise NotImplementedError()

        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1)
        # apply weights
        y_pred = y_pred.apply(lambda x: np.average(x, weights=self.weights_), axis=1)
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self.
        """
        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)
        return self


def _get_weights(regressor):
    # tree-based models from sklearn which have feature importance values
    if hasattr(regressor, "feature_importances_"):
        weights = regressor.feature_importances_
    # linear regression models from sklearn which have coefficient values
    elif hasattr(regressor, "coef_"):
        weights = regressor.coef_
    else:
        raise NotImplementedError("The given regressor is not supported.")
    return weights


class EnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Ensemble of forecasters.

    Overview: Input one series of length `n` and EnsembleForecaster performs
    fitting and prediction for each estimator passed in `forecasters`. It then
    applies `aggfunc` aggregation function by row to the predictions dataframe
    and returns final prediction - one series.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    aggfunc : str, {'mean', 'median', 'min', 'max'}, default='mean'
        The function to aggregate prediction from individual forecasters.

    Example
    -------
    >>> from sktime.forecasting.compose import EnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [("trend", PolynomialTrendForecaster()),\
                        ("naive", NaiveForecaster())]
    >>> forecaster = EnsembleForecaster(forecasters=forecasters, n_jobs=2)
    >>> forecaster.fit(y=y, X=None, fh=[1,2,3])
    EnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, n_jobs=None, aggfunc="mean"):
        super(EnsembleForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.aggfunc = aggfunc

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self.
        """
        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame
        return_pred_int : boolean, optional, default=False
        alpha : fh : float, default=DEFAULT_ALPHA

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        if return_pred_int:
            raise NotImplementedError()

        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1)

        valid_aggfuncs = ("median", "mean", "min", "max")
        if self.aggfunc not in valid_aggfuncs:
            raise ValueError(f"Invalid `aggfunc`. Please use one of {valid_aggfuncs}")

        if self.aggfunc == "median":
            return y_pred.median(axis=1)
        elif self.aggfunc == "min":
            return y_pred.min(axis=1)
        elif self.aggfunc == "max":
            return y_pred.max(axis=1)
        else:
            return y_pred.mean(axis=1)
