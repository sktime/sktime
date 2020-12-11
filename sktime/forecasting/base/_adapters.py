#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning", "Martin Walter"]
__all__ = ["_StatsModelsAdapter"]

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.validation.forecasting import check_X, check_y_X


class _ProphetAdapter(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """Base class for interfacing fbprophet and neuralprophet"""

    def fit(self, y, X=None, fh=None, **fit_params):
        """Fit to training data.
        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        Returns
        -------
        self : returns an instance of self.
        """
        self._instantiate_model()
        self._check_changepoints()
        y, X = check_y_X(y, X, enforce_index_type=pd.DatetimeIndex)
        self._set_y_X(y, X)
        self._set_fh(fh)

        # We have to bring the data into the required format for fbprophet:
        df = pd.DataFrame(y.rename("y"))
        df["ds"] = y.index

        # Add seasonality
        if self.add_seasonality:
            self._forecaster.add_seasonality(**self.add_seasonality)

        # Add country holidays
        if self.add_country_holidays:
            self._forecaster.add_country_holidays(**self.add_country_holidays)

        # Add regressor (multivariate)
        if X is not None:
            df = df.merge(X, left_index=True, right_on=X.index)
            for col in X.columns:
                self._forecaster.add_regressor(col)

        self._forecaster.fit(df=df, **fit_params)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Predict

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional
            Exogenous data, by default None
        return_pred_int : bool, optional
            Returns a pd.DataFrame with confidence intervalls, by default False
        alpha : float, optional
            Alpha level for confidence intervalls, by default DEFAULT_ALPHA

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.

        Raises
        ------
        Exception
            Error when merging data
        """
        if alpha != DEFAULT_ALPHA:
            raise NotImplementedError(
                "alpha must be given in Prophet() as interval_width (1-alpha)"
            )

        fh = fh.to_relative(cutoff=self.cutoff)

        if isinstance(fh.to_pandas(), pd.DatetimeIndex):
            df = pd.DataFrame()
            df["ds"] = fh.to_pandas()
        else:
            # Try to create pd.DatetimeIndex
            df = self._coerce_to_datetime_index(fh=fh)

        # Merge X with df (of created future DatetimeIndex values)
        df = _merge_X(fh=fh, X=X, df=df)

        # Prediction
        out = self._forecaster.predict(df)
        out.index = [x for x in range(-len(self._y), len(out) - len(self._y))]

        # Workaraound for slicing on negative index
        out["idx"] = out.index
        out = out.loc[out["idx"].isin(fh.to_indexer(self.cutoff).values)]
        out.index = fh.to_absolute(self.cutoff)
        out = out.drop(columns=["idx"])

        y_pred = out["yhat"].rename(None)
        if return_pred_int:
            pred_int = out[["yhat_lower", "yhat_upper"]].rename(
                columns={"yhat_lower": "lower", "yhat_upper": "upper"}
            )
            return y_pred, pred_int
        else:
            return y_pred

    def _get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict

        References
        ----------
        https://facebook.github.io/prophet/docs/additional_topics.html
        """
        self.check_is_fitted()
        fitted_params = {}
        for name in ["k", "m", "sigma_obs"]:
            fitted_params[name] = self._forecaster.params[name][0][0]
        for name in ["delta", "beta"]:
            fitted_params[name] = self._forecaster.params[name][0]
        return fitted_params

    def _check_changepoints(self):
        """Checking arguments for changepoints and assign related arguments

        Returns
        -------
        self
        """
        if self.changepoints is not None:
            self.changepoints = pd.Series(pd.to_datetime(self.changepoints), name="ds")
            self.n_changepoints = len(self.changepoints)
            self.specified_changepoints = True
        else:
            self.specified_changepoints = False
        return self

    def _coerce_to_datetime_index(self, fh):
        """Create DatetimeIndex

        Parameters
        ----------
        fh : sktime.ForecastingHorizon

        Returns
        -------
        pd.DataFrame
            DataFrame with pd.DatetimeIndex as column "ds"

        Raises
        ------
        TypeError
            Error when fh values have wrong type
        """
        if self.freq is None:
            self.freq = self._y.index.freq
        try:
            periods = fh.to_pandas().max()
            df = self._forecaster.make_future_dataframe(
                periods=periods + 1, freq=self.freq, include_history=True
            )
        except Exception:
            raise TypeError(
                "Type of fh values must be int, np.array, list or pd.DatetimeIndex"
            )
        return df


def _merge_X(fh, X, df):
    """Merge X and df on the DatetimeIndex

    Parameters
    ----------
    fh : sktime.ForecastingHorizon
    X : pd.DataFrame
        Exog data
    df : pd.DataFrame
        Contains a DatetimeIndex column "ds"

    Returns
    -------
    pd.DataFrame
        DataFrame with containing X and df (with a DatetimeIndex column "ds")

    Raises
    ------
    TypeError
        Error if merging was not possible
    """
    merge_error = (
        "Either length of fh and X must be "
        "same or X must have future DatetimeIndex values."
    )
    if X is not None:
        X = check_X(X)
        try:
            if len(X) == len(fh.to_pandas()):
                X = X.set_index(df.index)
                df = pd.concat([df, X], axis=1)
            else:
                df.index = df["ds"]
                df = df.merge(X, left_index=True, right_on=X.index)
        except Exception:
            raise TypeError(merge_error)
    if df.empty:
        raise TypeError(merge_error)
    return df


class _StatsModelsAdapter(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """Base class for interfacing statsmodels forecasting algorithms"""

    _fitted_param_names = ()

    def __init__(self):
        self._forecaster = None
        self._fitted_forecaster = None
        super(_StatsModelsAdapter, self).__init__()

    def fit(self, y, X=None, fh=None):
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
        # statsmodels does not support the pd.Int64Index as required,
        # so we coerce them here to pd.RangeIndex
        if isinstance(y, pd.Series) and type(y.index) == pd.Int64Index:
            y, X = _coerce_int_to_range_index(y, X)

        self._set_y_X(y, X)
        self._set_fh(fh)
        self._fit_forecaster(y, X)
        self._is_fitted = True
        return self

    def _fit_forecaster(self, y_train, X_train=None):
        """Internal fit"""
        raise NotImplementedError("abstract method")

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Make forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        return_pred_int : bool, optional (default=False)
        alpha : int or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Returns series of predicted values.
        """
        if return_pred_int:
            raise NotImplementedError()

        # statsmodels requires zero-based indexing starting at the
        # beginning of the training series when passing integers
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]
        y_pred = self._fitted_forecaster.predict(start, end)

        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon
        return y_pred.loc[fh.to_absolute(self.cutoff).to_pandas()]

    def get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        return {
            name: self._fitted_forecaster.params.get(name)
            for name in self._get_fitted_param_names()
        }

    def _get_fitted_param_names(self):
        """Get names of fitted parameters"""
        return self._fitted_param_names


def _coerce_int_to_range_index(y, X=None):
    new_index = pd.RangeIndex(y.index[0], y.index[-1] + 1)
    try:
        np.testing.assert_array_equal(y.index, new_index)
    except AssertionError:
        raise ValueError(
            "Coercion of pd.Int64Index to pd.RangeIndex "
            "failed. Please provide `y_train` with a "
            "pd.RangeIndex."
        )
    y.index = new_index
    if X is not None:
        X.index = new_index
    return y, X
