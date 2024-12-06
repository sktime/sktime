# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SplineTrendForecaster."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly", "jgyasu"]
__all__ = ["SplineTrendForecaster"]

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas


class SplineTrendForecaster(BaseForecaster):
    """Forecast time series data with a spline trend."""

    _tags = {
        "authors": [
            "tensorflow-as-tf",
            "mloning",
            "aiwalter",
            "fkiraly",
            "ericjb",
            "jgyasu",
        ],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        regressor=None,
        degree=1,
        with_intercept=True,
        prediction_intervals=False,
        knots="uniform",
        extrapolation="constant",
    ):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.regressor_ = self.regressor
        self.prediction_intervals = prediction_intervals
        self.knots = knots
        self.extrapolation = extrapolation
        # prediction_intervals : bool, default=False
        # By default, the extra information needed to later generate the prediction
        # intervals is not calculated. If set to True, the extra information is
        # calculated and stored in the forecaster.
        super().__init__()

        self.set_tags(**{"capability:pred_int": prediction_intervals})

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored
        fh : int, list or np.array, default=None
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.regressor is None:
            regressor = LinearRegression(fit_intercept=False)
        else:
            regressor = clone(self.regressor)

        # make pipeline with spline features
        self.regressor_ = make_pipeline(
            SplineTransformer(
                degree=self.degree,
                knots=self.knots,
                extrapolation=self.extrapolation,
                include_bias=self.with_intercept,
            ),
            regressor,
        )

        # we regress index on series values
        # the sklearn X is obtained from the index of y
        # the sklearn y can be taken as the y seen here
        X_sklearn = _get_X_numpy_int_from_pandas(y.index)

        # fit regressor
        self.regressor_.fit(X_sklearn, y)

        if self.prediction_intervals:
            # calculate and save values needed for the prediction interval method
            fitted_values = self.regressor_.predict(X_sklearn)
            residuals = y - fitted_values
            p = self.degree + int(self.with_intercept)
            self.s_squared_ = np.sum(residuals**2) / (len(y) - p)
            self.train_index_ = y.index

        return self

    def _predict(self, fh=None, X=None):
        """Make forecasts for the given forecast horizon.

        Parameters
        ----------
        fh : int, list or np.array
            The forecast horizon with the steps ahead to predict
        X : pd.DataFrame, default=None
            Exogenous variables (ignored)

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast
        """
        # use relative fh as time index to predict
        fh = self.fh.to_absolute_index(self.cutoff)
        X_sklearn = _get_X_numpy_int_from_pandas(fh)
        y_pred_sklearn = self.regressor_.predict(X_sklearn)
        y_pred = pd.Series(y_pred_sklearn, index=fh)
        y_pred.name = self._y.name
        return y_pred

    def _predict_var(self, fh=None, X=None, cov=False):
        """Compute the variance at each forecast horizon."""
        if not self.prediction_intervals:
            raise ValueError(
                "Prediction intervals were not calculated during fit. \
                Set prediction_intervals=True at initialization."
            )

        # 1. get X (design matrix) and M = (X^t X)^-1
        t_train = _get_X_numpy_int_from_pandas(self.train_index_).flatten()
        X = np.polynomial.polynomial.polyvander(t_train, self.degree)
        if not self.with_intercept:
            X = X[:, 1:]  # remove the column of 1's that handles the intercept

        M = np.linalg.inv(X.T @ X)

        # 2. get time vector t for the forecast horizons
        if fh.is_relative:
            fh = fh.to_absolute(cutoff=self.train_index_[-1])

        t_fh = fh.to_pandas()
        fh_periods = _get_X_numpy_int_from_pandas(t_fh)
        t = np.array(fh_periods)

        # 3. calculate (half-) range of PI (1 + sqrt(x_0^t M x_0)) (up to scaling)
        start = 0 if self.with_intercept else 1
        v = []

        for _, z in enumerate(t):
            w = np.array([z**j for j in range(start, self.degree + 1)])
            v.append(w.T @ M @ w)

        v = (1 + np.array(v)).flatten()  # see Hyndman FPP3 Section 7.9

        l_var = v * self.s_squared_  # see Hyndman FPP3 Section 7.9
        pred_var = pd.DataFrame(l_var, columns=[self._y.name])
        return pred_var

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.ensemble import RandomForestRegressor

        params_list = [
            {},
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": False,
                "prediction_intervals": False,
            },
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": True,
                "prediction_intervals": True,
            },
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": False,
                "prediction_intervals": True,
            },
        ]

        return params_list
