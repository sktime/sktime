# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements PolynomialTrendForecaster."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]
__all__ = ["PolynomialTrendForecaster"]

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas


class PolynomialTrendForecaster(BaseForecaster):
    r"""Forecast time series data with a polynomial trend.

    Uses an ``sklearn`` regressor specified by the ``regressor`` parameter
    to perform regression on time series values against their corresponding indices,
    after extraction of polynomial features.
    Same as ``TrendForecaster`` where ``regressor`` is pipelined with transformation
    step ``PolynomialFeatures(degree, with_intercept)`` applied to time index,
    at the start.

    In ``fit``, for input time series :math:`(v_i, p(t_i)), i = 1, \dots, T`,
    where :math:`v_i` are values, :math:`t_i` are time stamps,
    and :math:`p` is the polynomial feature transform with degree ``degree``,
    and with/without intercept depending on ``with_intercept``,
    fits an ``sklearn`` model :math:`v_i = f(p(t_i)) + \epsilon_i`, where :math:`f` is
    the model fitted when ``regressor.fit`` is passed ``X`` = vector of :math:`p(t_i)`,
    and ``y`` = vector of :math:`v_i`.

    In ``predict``, for a new time point :math:`t_*`, predicts :math:`f(p(t_*))`,
    where :math:`f` is the function as fitted above in ``fit``,
    and :math:`p` is the same polynomial feature transform as above.

    Default for ``regressor`` is linear regression = ``sklearn`` ``LinearRegression``,
    with default parameters. Default for ``degree`` is 1.

    If time stamps are ``pd.DatetimeIndex``, fitted coefficients are in units
    of days since start of 1970. If time stamps are ``pd.PeriodIndex``,
    coefficients are in units of (full) periods since start of 1970.

    Parameters
    ----------
    regressor : sklearn regressor estimator object, default = None
        Define the regression model type. If not set, will default to
        sklearn.linear_model.LinearRegression
    degree : int, default = 1
        Degree of polynomial function
    with_intercept : bool, default=True
        If true, then include a feature in which all polynomial powers are
        zero. (i.e. a column of ones, acts as an intercept term in a linear
        model)
    prediction_intervals : bool, default=False
        Whether to compute prediction intervals.
        If True, additional calculations are done during fit to enable prediction
        intervals to be calculated during predict.
        The prediction intervals are calculated according to Section 7.9 in [1].
        The formulas are standard and are based on an OLS regression model fitted to
        the data. The formulas in [1] assume a regression with
        intercept and are modified appropriately if with_intercept is False.

    Attributes
    ----------
    regressor_ : sklearn regressor estimator object
        The fitted regressor object.
        This is a fitted ``sklearn`` pipeline with steps
        ``PolynomialFeatures(degree, with_intercept)``,
        followed by a clone of ``regressor``.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice, 3rd edition. OTexts: Melbourne, Australia. OTexts.com/fpp3.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> y = load_airline()
    >>> forecaster = PolynomialTrendForecaster(degree=1)
    >>> forecaster.fit(y)
    PolynomialTrendForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "authors": ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly", "ericjb"],
        "maintainers": ["tensorflow-as-tf"],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": True,
    }

    def __init__(
        self, regressor=None, degree=1, with_intercept=True, prediction_intervals=False
    ):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.regressor_ = self.regressor
        self.prediction_intervals = prediction_intervals
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
        # for default regressor, set fit_intercept=False as we generate a
        # dummy variable in polynomial features
        if self.regressor is None:
            regressor = LinearRegression(fit_intercept=False)
        else:
            regressor = clone(self.regressor)

        # make pipeline with polynomial features
        self.regressor_ = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=self.with_intercept),
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
