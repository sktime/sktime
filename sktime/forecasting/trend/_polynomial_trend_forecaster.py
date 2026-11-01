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
from sktime.forecasting.trend._util import (
    _get_X_numpy_int_from_pandas,
    _trend_time_origin,
)


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
    of per days (to the power of the respective degree).
    If time stamps are ``pd.PeriodIndex``,
    coefficients are in units of (full) periods (to the power of the respective degree).

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
        "category": ["statistical"],
        "authors": ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly", "ericjb"],
        "maintainers": ["tensorflow-as-tf"],
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "y_inner_mtype": "pd.DataFrame",
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

    def _trend_origin(self):
        """Origin stored at fit. Unfitted or pre-fix pickles stay on absolute time."""
        return getattr(self, "_trend_origin_", 0.0)

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored
        fh : int, list or np.array, default=None
            The forecasters horizon with the steps ahead to predict.

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
        # Time is shifted to the first training stamp. See _trend_time_origin.
        X_absolute = _get_X_numpy_int_from_pandas(y.index)
        self._trend_origin_ = _trend_time_origin(X_absolute)
        X_sklearn = X_absolute - self._trend_origin_

        # time stamps can be large numbers, e.g., days since 1970, which makes
        # polynomial features of higher degree severely ill-conditioned.
        # We therefore fit on an affinely rescaled time axis, see _get_time_rescaling
        self._t_offset, self._t_scale = self._get_time_rescaling(X_sklearn)

        if self._fits_in_rescaled_time(regressor):
            # fit in rescaled time, then convert coefficients back exactly,
            # so regressor_ remains a pipeline in original time units
            X_fit = self._rescale_time(X_sklearn)
            self.regressor_.fit(X_fit, y.iloc[:, 0])
            self._convert_coefs_to_original_time(self.regressor_.steps[-1][1])
        else:
            self.regressor_.fit(X_sklearn, y.iloc[:, 0])

        if self.prediction_intervals:
            # calculate and save values needed for the prediction interval method
            fitted_values = self.regressor_.predict(X_sklearn)
            residuals = y.iloc[:, 0] - fitted_values
            p = self.degree + int(self.with_intercept)
            self.s_squared_ = np.sum(residuals**2) / (len(y) - p)
            self.train_index_ = y.index

        return self

    def _get_time_rescaling(self, X):
        """Get offset and scale of the affine time rescaling used in fitting.

        The rescaled time is ``(t - offset) / scale``.

        If ``with_intercept=True``, time is shifted and scaled to the unit interval.
        If ``with_intercept=False``, the polynomial must vanish at the origin
        of the original time axis, so time is only scaled, not shifted.

        Parameters
        ----------
        X : 2D np.ndarray of shape (n, 1)
            time stamps in original units, as returned by
            ``_get_X_numpy_int_from_pandas``

        Returns
        -------
        offset : float
        scale : float, always positive
        """
        X = np.asarray(X, dtype="float64").reshape(-1)
        if self.with_intercept:
            offset = X.min()
            scale = X.max() - offset
        else:
            offset = 0.0
            scale = np.max(np.abs(X))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        return float(offset), float(scale)

    def _rescale_time(self, X):
        """Apply the affine time rescaling obtained in fit to X."""
        return (np.asarray(X, dtype="float64") - self._t_offset) / self._t_scale

    def _fits_in_rescaled_time(self, regressor):
        """Whether fitting is done in rescaled time, with exact back-conversion.

        This is the case for an unconstrained ``LinearRegression``, where
        the fitted polynomial does not depend on the parametrization of time,
        and the coefficients can be converted back to original units exactly.
        """
        if not isinstance(regressor, LinearRegression):
            return False
        return not getattr(regressor, "positive", False)

    def _convert_coefs_to_original_time(self, linreg):
        """Convert coefficients of a fitted linear regressor to original time units.

        ``linreg`` was fitted on polynomial features of the rescaled time
        ``u = (t - offset) / scale``. Its ``coef_`` and ``intercept_`` are
        replaced in-place by the coefficients of the same polynomial,
        expressed in powers of the original time ``t``.

        Parameters
        ----------
        linreg : fitted sklearn ``LinearRegression``
            final step of ``self.regressor_``, fitted on rescaled time
        """
        poly = np.polynomial.Polynomial
        start = 0 if self.with_intercept else 1

        # polynomial in u, coefficients in increasing order of powers
        coef_u = np.zeros(self.degree + 1)
        coef_u[start:] = np.asarray(linreg.coef_, dtype="float64").reshape(-1)
        intercept_u = float(np.asarray(linreg.intercept_).reshape(-1)[0])
        coef_u[0] += intercept_u

        # substitute u = (t - offset) / scale to obtain polynomial in t
        u_of_t = poly([-self._t_offset / self._t_scale, 1 / self._t_scale])
        coef_t = poly(coef_u)(u_of_t).coef
        coef_t = np.pad(coef_t, (0, self.degree + 1 - len(coef_t)))

        if linreg.fit_intercept:
            # constant term goes to the intercept; the bias feature, if present,
            # is centered away by sklearn and gets a zero coefficient
            linreg.intercept_ = coef_t[0]
            coef_t = coef_t.copy()
            coef_t[0] = 0.0
            linreg.coef_ = coef_t[start:]
        else:
            # no shift is applied if with_intercept=False, so coef_t[0] == 0
            linreg.intercept_ = 0.0
            linreg.coef_ = coef_t[start:]

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
        X_sklearn = _get_X_numpy_int_from_pandas(fh) - self._trend_origin()
        y_pred_sklearn = self.regressor_.predict(X_sklearn)
        cols = self._get_varnames()
        y_pred = pd.DataFrame(y_pred_sklearn, index=fh, columns=cols)
        return y_pred

    def _predict_var(self, fh=None, X=None, cov=False):
        """Compute the variance at each forecast horizon."""
        if not self.prediction_intervals:
            raise ValueError(
                "Prediction intervals were not calculated during fit. \
                Set prediction_intervals=True at initialization."
            )

        # 1. get X (design matrix) and M = (X^t X)^-1
        # the term x_0^t M x_0 below is invariant under affine reparametrization
        # of time, so we compute it in rescaled time for numerical stability
        t_train = _get_X_numpy_int_from_pandas(self.train_index_).flatten()
        t_train = t_train - self._trend_origin()
        t_train = self._rescale_time(t_train)
        X = np.polynomial.polynomial.polyvander(t_train, self.degree)
        if not self.with_intercept:
            X = X[:, 1:]  # remove the column of 1's that handles the intercept

        M = np.linalg.inv(X.T @ X)

        # 2. get time vector t for the forecast horizons
        if fh.is_relative:
            fh = fh.to_absolute(cutoff=self.train_index_[-1])

        t_fh = fh.to_pandas()
        fh_periods = _get_X_numpy_int_from_pandas(t_fh) - self._trend_origin()
        t = self._rescale_time(np.array(fh_periods))

        # 3. calculate (half-) range of PI (1 + sqrt(x_0^t M x_0)) (up to scaling)
        start = 0 if self.with_intercept else 1
        v = []

        for _, z in enumerate(t):
            w = np.array([z**j for j in range(start, self.degree + 1)])
            v.append(w.T @ M @ w)

        v = (1 + np.array(v)).flatten()  # see Hyndman FPP3 Section 7.9

        l_var = v * self.s_squared_  # see Hyndman FPP3 Section 7.9
        cols = self._get_varnames()
        fh = self.fh.to_absolute_index(self.cutoff)
        pred_var = pd.DataFrame(l_var, index=fh, columns=cols)
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
