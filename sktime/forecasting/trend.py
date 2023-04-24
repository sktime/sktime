# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements trend based forecasters."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]
__all__ = ["TrendForecaster", "PolynomialTrendForecaster", "STLForecaster"]

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sktime.forecasting.base import BaseForecaster


def _get_X_numpy_int_from_pandas(x):
    """Convert pandas index to an sklearn compatible X, 2D np.ndarray, int type."""
    if isinstance(x, (pd.DatetimeIndex)):
        x = x.astype("int64") / 864e11
    else:
        x = x.astype("int64")
    return x.to_numpy().reshape(-1, 1)


class TrendForecaster(BaseForecaster):
    r"""Trend based forecasts of time series data, regressing values on index.

    Uses a `sklearn` regressor `regressor` to regress values of time series on index:

    In `fit`, for input time series :math:`(v_i, t_i), i = 1, \dots, T`,
    where :math:`v_i` are values and :math:`t_i` are time stamps,
    fits an `sklearn` model :math:`v_i = f(t_i) + \epsilon_i`, where `f` is
    the model fitted when `regressor.fit` is passed `X` = vector of :math:`t_i`,
    and `y` = vector of :math:`v_i`.

    In `predict`, for a new time point :math:`t_*`, predicts :math:`f(t_*)`,
    where :math:`f` is the function as fitted above in `fit`.

    Default for `regressor` is linear regression = `sklearn` `LinearRegression` default.

    If time stamps are `pd.DatetimeIndex`, fitted coefficients are in units
    of days since start of 1970. If time stamps are `pd.PeriodIndex`,
    coefficients are in units of (full) periods since start of 1970.

    Parameters
    ----------
    regressor : estimator object, default = None
        Define the regression model type. If not set, will default to
         sklearn.linear_model.LinearRegression

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import TrendForecaster
    >>> y = load_airline()
    >>> forecaster = TrendForecaster()
    >>> forecaster.fit(y)
    TrendForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, regressor=None):
        # for default regressor, set fit_intercept=True
        self.regressor = regressor
        super(TrendForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.regressor is None:
            self.regressor_ = LinearRegression(fit_intercept=True)
        else:
            self.regressor_ = clone(self.regressor)

        # we regress index on series values
        # the sklearn X is obtained from the index of y
        # the sklearn y can be taken as the y seen here
        X_sklearn = _get_X_numpy_int_from_pandas(y.index)

        # fit regressor
        self.regressor_.fit(X_sklearn, y)
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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.ensemble import RandomForestRegressor

        params_list = [{}, {"regressor": RandomForestRegressor()}]

        return params_list


class PolynomialTrendForecaster(BaseForecaster):
    r"""Forecast time series data with a polynomial trend.

    Uses a `sklearn` regressor `regressor` to regress values of time series on index,
    after extraction of polynomial features.
    Same `TrendForecaster` where `regressor` is pipelined with transformation step
    `PolynomialFeatures(degree, with_intercept)` applied to time, at the start.

    In `fit`, for input time series :math:`(v_i, p(t_i)), i = 1, \dots, T`,
    where :math:`v_i` are values, :math:`t_i` are time stamps,
    and :math:`p` is the polynomial feature transform with degree `degree`,
    and with/without intercept depending on `with_intercept`,
    fits an `sklearn` model :math:`v_i = f(p(t_i)) + \epsilon_i`, where `f` is
    the model fitted when `regressor.fit` is passed `X` = vector of :math:`p(t_i)`,
    and `y` = vector of :math:`v_i`.

    In `predict`, for a new time point :math:`t_*`, predicts :math:`f(p(t_*))`,
    where :math:`f` is the function as fitted above in `fit`,
    and :math:`p` is the same polynomial feature transform as above.

    Default for `regressor` is linear regression = `sklearn` `LinearRegression` default.

    If time stamps are `pd.DatetimeIndex`, fitted coefficients are in units
    of days since start of 1970. If time stamps are `pd.PeriodIndex`,
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
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, regressor=None, degree=1, with_intercept=True):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.regressor_ = self.regressor
        super(PolynomialTrendForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.ensemble import RandomForestRegressor

        params_list = [
            {},
            {
                "regressor": RandomForestRegressor(),
                "degree": 2,
                "with_intercept": False,
            },
        ]

        return params_list


class STLForecaster(BaseForecaster):
    """Implements STLForecaster based on statsmodels.tsa.seasonal.STL implementation.

    The STLForecaster applies the following algorithm, also see [1]_.

    in `fit`:
    1. use `statsmodels` `STL` [2]_ to decompose the given series `y` into
        the three components: `trend`, `season` and `residuals`.
    2. fit clones of `forecaster_trend` to `trend`, `forecaster_seasonal` to `season`,
        and `forecaster_resid` to `residuals`, using `y`, `X`, `fh` from `fit`.
        The forecasters are fitted as clones, stored in the attributes
        `forecaster_trend_`, `forecaster_seasonal_`, `forecaster_resid_`.

    In `predict`, forecasts as follows:
    1. obtain forecasts `y_pred_trend` from `forecaster_trend_`,
        `y_pred_seasonal` from `forecaster_seasonal_`, and
        `y_pred_residual` from `forecaster_resid_`, using `X`, `fh`, from `predict`.
    2. recompose `y_pred` as `y_pred = y_pred_trend + y_pred_seasonal + y_pred_residual`
    3. return `y_pred`

    `update` refits entirely, i.e., behaves as `fit` on all data seen so far.

    Parameters
    ----------
    sp : int, optional, default=2. Passed to `statsmodels` `STL`.
        Length of the seasonal period passed to `statsmodels` `STL`.
        (forecaster_seasonal, forecaster_resid) that are None. The
        default forecaster_trend does not get sp as trend is independent
        to seasonality.
    seasonal : int, optional., default=7. Passed to `statsmodels` `STL`.
        Length of the seasonal smoother. Must be an odd integer >=3, and should
        normally be >= 7 (default).
    trend : {int, None}, optional, default=None. Passed to `statsmodels` `STL`.
        Length of the trend smoother. Must be an odd integer. If not provided
        uses the smallest odd integer greater than
        1.5 * period / (1 - 1.5 / seasonal), following the suggestion in
        the original implementation.
    low_pass : {int, None}, optional, default=None. Passed to `statsmodels` `STL`.
        Length of the low-pass filter. Must be an odd integer >=3. If not
        provided, uses the smallest odd integer > period.
    seasonal_deg : int, optional, default=1. Passed to `statsmodels` `STL`.
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
    trend_deg : int, optional, default=1. Passed to `statsmodels` `STL`.
        Degree of trend LOESS. 0 (constant) or 1 (constant and trend).
    low_pass_deg : int, optional, default=1. Passed to `statsmodels` `STL`.
        Degree of low pass LOESS. 0 (constant) or 1 (constant and trend).
    robust : bool, optional, default=False. Passed to `statsmodels` `STL`.
        Flag indicating whether to use a weighted version that is robust to
        some forms of outliers.
    seasonal_jump : int, optional, default=1. Passed to `statsmodels` `STL`.
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every seasonal_jump points and linear
        interpolation is between fitted points. Higher values reduce
        estimation time.
    trend_jump : int, optional, default=1. Passed to `statsmodels` `STL`.
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every trend_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.
    low_pass_jump : int, optional, default=1. Passed to `statsmodels` `STL`.
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every low_pass_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.
    inner_iter: int or None, optional, default=None. Passed to `statsmodels` `STL`.
        Number of iterations to perform in the inner loop. If not provided uses 2 if
        robust is True, or 5 if not. This param goes into STL.fit() from statsmodels.
    outer_iter: int or None, optional, default=None. Passed to `statsmodels` `STL`.
        Number of iterations to perform in the outer loop. If not provided uses 15 if
        robust is True, or 0 if not. This param goes into STL.fit() from statsmodels.
    forecaster_trend : sktime forecaster, optional
        Forecaster to be fitted on trend_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="drift") is used.
    forecaster_seasonal : sktime forecaster, optional
        Forecaster to be fitted on seasonal_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="last") is used.
    forecaster_resid : sktime forecaster, optional
        Forecaster to be fitted on resid_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="mean") is used.

    Attributes
    ----------
    trend_ : pd.Series
        Trend component.
    seasonal_ : pd.Series
        Seasonal component.
    resid_ : pd.Series
        Residuals component.
    forecaster_trend_ : sktime forecaster
        Fitted trend forecaster.
    forecaster_seasonal_ : sktime forecaster
        Fitted seasonal forecaster.
    forecaster_resid_ : sktime forecaster
        Fitted residual forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import STLForecaster
    >>> y = load_airline()
    >>> forecaster = STLForecaster(sp=12)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    STLForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP

    See Also
    --------
    Deseasonalizer
    Detrender

    References
    ----------
    .. [1] R. B. Cleveland, W. S. Cleveland, J.E. McRae, and I. Terpenning (1990)
       STL: A Seasonal-Trend Decomposition Procedure Based on LOESS.
       Journal of Official Statistics, 6, 3-73.
    .. [2] https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        sp=2,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=False,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        inner_iter=None,
        outer_iter=None,
        forecaster_trend=None,
        forecaster_seasonal=None,
        forecaster_resid=None,
    ):
        self.sp = sp
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.forecaster_trend = forecaster_trend
        self.forecaster_seasonal = forecaster_seasonal
        self.forecaster_resid = forecaster_resid
        super(STLForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        from statsmodels.tsa.seasonal import STL as _STL

        from sktime.forecasting.naive import NaiveForecaster

        self._stl = _STL(
            y.values,
            period=self.sp,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump,
        ).fit(inner_iter=self.inner_iter, outer_iter=self.outer_iter)

        self.seasonal_ = pd.Series(self._stl.seasonal, index=y.index)
        self.resid_ = pd.Series(self._stl.resid, index=y.index)
        self.trend_ = pd.Series(self._stl.trend, index=y.index)

        self.forecaster_seasonal_ = (
            NaiveForecaster(sp=self.sp, strategy="last")
            if self.forecaster_seasonal is None
            else self.forecaster_seasonal.clone()
        )
        # trend forecaster does not need sp
        self.forecaster_trend_ = (
            NaiveForecaster(strategy="drift")
            if self.forecaster_trend is None
            else self.forecaster_trend.clone()
        )
        self.forecaster_resid_ = (
            NaiveForecaster(sp=self.sp, strategy="mean")
            if self.forecaster_resid is None
            else self.forecaster_resid.clone()
        )

        # fitting forecasters to different components
        self.forecaster_seasonal_.fit(y=self.seasonal_, X=X, fh=fh)
        self.forecaster_trend_.fit(y=self.trend_, X=X, fh=fh)
        self.forecaster_resid_.fit(y=self.resid_, X=X, fh=fh)

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
                Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_pred_seasonal = self.forecaster_seasonal_.predict(fh=fh, X=X)
        y_pred_trend = self.forecaster_trend_.predict(fh=fh, X=X)
        y_pred_resid = self.forecaster_resid_.predict(fh=fh, X=X)
        y_pred = y_pred_seasonal + y_pred_trend + y_pred_resid
        y_pred.name = self._y.name
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.array
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.seasonal import STL as _STL

        self._stl = _STL(
            y.values,
            period=self.sp,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump,
        ).fit(inner_iter=self.inner_iter, outer_iter=self.outer_iter)

        self.seasonal_ = pd.Series(self._stl.seasonal, index=y.index)
        self.resid_ = pd.Series(self._stl.resid, index=y.index)
        self.trend_ = pd.Series(self._stl.trend, index=y.index)

        self.forecaster_seasonal_.update(
            y=self.seasonal_, X=X, update_params=update_params
        )
        self.forecaster_trend_.update(y=self.trend_, X=X, update_params=update_params)
        self.forecaster_resid_.update(y=self.resid_, X=X, update_params=update_params)
        return self

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.naive import NaiveForecaster

        params_list = [
            {},
            {
                "sp": 3,
                "seasonal": 7,
                "trend": 5,
                "seasonal_deg": 2,
                "trend_deg": 2,
                "robust": True,
                "seasonal_jump": 2,
                "trend_jump": 2,
                "low_pass_jump": 2,
                "forecaster_trend": NaiveForecaster(strategy="drift"),
                "forecaster_seasonal": NaiveForecaster(sp=3),
                "forecaster_resid": NaiveForecaster(strategy="mean"),
            },
        ]

        return params_list
