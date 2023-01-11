# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SARIMAX."""

__all__ = ["SARIMAX"]
__author__ = ["TNTran92"]

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class SARIMAX(_StatsModelsAdapter):
    """SARIMAX forecaster.

    Direct interface for `statsmodels.tsa.api.SARIMAX`.

    Parameters
    ----------
    order : iterable or iterable of iterables, optional, default=(1,0,0)
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. `d` must be an integer
        indicating the integration order of the process, while
        `p` and `q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is
        an AR(1) model: (1,0,0).
    seasonal_order : iterable, optional, default=(0,0,0,0)
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
        `D` must be an integer indicating the integration order of the process,
        while `P` and `Q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. `s` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.
    trend : str{'n','c','t','ct'} or iterable, optional, default="c"
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is to not include a trend component.
    measurement_error : bool, optional, default=False
        Whether or not to assume the endogenous observations `endog` were
        measured with error.
    time_varying_regression : bool, optional, default=False
        Used when an explanatory variables, `exog`, are provided
        to select whether or not coefficients on the exogenous regressors are
        allowed to vary over time.
    mle_regression : bool, optional, default=True
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or through
        the Kalman filter (i.e. recursive least squares). If
        `time_varying_regression` is True, this must be set to False.
    simple_differencing : bool, optional, default=False
        Whether or not to use partially conditional maximum likelihood
        estimation. If True, differencing is performed prior to estimation,
        which discards the first :math:`s D + d` initial rows but results in a
        smaller state-space formulation. See the Notes section for important
        details about interpreting results when this option is used. If False,
        the full SARIMAX model is put in state-space form so that all
        datapoints can be used in estimation.
    enforce_stationarity : bool, optional, default=True
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model.
    enforce_invertibility : bool, optional, default=True
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model.
    hamilton_representation : bool, optional, default=False
        Whether or not to use the Hamilton representation of an ARMA process
        (if True) or the Harvey representation (if False).
    concentrate_scale : bool, optional, default=False
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters estimated
        by maximum likelihood by one, but standard errors will then not
        be available for the scale parameter.
    trend_offset : int, optional, default=1
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    use_exact_diffuse : bool, optional, default=False
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).
    random_state : int, RandomState instance or None, optional ,
        default=None – If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    See Also
    --------
    ARIMA
    AutoARIMA
    StatsForecastAutoARIMA

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice. OTexts, 2014.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sarimax import SARIMAX
    >>> y = load_airline()
    >>> forecaster = SARIMAX(
    ...     order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))  # doctest: +SKIP
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    SARIMAX(...)
    >>> y_pred = forecaster.predict(fh=y.index)  # doctest: +SKIP
    """

    _tags = {
        "ignores-exogeneous-X": False,
    }

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        measurement_error=False,
        time_varying_regression=False,
        mle_regression=True,
        simple_differencing=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
        hamilton_representation=False,
        concentrate_scale=False,
        trend_offset=1,
        use_exact_diffuse=False,
        dates=None,
        freq=None,
        missing="none",
        validate_specification=True,
        random_state=None,
    ):

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        self.use_exact_diffuse = use_exact_diffuse
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.validate_specification = validate_specification

        super().__init__(random_state=random_state)

    def _fit_forecaster(self, y, X=None):
        from statsmodels.tsa.api import SARIMAX as _SARIMAX

        self._forecaster = _SARIMAX(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            measurement_error=self.measurement_error,
            time_varying_regression=self.time_varying_regression,
            mle_regression=self.mle_regression,
            simple_differencing=self.simple_differencing,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            hamilton_representation=self.hamilton_representation,
            concentrate_scale=self.concentrate_scale,
            trend_offset=self.trend_offset,
            use_exact_diffuse=self.use_exact_diffuse,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
            validate_specification=self.validate_specification,
        )
        self._fitted_forecaster = self._forecaster.fit()

    def summary(self):
        """Get a summary of the fitted forecaster.

        This is the same as the implementation in statsmodels:
        https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_structural_harvey_jaeger.html
        """
        return self._fitted_forecaster.summary()
