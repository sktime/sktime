#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a piecewise linear trend forecaster by wrapping fbprophet."""

__author__ = ["sbuse"]

import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base.adapters import _ProphetAdapter


class ProphetPiecewiseLinearTrendForecaster(_ProphetAdapter):
    """
    Forecast time series data with a piecewise linear trend, fitted via prophet.

    The forecaster uses Facebook's prophet algorithm [1]_ and extracts the piecewise
    linear trend from it. Only hyper-parameters relevant for the trend modelling are
    exposed via the constructor.

    Seasonalities are set to additive and "auto" detection in prophet,
    which means that yearly, weekly and daily seasonality are automatically detected,
    and included in the model if present, using prophet's default settings.

    For more granular control of components or seasonality, use
    ``sktime.forecasting.fbprophet.Prophet`` directly.

    Data can be passed in one of the sktime compatible formats,
    naming a column ``ds`` such as in the prophet package is not necessary.

    Unlike vanilla ``prophet``, also supports integer/range and period index:
    * integer/range index is interpreted as days since Jan 1, 2000
    * ``PeriodIndex`` is converted using the ``pandas`` method ``to_timestamp``

    Parameters
    ----------
    changepoints: list or None, default=None
        List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: int, default=25
        Number of potential changepoints to include. Not used
        if input ``changepoints`` is supplied. If ``changepoints`` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first ``changepoint_range`` proportion of the history.
    changepoint_range: float, default=0.8
        Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        ``changepoints`` is specified.
    changepoint_prior_scale: float, default=0.05
        Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
        Recommended to take values within [0.001,0.5].
    yearly_seasonality: str or bool or int, default=False
        Include yearly seasonality in the model. "auto" for automatic determination,
        True to enable, False to disable, or an integer specifying the number of terms
        to include in the Fourier series.
    weekly_seasonality: str or bool or int, default=False
        Include weekly seasonality in the model. "auto" for automatic determination,
        True to enable, False to disable, or an integer specifying the number of terms
        to include in the Fourier series.
    daily_seasonality: str or bool or int, default=False
        Include weekly seasonality in the model. "auto" for automatic determination,
        True to enable, False to disable, or an integer specifying the number of terms
        to include in the Fourier series.

    References
    ----------
    .. [1] https://facebook.github.io/prophet

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import ProphetPiecewiseLinearTrendForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.split import temporal_train_test_split
    >>> y =load_airline().to_timestamp(freq='M')
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y.index, is_relative=False)
    >>> forecaster =  ProphetPiecewiseLinearTrendForecaster() # doctest: +SKIP
    >>> forecaster.fit(y_train) # doctest: +SKIP
    ProphetPiecewiseLinearTrendForecaster(...)
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP
    """

    _tags = {
        "authors": ["sbuse", "bletham", "tcuongd"],
        # bletham, tcuongd for prophet
        "maintainers": ["sbuse"],
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "python_dependencies": ["prophet", "numpy<2.0"],
    }

    def __init__(
        self,
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.05,
        verbose=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    ):
        self.freq = None
        self.add_seasonality = None
        self.add_country_holidays = None
        self.growth = "linear"
        self.growth_floor = 0.0
        self.growth_cap = None
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = None
        self.seasonality_mode = "additive"
        self.seasonality_prior_scale = 10.0
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays_prior_scale = 10.0
        self.mcmc_samples = 0
        self.alpha = DEFAULT_ALPHA
        self.uncertainty_samples = 1000
        self.stan_backend = None
        self.verbose = verbose

        super().__init__()

        # import inside method to avoid hard dependency
        from prophet.forecaster import Prophet as _Prophet

        self._ModelClass = _Prophet

    def _instantiate_model(self):
        self._forecaster = self._ModelClass(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=float(self.seasonality_prior_scale),
            holidays_prior_scale=float(self.holidays_prior_scale),
            changepoint_prior_scale=float(self.changepoint_prior_scale),
            mcmc_samples=self.mcmc_samples,
            interval_width=1 - self.alpha,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend,
        )
        return self

    # _fit is defined in the superclass and is fine as it is.

    def _predict(self, fh, X=None):
        """Forecast time series trend at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        fh = self._get_prophet_fh()
        future = pd.DataFrame({"ds": fh}, index=fh)

        out = self._forecaster.setup_dataframe(future.copy())
        out["trend"] = self._forecaster.predict_trend(out)

        y_pred = out.loc[:, "trend"]
        y_pred.index = future.index

        if isinstance(self._y.columns[0], str):
            y_pred.name = self._y.columns[0]
        else:
            y_pred.name = None

        if self.y_index_was_int_ or self.y_index_was_period_:
            y_pred.index = self.fh.to_absolute_index(cutoff=self.cutoff)

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
        params0 = {
            "changepoint_range": 0.8,
            "changepoint_prior_scale": 0.05,
        }

        return params0
