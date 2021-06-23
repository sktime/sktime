#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Prophet forecaster by wrapping fbprophet."""

__author__ = ["Martin Walter"]
__all__ = ["Prophet"]

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base.adapters import _ProphetAdapter
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("fbprophet")


class Prophet(_ProphetAdapter):
    """Prophet forecaster by wrapping fbprophet.

    Parameters
    ----------
    freq: String of DatetimeIndex frequency. See here for possible values:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        #timeseries-offset-aliases
    add_seasonality: Dict with args for Prophet.add_seasonality().
        Dict can have the following keys/values:
            name: string name of the seasonality component.
            period: float number of days in one period.
            fourier_order: int number of Fourier components to use.
            prior_scale: optional float prior scale for this component.
            mode: optional 'additive' or 'multiplicative'
            condition_name: string name of the seasonality condition.
    add_country_holidays: Dict with args for Prophet.add_country_holidays().
        Dict can have the following keys/values:
            country_name: Name of the country, like 'UnitedStates' or 'US'
    growth: String 'linear' or 'logistic' to specify a linear or logistic
        trend.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_prior_scale: Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    alpha: Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: Number of simulated draws used to estimate
        uncertainty intervals. Settings this value to 0 or False will disable
        uncertainty estimation and speed up the calculation.
    stan_backend: str as defined in StanBackendEnum default: None - will try to
        iterate over all available backends and find the working one

    References
    ----------
    https://facebook.github.io/prophet
    https://github.com/facebook/prophet

    Example
    ----------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.fbprophet import Prophet
    >>> # Prophet requires to have data with a pandas.DatetimeIndex
    >>> y = load_airline().to_timestamp(freq='M')
    >>> forecaster = Prophet(
    ...     seasonality_mode='multiplicative',
    ...     n_changepoints=int(len(y) / 12),
    ...     add_country_holidays={'country_name': 'Germany'},
    ...     yearly_seasonality=True)
    >>> forecaster.fit(y)
    Prophet(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    def __init__(
        self,
        # Args due to wrapping
        freq=None,
        add_seasonality=None,
        add_country_holidays=None,
        # Args of fbprophet
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        alpha=DEFAULT_ALPHA,
        uncertainty_samples=1000,
        stan_backend=None,
        verbose=0,
    ):
        self.freq = freq
        self.add_seasonality = add_seasonality
        self.add_country_holidays = add_country_holidays

        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.mcmc_samples = mcmc_samples
        self.alpha = alpha
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        self.verbose = verbose

        # import inside method to avoid hard dependency
        from fbprophet.forecaster import Prophet as _Prophet

        self._ModelClass = _Prophet

        super(Prophet, self).__init__()

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
