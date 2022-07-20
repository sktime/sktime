#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Prophet forecaster by wrapping fbprophet."""

__author__ = ["aiwalter"]
__all__ = ["Prophet"]


from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base.adapters import _ProphetAdapter
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("prophet", severity="warning")


class Prophet(_ProphetAdapter):
    """Prophet forecaster by wrapping Facebook's prophet algorithm [1]_.

    Direct interface to Facebook prophet, using the sktime interface.
    All hyper-parameters are exposed via the constructor.

    Data can be passed in one of the sktime compatible formats,
    naming a column `ds` such as in the prophet package is not necessary.

    Integer indices can also be passed, in which case internally a conversion
    to days since Jan 1, 2000 is carried out before passing to prophet.

    Parameters
    ----------
    freq: str, default=None
        A DatetimeIndex frequency. For possible values see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    add_seasonality: dict or None, default=None
        Dict with args for Prophet.add_seasonality().
        Dict can have the following keys/values:
            * name: string name of the seasonality component.
            * period: float number of days in one period.
            * fourier_order: int number of Fourier components to use.
            * prior_scale: optional float prior scale for this component.
            * mode: optional 'additive' or 'multiplicative'
            * condition_name: string name of the seasonality condition.
    add_country_holidays: dict or None, default=None
        Dict with args for Prophet.add_country_holidays().
        Dict can have the following keys/values:
            country_name: Name of the country, like 'UnitedStates' or 'US'
    growth: str, default="linear"
        String 'linear' or 'logistic' to specify a linear or logistic
        trend. If 'logistic' specified float for 'growth_cap' must be provided.
    growth_floor: float, default=0
        Growth saturation minimum value.
        Used only if  `growth="logistic"`, has no effect otherwise
        (if `growth` is not `"logistic"`).
    growth_cap: float, default=None
        Growth saturation maximum aka carrying capacity.
        Mandatory (float) iff `growth="logistic"`, has no effect and is optional,
        otherwise (if `growth` is not `"logistic"`).
    changepoints: list or None, default=None
        List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: int, default=25
        Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: float, default=0.8
        Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: str or bool or int, default="auto"
        Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: str or bool or int, default="auto"
        Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: str or bool or int, default="auto"
        Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame or None, default=None
        pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: str, default='additive'
        Take one of 'additive' or 'multiplicative'.
    seasonality_prior_scale: float, default=10.0
        Parameter modulating the strength of the seasonality model.
        Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: float, default=10.0
        Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: float, default=0.05
        Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: int, default=0
        If greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    alpha: float, default=0.05
        Width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: int, default=1000
        Number of simulated draws used to estimate uncertainty intervals.
        Settings this value to 0 or False will disable
        uncertainty estimation and speed up the calculation.
    stan_backend: str or None, default=None
        str as defined in StanBackendEnum. If None, will try to
        iterate over all available backends and find the working one.

    References
    ----------
    .. [1] https://facebook.github.io/prophet

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.fbprophet import Prophet
    >>> # Prophet requires to have data with a pandas.DatetimeIndex
    >>> y = load_airline().to_timestamp(freq='M')
    >>> forecaster = Prophet(  # doctest: +SKIP
    ...     seasonality_mode='multiplicative',
    ...     n_changepoints=int(len(y) / 12),
    ...     add_country_holidays={'country_name': 'Germany'},
    ...     yearly_seasonality=True)
    >>> forecaster.fit(y)  # doctest: +SKIP
    Prophet(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    def __init__(
        self,
        # Args due to wrapping
        freq=None,
        add_seasonality=None,
        add_country_holidays=None,
        # Args of fbprophet
        growth="linear",
        growth_floor=0.0,
        growth_cap=None,
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
        self.growth_floor = growth_floor
        self.growth_cap = growth_cap
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

        super(Prophet, self).__init__()

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
        params : dict or list of dict
        """
        params = {
            "n_changepoints": 0,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "uncertainty_samples": 10,
            "verbose": False,
        }
        return params
