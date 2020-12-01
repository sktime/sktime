#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter"]
__all__ = ["Prophet"]

import pandas as pd
import numpy as np

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.check_imports import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_X, check_y_X


_check_soft_dependencies("fbprophet")


class Prophet(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """Prophet forecaster by wrapping fbprophet.
    Parameters
    ----------
    freq: String of DatetimeIndex frequency. See here for possible values:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    add_seasonality: Dictionary with arguments for the Prophet.add_seasonality() function.
        Dictionary can have the following keys/values:
            name: string name of the seasonality component.
            period: float number of days in one period.
            fourier_order: int number of Fourier components to use.
            prior_scale: optional float prior scale for this component.
            mode: optional 'additive' or 'multiplicative'
            condition_name: string name of the seasonality condition.
    add_country_holidays: Dictionary with arguments for the Prophet.add_country_holidays() function.
        Dictionary can have the following keys/values:
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
    interval_width: Float, width of the uncertainty intervals provided
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

    """

    def __init__(
        self,
        # Args due to wrapping
        freq,
        add_seasonality=None,
        add_country_holidays=None,
        # Args of fbprophet
        growth='linear',
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto',
        holidays=None,
        seasonality_mode='additive',
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=1 - DEFAULT_ALPHA,
        uncertainty_samples=1000,
        stan_backend=None,
        **kwargs
    ):
        self.freq = freq
        self.add_seasonality = add_seasonality
        self.add_country_holidays = add_country_holidays

        self.growth = growth
        self.changepoints = changepoints
        if self.changepoints is not None:
            self.changepoints = pd.Series(pd.to_datetime(self.changepoints), name='ds')
            self.n_changepoints = len(self.changepoints)
            self.specified_changepoints = True
        else:
            self.n_changepoints = n_changepoints
            self.specified_changepoints = False

        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays

        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)

        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        self.stan_backend = stan_backend

        super(Prophet, self).__init__()

        # import inside method to avoid hard dependency
        from fbprophet.forecaster import Prophet as _Prophet

        self._forecaster = _Prophet(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            stan_backend=stan_backend,
            **kwargs
        )

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
        y, X = check_y_X(y, X, index_type=pd.DatetimeIndex)
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
        if type(fh._values) == pd.DatetimeIndex:
            df = pd.DataFrame()
            df["ds"] = fh._values
        else:
            # Try to create pd.DatetimeIndex
            try:
                periods = fh.to_pandas().max()
                periods = fh.to_pandas().max()
                df = self._forecaster.make_future_dataframe(
                    periods=periods + 1, freq=self.freq, include_history=False)
                df = df.iloc[fh.to_pandas()]
            except Exception:
                raise TypeError("Type of fh values must be int, np.array, list or pd.DatetimeIndex")

        # Merge X with df (of created future DatetimeIndex values)
        merge_error = "Either length of fh and X must be " \
            "same or X must have future DatetimeIndex values."
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

        # Prediction
        out = self._forecaster.predict(df)
        out.index = out["ds"]
        pred = out["yhat"]
        pred = pred.rename(None)
        pred_int = out[["yhat_upper", "yhat_lower"]].rename(columns={
            "yhat_upper": "upper",
            "yhat_lower": "lower"})
        if return_pred_int:
            return pred, pred_int
        else:
            return pred

    def update(self, y, X=None, fh=None):
        """Warm-starting with an existing model.

        Prophet models can only be fit once, and a new model must
        be re-fit when new data become available. In most settings,
        model fitting is fast enough that there isn’t any issue with
        re-fitting from scratch. However, it is possible to speed things
        up a little by warm-starting the fit from the model parameters
        of the earlier model. We retrieve parameters from a trained model
        in the format used to initialize a new Stan model.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        ----------
        model : sktime.Prophet
            A fitted sktime.Prophet model

        Raises
        ----------
        ValueError
            Error when data does not contain old training data,
            because fbprophet requires it.
        """
        if len(self._forecaster.history) >= len(y):
            raise ValueError("y must contain past and new train data.")
        if fh is None:
            fh = self.fh
        model_params = self._get_model_params()
        fitted_params = self._get_fitted_params()
        model = Prophet(**model_params)
        model.fit(y=y, X=X, fh=fh, init=fitted_params)
        return model

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
        for name in ['k', 'm', 'sigma_obs']:
            fitted_params[name] = self._forecaster.params[name][0][0]
        for name in ['delta', 'beta']:
            fitted_params[name] = self._forecaster.params[name][0]
        return fitted_params
