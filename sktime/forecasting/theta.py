# -*- coding: utf-8 -*-

"""Theta forecaster."""

__all__ = ["ThetaForecaster"]
__author__ = ["big-o", "mloning", "kejsitake", "fkiraly"]

from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import norm

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.utils.slope_and_trend import _fit_trend
from sktime.utils.validation.forecasting import check_sp


class ThetaForecaster(ExponentialSmoothing):
    """Theta method for forecasting.

    The theta method as defined in [1]_ is equivalent to simple exponential
    smoothing (SES) with drift (as demonstrated in [2]_).

    The series is tested for seasonality using the test outlined in A&N. If
    deemed seasonal, the series is seasonally adjusted using a classical
    multiplicative decomposition before applying the theta method. The
    resulting forecasts are then reseasonalised.

    In cases where SES results in a constant forecast, the theta forecaster
    will revert to predicting the SES constant plus a linear trend derived
    from the training data.

    Prediction intervals are computed using the underlying state space model.

    Parameters
    ----------
    initial_level : float, optional
        The alpha value of the simple exponential smoothing, if the value is
        set then
        this will be used, otherwise it will be estimated from the data.
    deseasonalize : bool, optional (default=True)
        If True, data is seasonally adjusted.
    sp : int, optional (default=1)
        The number of observations that constitute a seasonal period for a
        multiplicative deseasonaliser, which is used if seasonality is
        detected in the
        training data. Ignored if a deseasonaliser transformer is provided.
        Default is
        1 (no seasonality).

    Attributes
    ----------
    initial_level_ : float
        The estimated alpha value of the SES fit.
    drift_ : float
        The estimated drift of the fitted model.
    se_ : float
        The standard error of the predictions. Used to calculate prediction
        intervals.

    References
    ----------
    .. [1] Assimakopoulos, V. and Nikolopoulos, K. The theta model: a
       decomposition approach to forecasting. International Journal of
       Forecasting 16, 521-530, 2000.
       https://www.sciencedirect.com/science/article/pii/S0169207000000662

    .. [2] `Hyndman, Rob J., and Billah, Baki. Unmasking the Theta method.
       International J. Forecasting, 19, 287-290, 2003.
       https://www.sciencedirect.com/science/article/pii/S0169207001001431

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> y = load_airline()
    >>> forecaster = ThetaForecaster(sp=12)
    >>> forecaster.fit(y)
    ThetaForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _fitted_param_names = ("initial_level", "smoothing_level")
    _tags = {
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, initial_level=None, deseasonalize=True, sp=1):

        self.sp = sp
        self.deseasonalize = deseasonalize
        self.deseasonalizer_ = None
        self.trend_ = None
        self.initial_level_ = None
        self.drift_ = None
        self.se_ = None
        super(ThetaForecaster, self).__init__(initial_level=initial_level, sp=sp)

    def _fit(self, y, X=None, fh=None):
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
        sp = check_sp(self.sp)
        if sp > 1 and not self.deseasonalize:
            warn("`sp` is ignored when `deseasonalise`=False")

        if self.deseasonalize:
            self.deseasonalizer_ = Deseasonalizer(sp=self.sp, model="multiplicative")
            y = self.deseasonalizer_.fit_transform(y)

        self.initialization_method = "known" if self.initial_level else "estimated"
        # fit exponential smoothing forecaster
        # find theta lines: Theta lines are just SES + drift
        super(ThetaForecaster, self)._fit(y, fh=fh)
        self.initial_level_ = self._fitted_forecaster.params["smoothing_level"]

        # compute and store historical residual standard error
        self.sigma_ = np.sqrt(self._fitted_forecaster.sse / (len(y) - 1))

        # compute trend
        self.trend_ = self._compute_trend(y)

        return self

    def _predict(self, fh, X=None):
        """Make forecasts.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        y_pred = super(ThetaForecaster, self)._predict(fh, X)

        # Add drift.
        drift = self._compute_drift()
        y_pred += drift

        if self.deseasonalize:
            y_pred = self.deseasonalizer_.inverse_transform(y_pred)

        return y_pred

    @staticmethod
    def _compute_trend(y):
        # Trend calculated through least squares regression.
        coefs = _fit_trend(y.values.reshape(1, -1), order=1)
        return coefs[0, 0] / 2

    def _compute_drift(self):
        fh = self.fh.to_relative(self.cutoff)
        if np.isclose(self.initial_level_, 0.0):
            # SES was constant, so revert to simple trend
            drift = self.trend_ * fh
        else:
            # Calculate drift from SES parameters
            n_timepoints = len(self._y)
            drift = self.trend_ * (
                fh
                + (1 - (1 - self.initial_level_) ** n_timepoints) / self.initial_level_
            )

        return drift

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and predict_interval

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # prepare return data frame
        index = pd.MultiIndex.from_product([["Quantiles"], alpha])
        pred_quantiles = pd.DataFrame(columns=index)

        sem = self.sigma_ * np.sqrt(
            self.fh.to_relative(self.cutoff) * self.initial_level_**2 + 1
        )

        y_pred = self._predict(fh, X)

        # we assume normal additive noise with sem variance
        for a in alpha:
            pred_quantiles[("Quantiles", a)] = y_pred + norm.ppf(a) * sem
        # todo: should this not increase with the horizon?
        # i.e., sth like norm.ppf(a) * sem * fh.to_absolute(cutoff) ?
        # I've just refactored this so will leave it for now

        return pred_quantiles

    def _update(self, y, X=None, update_params=True):
        super(ThetaForecaster, self)._update(
            y, X, update_params=False
        )  # use custom update_params routine
        if update_params:
            if self.deseasonalize:
                y = self.deseasonalizer_.transform(self._y)  # use updated y
            self.initial_level_ = self._fitted_forecaster.params["smoothing_level"]
            self.trend_ = self._compute_trend(y)
        return self


def _zscore(level: float, two_tailed: bool = True) -> float:
    """Calculate a z-score from a confidence level.

    Parameters
    ----------
    level : float
        A confidence level, in the open interval (0, 1).
    two_tailed : bool (default=True)
        If True, return the two-tailed z score.

    Returns
    -------
    z : float
        The z score.
    """
    alpha = 1 - level
    if two_tailed:
        alpha /= 2
    return -norm.ppf(alpha)
