# -*- coding: utf-8 -*-

"""Implements naive variance functionality to tune forecasters."""

__author__ = ["Ilyas Moutawwakil"]


import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_fh


class NaiveVariance(BaseForecaster):
    """
    Compute the prediction variance using the 'simple strategy'.

    Parameters
    ----------
    fh : int, list, np.array or ForecastingHorizon
        Forecasting horizon
    X : pd.DataFrame, optional (default=None)
        Exogenous time series
    cov : bool, optional (default=False)
        Wether to return marginal variances or covariance matrices.
    """

    _tags = {
        "handles-missing-data": False,
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        # deprecated and will be renamed to capability:predict_quantiles in 0.11.0
        "capability:pred_var": True,
        # deprecated and will be renamed to capability:predict_variance in 0.11.0
    }

    def __init__(self, forecaster):

        self.forecaster = forecaster
        super(NaiveVariance, self).__init__()

        tags_to_clone = [
            "requires-fh-in-fit",
            "scitype:y",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(forecaster, tags_to_clone)

    def _fit(self, y, X=None, fh=None):
        return self.forecaster.fit(y, X, fh)

    # todo: implement this, mandatory
    def _predict(self, fh, X=None):
        return self.forecaster.predict(fh, X)

    def _predict_quantiles(self, fh, X=None, alpha=0.5):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and predict_interval

        If alpha is iterable, multiple quantiles will be calculated.

        Users can also implement _predict_interval if calling it makes this faster.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : float or list of float, optional (default=0.5)
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        # implement here

    def _predict_var(self, fh, X=None, cov=False):
        """
        Compute/return prediction variance for a forecast.

        Must be run *after* the forecaster has been fitted.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            Wether to return marginal variances or covariance matrices.

        Returns
        -------
        pred_var : np.ndarray
            if cov=False,
                a vector of same length as fh with predictive marginal variances;
            if cov=True,
                a square matrix of size len(fh) with predictive covariance matrix.
        """
        self.forecaster.check_is_fitted()

        fh_ = check_fh(fh)

        n_timepoints = len(self._y)
        residuals = []
        for i in range(0, n_timepoints):
            # a duplicate for rolling window fit/predict
            forecaster = clone(self.forecaster)

            try:
                forecaster.fit(self._y[: i + 1])
                y_hat = forecaster.predict(self._y.index, self._X)
                residuals.append(y_hat - self._y)
            except Exception:
                # warn ?
                pass

        if len(residuals) == 0:
            # eiter raise an error or warn (not enough data)
            pass

        residuals = np.array(residuals)

        if cov:
            # working on it
            pass
        else:
            pred_var = pd.Series(
                [(np.diagonal(residuals, offset=offset) ** 2).mean() for offset in fh_],
                index=fh_,
            )

        return pred_var


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
