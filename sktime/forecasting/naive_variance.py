# -*- coding: utf-8 -*-

"""Implements naive variance functionality to tune forecasters."""

__author__ = ["IlyasMoutawwakil"]


import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_alpha, check_fh


class NaiveVariance(BaseForecaster):
    """Compute the prediction variance using a naive strategy.

    The strategy works as follows:
    -train the internal forecaster on subsets of `self._y` (rolling window)
    -computes the residuals of the rest of the time series for each subset
    -for each point k in `fh`, computes the variance of the prediction
        by averaging the squared residuals that are k steps ahead.
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

        self.forecaster = clone(forecaster)
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

    def _predict(self, fh, X=None):
        return self.forecaster.predict(fh, X)

    def _predict_quantiles(self, fh, X=None, alpha=0.5):
        """Compute/return prediction quantiles for a forecast.

        Uses normal distribution as predictive distribution to compute the
        quantiles.

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

        y_pred = self.predict(fh, X)
        pred_var = self.predict_var(fh, X)

        z_scores = norm.ppf(alpha)
        errors = [pred_var * z for z in z_scores]

        pred_quantiles = pd.DataFrame()
        for a, error in zip(alpha, errors):
            pred_quantiles[a] = y_pred + error

        return pred_quantiles

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
            If True, return the covariance matrix.
            If False, return the marginal variance.

        Returns
        -------
        pred_var : 
            if cov=False, pd.Series with index fh.
                a vector of same length as fh with predictive marginal variances;
            if cov=True, pd.DataFrame with index fh and columns fh.
                a square matrix of size len(fh) with predictive covariance matrix.
        """

        residuals_matrix = pd.DataFrame(
            columns=self._y.index[1:], index=self._y.index[1:])
        for id in residuals_matrix.columns:
            forecaster = clone(self.forecaster)

            subset = self._y[:id]
            forecaster.fit(subset)

            # predict_residuals not supported yet in many forecasters
            y_true = self._y[id:]
            y_hat = forecaster.predict(y_true.index, self._X)

            residuals_matrix[id] = y_true - y_hat

        residuals_matrix = residuals_matrix.T
        variance = [(np.diagonal(residuals_matrix, offset=offset) ** 2).mean()
                    for offset in fh.to_relative(self.cutoff)]

        if cov:
            pred_var = pd.DataFrame(np.diag(variance), index=fh.to_absolute(
                self.cutoff), columns=fh.to_absolute(self.cutoff))
        else:
            pred_var = pd.Series(
                variance,
                index=fh.to_absolute(self.cutoff),
            )

        return pred_var
