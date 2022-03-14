# -*- coding: utf-8 -*-

"""Implements naive variance functionality to tune forecasters."""

__author__ = ["IlyasMoutawwakil"]

import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster


class NaiveVariance(BaseForecaster):
    """Compute the prediction variance using a naive strategy.

    The strategy works as follows:
    -train the internal forecaster on subsets of `self._y` (rolling window)
    -computes the residuals of the rest of the time series for each subset
    -for each point k in `fh`, computes the variance of the prediction
        by averaging the squared residuals that are k steps ahead.
    """

    _required_parameters = ["forecaster"]
    _tags = {
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": False,
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
            "ignores-exogeneous-X",
            "handles-missing-data",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.forecaster, tags_to_clone)

    def _fit(self, y, X=None, fh=None):
        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(y=y, X=X, fh=fh)
        return self

    def _predict(self, fh, X=None):
        return self.forecaster_.predict(fh=fh, X=X)

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.
        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)
        Returns
        -------
        self : an instance of self
        """

        self.forecaster_.update(y, X, update_params=update_params)
        return self

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
        errors = [pred_var ** 0.5 * z for z in z_scores]

        index = pd.MultiIndex.from_product([["Quantiles"], alpha])
        pred_quantiles = pd.DataFrame(columns=index)
        for a, error in zip(alpha, errors):
            pred_quantiles[("Quantiles", a)] = y_pred + error

        return pred_quantiles

    def _predict_var(self, fh, X=None, cov=False):
        """Compute/return prediction variance for a forecast.

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
        y_index = self._y.index
        fh_relative = fh.to_relative(self.cutoff)
        fh_absolute = fh.to_absolute(self.cutoff)

        residuals_matrix = pd.DataFrame(columns=y_index, index=y_index, dtype="float")
        for id in y_index:
            forecaster = clone(self.forecaster)
            subset = self._y[:id]  # subset on which we fit
            try:
                forecaster.fit(subset)
            except ValueError:
                sys.stdout.write(
                    f"Couldn't fit the model on time series of length {len(subset)}.\n"
                )
                continue

            y_true = self._y[id:]  # subset on which we predict
            try:
                residuals_matrix.loc[id] = forecaster.predict_residuals(y_true, self._X)
            except IndexError:
                sys.stdout.write(
                    f"Couldn't predict on time series of length {len(subset)}.\n"
                )

        if cov:
            fh_size = len(fh)
            covariance = np.zeros(shape=(len(fh), fh_size))
            for i in range(fh_size):
                i_residuals = np.diagonal(residuals_matrix, offset=fh_relative[i])
                for j in range(i, fh_size):  # since the matrix is symmetric
                    j_residuals = np.diagonal(residuals_matrix, offset=fh_relative[j])
                    max_residuals = min(len(j_residuals), len(i_residuals))
                    covariance[i, j] = covariance[j, i] = np.nanmean(
                        i_residuals[:max_residuals] * j_residuals[:max_residuals]
                    )
            pred_var = pd.DataFrame(
                covariance,
                index=fh_absolute,
                columns=fh_absolute,
            )
        else:
            variance = [
                np.nanmean(np.diagonal(residuals_matrix, offset=offset) ** 2)
                for offset in fh_relative
            ]
            pred_var = pd.Series(
                variance,
                index=fh_absolute,
            )

        return pred_var

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params_list = {"forecaster": FORECASTER}

        return params_list
