#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements Bagging Forecaster."""

__author__ = ["ltsaprounis"]

from typing import List

import pandas as pd
from sklearn import clone

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.ets import AutoETS
from sktime.transformations.base import BaseTransformer
from sktime.transformations.bootstrap import STLBootstrapTransformer


class BaggingForecaster(BaseForecaster):
    """Bagged "Bootrstrap Aggregating" Forecasts.

    Bagged Forecasts are obtained by forecasting bootsrapped time series and then
    aggregating the resulting forecasts. For the point forecast, the different forecasts
    are aggregated using the mean function [1]. Prediction intervals and quantiles are
    calculated for each time point in the forecasting horizon by calculating the sampled
    forecast quantiles.

    Bergmeir et al. (2016) [2] show that, on average, bagging ETS forecasts gives better
    forecasts than just applying ETS directly.

    Parameters
    ----------
    bootstrapping_transformer : BaseTransformer
        Bootrstrapping Transformer that takes a series as input and returns a panel
        of bootstrapped time series
    forecaster : BaseForecaster
        A valid sktime Forecaster

    See Also
    --------
    sktime.transformations.bootstrap.MovingBlockBootstrapTransformer :
        Transofrmer that applies the Moving Block Bootstrapping method to create
        a panel of synthetic time series.

    sktime.transformations.bootstrap.STLBootstrapTransformer :
        Transofrmer that utilises BoxCox, STL and Moving Block Bootstrapping to create
        a panel of similar time series.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3,
        Chapter 12.5. Accessed on February 13th 2022.
    .. [2] Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential
        smoothing methods using STL decomposition and Box-Cox transformation.
        International Journal of Forecasting, 32(2), 303-312

    Examples
    --------
    >>> from sktime.transformations.bootstrap import STLBootstrapTransformer
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import BaggingForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = BaggingForecaster(
    ...     STLBootstrapTransformer(sp=12), NaiveForecaster(sp=12)
    ... )
    >>> forecaster.fit(y)
    BaggingForecaster(...)
    >>> y_hat = forecaster.predict([1,2,3])
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": True,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
        "capability:predict_quantiles": True,
    }

    def __init__(
        self, bootstrap_transformer: BaseTransformer, forecaster: BaseForecaster
    ):
        self.bootstrap_transformer = bootstrap_transformer
        self.forecaster = forecaster

        super(BaggingForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        if (
            self.bootstrap_transformer.get_tag(
                "scitype:transform-input", raise_error=False
            )
            != "Series"
            and self.bootstrap_transformer.get_tag(
                "scitype:transform-output", raise_error=False
            )
            != "Panel"
            and not isinstance(self.bootstrap_transformer, BaseTransformer)
        ):
            raise TypeError(
                "bootstrap_transformer in BaggingForecaster should be a Transformer "
                "that take as input a Series and output a Panel."
            )

        if not isinstance(self.forecaster, BaseForecaster):
            raise TypeError(
                "forecaster in BaggingForecaster should be an sktime Forecaster"
            )

        self.bootstrap_transformer_ = clone(self.bootstrap_transformer)
        self.forecaster_ = clone(self.forecaster)
        y_bootstraps = self.bootstrap_transformer_.fit_transform(y)
        self.forecaster_.fit(y_bootstraps)

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

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
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_bootstraps_pred = self.forecaster_.predict(fh)
        return y_bootstraps_pred.groupby(level=-1).mean()

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        y_pred = self.forecaster_.predict(fh, X)

        return _calculate_data_quantiles(y_pred, alpha)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {
                "forecaster": AutoETS(sp=1),
                "bootstrap_transformer": STLBootstrapTransformer(sp=3),
            }
        ]

        return params


def _calculate_data_quantiles(df: pd.DataFrame, alpha: List[float]) -> pd.DataFrame:
    """Generate quantiles for each time point.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of mtype pd-multiindex or hierarchical
    alpha : List[float]
        list of the desired quantiles

    Returns
    -------
    pd.DataFrame
        The specified quantiles
    """
    index = pd.MultiIndex.from_product([["Quantiles"], alpha])
    pred_quantiles = pd.DataFrame(columns=index)
    for a in alpha:
        pred_quantiles[("Quantiles", a)] = (
            df.groupby(level=-1, as_index=True).quantile(a).squeeze()
        )

    return pred_quantiles
