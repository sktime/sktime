#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements ensemble forecasters.

Creates univariate (optionally weighted)
combination of the predictions from underlying forecasts.
"""

__author__ = ["mloning", "GuzalBulatova", "aiwalter"]
__all__ = ["EnsembleForecaster", "AutoEnsembleForecaster"]

import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.pipeline import Pipeline
from sklearn.utils.stats import _weighted_percentile

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation.forecasting import check_regressor


class AutoEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Automatically find best weights for the ensembled forecasters.

    The AutoEnsembleForecaster uses a meta-model (regressor) to calculate the optimal
    weights for ensemble aggregation with mean. The regressor has to be sklearn-like
    and needs to have either an attribute `feature_importances_` or `coef_`, as this
    is used as weights. Regressor can also be a sklearn.Pipeline.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    regressor : sklearn-like regressor, optional, default=None.
        Used to infer optimal weights from coefficients (linear models) or from
        feature importance scores (decision tree-based models). If None, then
        a GradientBoostingRegressor(max_depth=5) is used.
        The regressor can also be a sklearn.Pipeline().
    test_size : int or float, optional, default=None
        Used to do an internal temporal_train_test_split(). The test_size data
        will be the endog data of the regressor and it is the most recent data.
        The exog data of the regressor are the predictions from the temporarily
        trained ensemble models. If None, it will be set to 0.25.
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor.
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.

    Attributes
    ----------
    regressor_ : sklearn-like regressor
        Fitted regressor.
    weights_ : np.array
        The weights based on either regressor.feature_importances_ or
        regressor.coef_ values.

    See Also
    --------
    EnsembleForecaster

    Examples
    --------
    >>> from sktime.forecasting.compose import AutoEnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [
    ...     ("trend", PolynomialTrendForecaster()),
    ...     ("naive", NaiveForecaster()),
    ... ]
    >>> forecaster = AutoEnsembleForecaster(forecasters=forecasters)
    >>> forecaster.fit(y=y, X=None, fh=[1,2,3])
    AutoEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(
        self,
        forecasters,
        regressor=None,
        test_size=None,
        random_state=None,
        n_jobs=None,
    ):
        super(AutoEnsembleForecaster, self).__init__(
            forecasters=forecasters,
            n_jobs=n_jobs,
        )
        self.regressor = regressor
        self.test_size = test_size
        self.random_state = random_state

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        _, forecasters = self._check_forecasters()
        self.regressor_ = check_regressor(
            regressor=self.regressor, random_state=self.random_state
        )

        # get training data for meta-model
        if X is not None:
            y_train, y_test, X_train, X_test = temporal_train_test_split(
                y, X, test_size=self.test_size
            )
        else:
            y_train, y_test = temporal_train_test_split(y, test_size=self.test_size)
            X_train, X_test = None, None

        # fit ensemble models
        fh_regressor = ForecastingHorizon(y_test.index, is_relative=False)
        self._fit_forecasters(forecasters, y_train, X_train, fh_regressor)
        X_meta = pd.concat(self._predict_forecasters(fh_regressor, X_test), axis=1)

        # fit meta-model (regressor) on predictions of ensemble models
        # with y_test as endog/target
        self.regressor_.fit(X=X_meta, y=y_test)

        # check if regressor is a sklearn.Pipeline
        if isinstance(self.regressor_, Pipeline):
            # extract regressor from pipeline to access its attributes
            self.weights_ = _get_weights(self.regressor_.steps[-1][1])
        else:
            self.weights_ = _get_weights(self.regressor_)

        # fit forecasters with all data
        self._fit_forecasters(forecasters, y, X, fh)

        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame
        return_pred_int : boolean, optional, default=False
        alpha : fh : float, default=DEFAULT_ALPHA

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        y_pred_df = pd.concat(self._predict_forecasters(fh, X), axis=1)
        # apply weights
        y_pred = y_pred_df.apply(lambda x: np.average(x, weights=self.weights_), axis=1)
        return y_pred


def _get_weights(regressor):
    # tree-based models from sklearn which have feature importance values
    if hasattr(regressor, "feature_importances_"):
        weights = regressor.feature_importances_
    # linear regression models from sklearn which have coefficient values
    elif hasattr(regressor, "coef_"):
        weights = regressor.coef_
    else:
        raise NotImplementedError(
            """The given regressor is not supported. It must have
            either an attribute feature_importances_ or coef_ after fitting."""
        )
    # avoid ZeroDivisionError if all weights are 0
    if weights.sum() == 0:
        weights += 1
    return list(weights)


class EnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Ensemble of forecasters.

    Overview: Input one series of length `n` and EnsembleForecaster performs
    fitting and prediction for each estimator passed in `forecasters`. It then
    applies `aggfunc` aggregation function by row to the predictions dataframe
    and returns final prediction - one series.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    aggfunc : str, {'mean', 'median', 'min', 'max'}, default='mean'
        The function to aggregate prediction from individual forecasters.
    weights : list of floats
        Weights to apply in aggregation.

    See Also
    --------
    AutoEnsembleForecaster

    Examples
    --------
    >>> from sktime.forecasting.compose import EnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [
    ...     ("trend", PolynomialTrendForecaster()),
    ...     ("naive", NaiveForecaster()),
    ... ]
    >>> forecaster = EnsembleForecaster(forecasters=forecasters, weights=[4, 10])
    >>> forecaster.fit(y=y, X=None, fh=[1,2,3])
    EnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, n_jobs=None, aggfunc="mean", weights=None):
        super(EnsembleForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.aggfunc = aggfunc
        self.weights = weights

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame
        return_pred_int : boolean, optional, default=False
        alpha : fh : float, default=DEFAULT_ALPHA

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1)
        y_pred = _aggregate(y=y_pred, aggfunc=self.aggfunc, weights=self.weights)

        return y_pred


def _aggregate(y, aggfunc, weights):
    """Apply aggregation function by row.

    Parameters
    ----------
    y : pd.DataFrame
        Multivariate series to transform.
    aggfunc : str
        Aggregation function used for transformation.
    weights : list of floats
        Weights to apply in aggregation.

    Returns
    -------
    y_agg: pd.Series
        Transformed univariate series.
    """
    if weights is None:
        aggfunc = _check_aggfunc(aggfunc, weighted=False)
        y_agg = aggfunc(y, axis=1)
    else:
        aggfunc = _check_aggfunc(aggfunc, weighted=True)
        y_agg = aggfunc(y, axis=1, weights=np.array(weights))

    return pd.Series(y_agg, index=y.index)


def _check_aggfunc(aggfunc, weighted=False):
    _weighted = "weighted" if weighted else "unweighted"
    valid_aggfuncs = {
        "mean": {"unweighted": np.mean, "weighted": np.average},
        "median": {"unweighted": np.median, "weighted": _weighted_median},
        "min": {"unweighted": np.min, "weighted": _weighted_min},
        "max": {"unweighted": np.max, "weighted": _weighted_max},
        "gmean": {"unweighted": gmean, "weighted": gmean},
    }
    if aggfunc not in valid_aggfuncs.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)
    return valid_aggfuncs[aggfunc][_weighted]


def _weighted_median(y, axis=1, weights=None):
    w_median = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y.values,
        sample_weight=weights,
        percentile=50,
    )
    return w_median


def _weighted_min(y, axis=1, weights=None):
    w_min = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y.values,
        sample_weight=weights,
        percentile=0,
    )
    return w_min


def _weighted_max(y, axis=1, weights=None):
    w_max = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y.values,
        sample_weight=weights,
        percentile=100,
    )
    return w_max
