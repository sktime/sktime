#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements ensemble forecasters.

Creates univariate (optionally weighted) combination of the predictions from underlying
forecasts.
"""

__author__ = ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"]
__all__ = ["EnsembleForecaster", "AutoEnsembleForecaster"]

import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.pipeline import Pipeline

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.stats import (
    _weighted_geometric_mean,
    _weighted_max,
    _weighted_median,
    _weighted_min,
)
from sktime.utils.validation.forecasting import check_regressor

VALID_AGG_FUNCS = {
    "mean": {"unweighted": np.mean, "weighted": np.average},
    "median": {"unweighted": np.median, "weighted": _weighted_median},
    "min": {"unweighted": np.min, "weighted": _weighted_min},
    "max": {"unweighted": np.max, "weighted": _weighted_max},
    "gmean": {"unweighted": gmean, "weighted": _weighted_geometric_mean},
}


class AutoEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Automatically find best weights for the ensembled forecasters.

    The AutoEnsembleForecaster finds optimal weights for the ensembled forecasters
    using given method or a meta-model (regressor) .
    The regressor has to be sklearn-like and needs to have either an attribute
    ``feature_importances_`` or ``coef_``, as this is used as weights.
    Regressor can also be a sklearn.Pipeline.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    method : str, optional, default="feature-importance"
        Strategy used to compute weights. Available choices:
        - "feature-importance": use the ``feature_importances_`` or ``coef_`` from
          given ``regressor`` as optimal weights.
        - "inverse-variance": use the inverse variance of the forecasting error
          (based on the internal train-test-split) to compute optimal weights.
    regressor : sklearn-like regressor, optional, default=None.
        Used to infer optimal weights from coefficients (linear models) or from
        feature importance scores (decision tree-based models). If None, then
        a GradientBoostingRegressor(max_depth=5) is used.
    test_size : int or float, optional, default=None
        Used to do an internal temporal_train_test_split(). If None, set to 0.25.
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor.
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. -1 means using all processors.

    Attributes
    ----------
    regressor_ : sklearn-like regressor
        Fitted regressor.
    weights_ : np.array
        The weights based on either ``regressor.feature_importances_`` or
        ``regressor.coef_`` values.

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
    >>> forecaster.fit(y=y, fh=[1, 2, 3])
    AutoEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"],
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "scitype:y": "univariate",
        "capability:random_state": True,
        "property:randomness": "derandomized",
    }

    def __init__(
        self,
        forecasters,
        method="feature-importance",
        regressor=None,
        test_size=None,
        random_state=None,
        n_jobs=None,
    ):
        self.method = method
        self.regressor = regressor
        self.test_size = test_size
        self.random_state = random_state

        super().__init__(
            forecasters=forecasters,
            n_jobs=n_jobs,
        )

    def _fit(self, y, X, fh):
        """Fit to training data."""
        forecasters = [x[1] for x in self.forecasters_]

        if X is not None:
            y_train, y_test, X_train, X_test = temporal_train_test_split(
                y, X, test_size=self.test_size
            )
        else:
            y_train, y_test = temporal_train_test_split(y, test_size=self.test_size)
            X_train, X_test = None, None

        fh_test = ForecastingHorizon(y_test.index, is_relative=False)
        self._fit_forecasters(forecasters, y_train, X_train, fh_test)

        if self.method == "feature-importance":
            self.regressor_ = check_regressor(
                regressor=self.regressor, random_state=self.random_state
            )
            X_meta = pd.concat(self._predict_forecasters(fh_test, X_test), axis=1)
            X_meta.columns = pd.RangeIndex(len(X_meta.columns))
            self.regressor_.fit(X=X_meta, y=y_test)

            if isinstance(self.regressor_, Pipeline):
                self.weights_ = _get_weights(self.regressor_.steps[-1][1])
            else:
                self.weights_ = _get_weights(self.regressor_)

        elif self.method == "inverse-variance":
            inv_var = np.array(
                [
                    1 / np.var(y_test - y_pred_test)
                    for y_pred_test in self._predict_forecasters(fh_test, X_test)
                ]
            )
            self.weights_ = list(inv_var / np.sum(inv_var))
        else:
            raise NotImplementedError(f"Given method {self.method} does not exist.")

        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _predict(self, fh, X):
        """Return the predicted reduction."""
        y_pred_df = pd.concat(self._predict_forecasters(fh, X), axis=1)
        y_pred = y_pred_df.apply(lambda x: np.average(x, weights=self.weights_), axis=1)
        y_pred.name = self._y.name
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params1 = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}
        params2 = {
            "forecasters": [("f1", FORECASTER), ("f2", FORECASTER)],
            "method": "inverse-variance",
            "regressor": LinearRegression(),
            "test_size": 0.2,
        }
        return [params1, params2]


def _get_weights(regressor):
    if hasattr(regressor, "feature_importances_"):
        weights = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        weights = regressor.coef_
    else:
        raise NotImplementedError("The given regressor is not supported.")
    if weights.sum() == 0:
        weights += 1
    return list(weights)


class EnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Ensemble of forecasters.

    Overview: Input one series of length ``n`` and EnsembleForecaster performs
    fitting and prediction for each estimator passed in ``forecasters``. It then
    applies ``aggfunc`` aggregation function by row to the predictions dataframe
    and returns final prediction - one series.

    Parameters
    ----------
    forecasters : list of estimator, (str, estimator), or (str, estimator, count) tuples
        Estimators to apply to the input series.
        * (str, estimator) tuples: the string is a name for the estimator.
        * estimator without string will be assigned unique name based on class name
        * (str, estimator, count) tuples: the estimator will be replicated count times.
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. -1 means using all processors.
    aggfunc : str, {'mean', 'median', 'min', 'max'}, default='mean'
        The function to aggregate prediction from individual forecasters.
    weights : list of floats, default=None
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
    >>> forecaster.fit(y=y, fh=[1, 2, 3])
    EnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"],
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        "tests:core": True,
    }

    def __init__(self, forecasters, n_jobs=None, aggfunc="mean", weights=None):
        self.aggfunc = aggfunc
        self.weights = weights
        fc = self._parse_fc_multiplicities(forecasters)
        super().__init__(forecasters=forecasters, n_jobs=n_jobs, fc_alt=fc)
        self._anytagis_then_set("requires-fh-in-fit", True, False, self._forecasters)

    def _parse_fc_multiplicities(self, forecasters):
        fc = []
        for forecaster in forecasters:
            if len(forecaster) <= 2:
                fc.append(forecaster)
            elif len(forecaster) == 3:
                name, estimator, num_replicates = forecaster
                fc.extend([(name, estimator)] * num_replicates)
            else:
                raise ValueError("Error in EnsembleForecaster construction.")
        return fc

    def _fit(self, y, X, fh):
        """Fit to training data."""
        self._fit_forecasters(None, y, X, fh)
        return self

    def _predict(self, fh, X):
        """Return the predicted reduction."""
        names = [f[0] for f in self._forecasters]
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1, keys=names)
        y_pred = (
            y_pred.T.groupby(level=1)
            .agg(
                lambda y, aggfunc, weights: _aggregate(y.T, aggfunc, weights),
                self.aggfunc,
                self.weights,
            )
            .T
        )
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.forecasting.compose._reduce import DirectReductionForecaster
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params0 = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}
        FORECASTER_M = DirectReductionForecaster.create_test_instance()
        params1 = {"forecasters": [("f1", FORECASTER_M), ("f2", FORECASTER_M)]}
        params2 = {"forecasters": [("f", FORECASTER_M, 2)]}
        return [params0, params1, params2]


def _aggregate(y, aggfunc, weights):
    if weights is None:
        aggfunc = _check_aggfunc(aggfunc, weighted=False)
        y_agg = aggfunc(y, axis=1)
    else:
        aggfunc = _check_aggfunc(aggfunc, weighted=True)
        y_agg = aggfunc(y, axis=1, weights=np.array(weights))
    return pd.Series(y_agg, index=y.index)


def _check_aggfunc(aggfunc, weighted=False):
    _weighted = "weighted" if weighted else "unweighted"
    if aggfunc not in VALID_AGG_FUNCS.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)
    return VALID_AGG_FUNCS[aggfunc][_weighted]