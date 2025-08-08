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

        - feature-importance:
            use the ``feature_importances_`` or ``coef_`` from
            given ``regressor`` as optimal weights.
        - inverse-variance:
            use the inverse variance of the forecasting error
            (based on the internal train-test-split) to compute optimal
            weights, a given ``regressor`` will be omitted.

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
    >>> forecaster.fit(y=y, fh=[1,2,3])
    AutoEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "scitype:y": "univariate",
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
        forecasters = [x[1] for x in self.forecasters_]

        # get training data for meta-model
        if X is not None:
            y_train, y_test, X_train, X_test = temporal_train_test_split(
                y, X, test_size=self.test_size
            )
        else:
            y_train, y_test = temporal_train_test_split(y, test_size=self.test_size)
            X_train, X_test = None, None

        # fit ensemble models
        fh_test = ForecastingHorizon(y_test.index, is_relative=False)
        self._fit_forecasters(forecasters, y_train, X_train, fh_test)

        if self.method == "feature-importance":
            self.regressor_ = check_regressor(
                regressor=self.regressor, random_state=self.random_state
            )
            X_meta = pd.concat(self._predict_forecasters(fh_test, X_test), axis=1)
            X_meta.columns = pd.RangeIndex(len(X_meta.columns))

            # fit meta-model (regressor) on predictions of ensemble models
            # with y_test as endog/target
            self.regressor_.fit(X=X_meta, y=y_test)

            # check if regressor is a sklearn.Pipeline
            if isinstance(self.regressor_, Pipeline):
                # extract regressor from pipeline to access its attributes
                self.weights_ = _get_weights(self.regressor_.steps[-1][1])
            else:
                self.weights_ = _get_weights(self.regressor_)

        elif self.method == "inverse-variance":
            # get in-sample forecasts
            if self.regressor is not None:
                Warning(f"regressor will not be used because ${self.method} is set.")
            inv_var = np.array(
                [
                    1 / np.var(y_test - y_pred_test)
                    for y_pred_test in self._predict_forecasters(fh_test, X_test)
                ]
            )
            # standardize the inverse variance
            self.weights_ = list(inv_var / np.sum(inv_var))
        else:
            raise NotImplementedError(
                f"Given method {self.method} does not exist, "
                f"please provide valid method parameter."
            )

        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _predict(self, fh, X):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        y_pred_df = pd.concat(self._predict_forecasters(fh, X), axis=1)
        # apply weights
        y_pred = y_pred_df.apply(lambda x: np.average(x, weights=self.weights_), axis=1)
        y_pred.name = self._y.name
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
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
    >>> forecaster.fit(y=y, fh=[1,2,3])
    EnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, forecasters, n_jobs=None, aggfunc="mean", weights=None):
        self.aggfunc = aggfunc
        self.weights = weights

        fc = self._parse_fc_multiplicities(forecasters)

        super().__init__(forecasters=forecasters, n_jobs=n_jobs, fc_alt=fc)

        # the ensemble requires fh in fit
        # iff any of the component forecasters require fh in fit
        self._anytagis_then_set("requires-fh-in-fit", True, False, self._forecasters)

    def _parse_fc_multiplicities(self, forecasters):
        """Parse forecasters with multiplicities.

        Turns tuples (name, estimator, count) into list of (name, estimator) tuples.
        """
        fc = []
        for forecaster in forecasters:
            if len(forecaster) <= 2:
                # Handle the (str, est) tuple
                fc.append(forecaster)
            elif len(forecaster) == 3:
                # Handle the (str, est, num_replicates) tuple
                name, estimator, num_replicates = forecaster
                fc.extend([(name, estimator)] * num_replicates)
            else:
                msg = (
                    "Error in EnsembleForecaster construction: "
                    "forecasters argument must be as list of "
                    "estimator, (str, estimator) or (str, estimator, count) tuples."
                )
                raise ValueError(msg)
        return fc

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.DataFrame - Series, Panel, or Hierarchical mtype format.
            Target time series to which to fit the forecaster.
        fh : ForecastingHorizon, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None, must be of same mtype as y
            Exogenous data to which to fit the forecaster.

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_forecasters(None, y, X, fh)
        return self

    def _predict(self, fh, X):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : ForecastingHorizon, optional, default=None
        X : pd.DataFrame, optional, default=None, must be of same mtype as y
            Exogenous data to which to fit the forecaster.

        Returns
        -------
        y_pred : pd.DataFrame - Series, Panel, or Hierarchical mtype format,
            will be of same mtype as y in _fit
            Ensembled predictions
        """
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
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.compose._reduce import DirectReductionForecaster
        from sktime.forecasting.naive import NaiveForecaster

        # univariate case
        FORECASTER = NaiveForecaster()
        params0 = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}

        # test multivariate case, i.e., ensembling multiple variables at same time
        FORECASTER = DirectReductionForecaster.create_test_instance()
        params1 = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}

        # test with multiplicities
        params2 = {"forecasters": [("f", FORECASTER, 2)]}

        return [params0, params1, params2]


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
    if aggfunc not in VALID_AGG_FUNCS.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)
    return VALID_AGG_FUNCS[aggfunc][_weighted]
