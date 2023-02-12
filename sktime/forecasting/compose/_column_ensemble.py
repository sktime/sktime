#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements forecaster for applying different univariates by column."""

__author__ = ["GuzalBulatova", "mloning", "fkiraly"]
__all__ = ["ColumnEnsembleForecaster"]

import numpy as np
import pandas as pd

from sktime.base._meta import flatten
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster

# mtypes that are native pandas
# ColumnEnsembleForecaster uses these internally, since we need (pandas) columns
PANDAS_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


class ColumnEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Forecast each series with separate forecaster.

    Applies different forecasters by columns.

    `ColumnEnsembleForecaster` is passed forecaster/index pairs, exact syntax below.
    Index can be single pandas index element, pd.Index, int, str, or list thereof.
    If iterable (pd.Index, list), refers to multiple columns.

    Behaviour in `fit`, `predict`, `update`:
    For index pairs f_i, ix_i passed, applies forecaster f_i to column(s) ix_i.
    `predict` results are concatenated to one container with same columns as in `fit`.

    Parameters
    ----------
    forecasters : sktime forecaster, or list of tuples (str, estimator, int or pd.index)
        if tuples, with name = str, estimator is forecaster, index as int or index
        if last element is index, it must be int, str, or pd.Index coercable
        if last element is int x, and is not in columns, is interpreted as x-th column
        all columns must be present in an index

        If forecaster, clones of forecaster are applied to all columns.
        If list of tuples, forecaster in tuple is applied to column with int/str index

    Examples
    --------
    >>> from sktime.forecasting.compose import ColumnEnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_longley

    Using integers (column iloc references) for indexing:
    >>> y = load_longley()[1][["GNP", "UNEMP"]]
    >>> forecasters = [
    ...     ("trend", PolynomialTrendForecaster(), 0),
    ...     ("naive", NaiveForecaster(), 1),
    ... ]
    >>> forecaster = ColumnEnsembleForecaster(forecasters=forecasters)
    >>> forecaster.fit(y, fh=[1, 2, 3])
    ColumnEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()

    Using strings for indexing:
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> fc = ColumnEnsembleForecaster(
    ...     [("foo", NaiveForecaster(), "a"), ("bar", NaiveForecaster(), "b")]
    ... )
    >>> fc.fit(df, fh=[1, 42])
    ColumnEnsembleForecaster(...)
    >>> y_pred = fc.predict()

    Applying one forecaster to multiple columns, multivariate:
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    >>> fc = ColumnEnsembleForecaster(
    ...    [("ab", NaiveForecaster(), ["a", 1]), ("c", NaiveForecaster(), 2)]
    ... )
    >>> fc.fit(df, fh=[1, 42])
    ColumnEnsembleForecaster(...)
    >>> y_pred = fc.predict()
    """

    _tags = {
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "y_inner_mtype": PANDAS_MTYPES,
        "X_inner_mtype": PANDAS_MTYPES,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_forecasters"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "forecasters_"

    def __init__(self, forecasters):
        self.forecasters = forecasters
        super(ColumnEnsembleForecaster, self).__init__(forecasters=forecasters)

        # set requires-fh-in-fit depending on forecasters
        if isinstance(forecasters, BaseForecaster):
            tags_to_clone = [
                "requires-fh-in-fit",
                "capability:pred_int",
                "ignores-exogeneous-X",
                "handles-missing-data",
            ]
            self.clone_tags(forecasters, tags_to_clone)
        else:
            l_forecasters = [(x[0], x[1]) for x in forecasters]
            self._anytagis_then_set("requires-fh-in-fit", True, False, l_forecasters)
            self._anytagis_then_set("capability:pred_int", False, True, l_forecasters)
            self._anytagis_then_set("ignores-exogeneous-X", False, True, l_forecasters)
            self._anytagis_then_set("handles-missing-data", False, True, l_forecasters)

    @property
    def _forecasters(self):
        """Make internal list of forecasters.

        The list only contains the name and forecasters, dropping
        the columns. This is for the implementation of get_params
        via _HeterogenousMetaEstimator._get_params which expects
        lists of tuples of len 2.
        """
        forecasters = self.forecasters
        if isinstance(forecasters, BaseForecaster):
            return [("forecasters", forecasters)]
        else:
            return [(name, forecaster) for name, forecaster, _ in self.forecasters]

    @_forecasters.setter
    def _forecasters(self, value):
        if len(value) == 1 and isinstance(value, BaseForecaster):
            self.forecasters = value
        elif len(value) == 1 and isinstance(value, list):
            self.forecasters = value[0][1]
        else:
            self.forecasters = [
                (name, forecaster, columns)
                for ((name, forecaster), (_, _, columns)) in zip(
                    value, self.forecasters
                )
            ]

    def _coerce_to_pd_index(self, obj):
        """Coerce obj to pandas Index."""
        # replace ints by column names
        obj = self._get_indices(self._y, obj)

        # deal with numpy int by coercing to python int
        if np.issubdtype(type(obj), np.integer):
            obj = int(obj)

        # coerce to pd.Index
        if isinstance(obj, (int, str)):
            return pd.Index([obj])
        else:
            return pd.Index(obj)

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        forecasters = self._check_forecasters(y)

        self.forecasters_ = []
        self.y_columns = list(y.columns)

        for (name, forecaster, index) in forecasters:
            forecaster_ = forecaster.clone()

            pd_index = self._coerce_to_pd_index(index)

            forecaster_.fit(y.loc[:, pd_index], X, fh)
            self.forecasters_.append((name, forecaster_, index))

        return self

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.DataFrame
        X : pd.DataFrame
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self.
        """
        for _, forecaster, index in self.forecasters_:
            pd_index = self._coerce_to_pd_index(index)
            forecaster.update(y.loc[:, pd_index], X, update_params=update_params)
        return self

    def _by_column(self, methodname, **kwargs):
        """Apply self.methdoname to kwargs by column, then column-concatenate.

        Parameters
        ----------
        methodname : str, one of the methods of self
            assumed to take kwargs and return pd.DataFrame
        col_multiindex : bool, optional, default=False
            if True, will add an additional column multiindex at top, entries = index

        Returns
        -------
        y_pred : pd.DataFrame
            result of [f.methodname(**kwargs) for _, f, _ in self.forecsaters_]
            column-concatenated with keys being the variable names last seen in y
        """
        # get col_multiindex arg from kwargs
        col_multiindex = kwargs.pop("col_multiindex", False)

        y_preds = []
        keys = []
        for _, forecaster, index in self.forecasters_:
            y_preds += [getattr(forecaster, methodname)(**kwargs)]
            keys += [index]

        keys = self._get_indices(self._y, keys)

        if col_multiindex:
            y_pred = pd.concat(y_preds, axis=1, keys=keys)
        else:
            y_pred = pd.concat(y_preds, axis=1)
        return y_pred

    def _predict(self, fh=None, X=None):
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
        return self._by_column("predict", fh=fh, X=X)

    def _predict_quantiles(self, fh=None, X=None, alpha=None):
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
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
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
        out = self._by_column(
            "predict_quantiles", fh=fh, X=X, alpha=alpha, col_multiindex=True
        )
        if len(out.columns.get_level_values(0).unique()) == 1:
            out.columns = out.columns.droplevel(level=0)
        else:
            out.columns = out.columns.droplevel(level=1)
        return out

    def _predict_interval(self, fh=None, X=None, coverage=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        out = self._by_column(
            "predict_interval", fh=fh, X=X, coverage=coverage, col_multiindex=True
        )
        if len(out.columns.get_level_values(0).unique()) == 1:
            out.columns = out.columns.droplevel(level=0)
        else:
            out.columns = out.columns.droplevel(level=1)
        return out

    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on `cov` variable
            If cov=False:
                Column names are exactly those of `y` passed in `fit`/`update`.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh. Entries are variance forecasts, for var in col index.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
        """
        return self._by_column("predict_var", fh=fh, X=X, cov=cov, col_multiindex=True)

    def _get_indices(self, y, idx):
        """Convert integer indices if necessary."""

        def _get_index(y, ix):
            # deal with numpy int by coercing to python int
            if np.issubdtype(type(ix), np.integer):
                ix = int(ix)

            if isinstance(ix, int) and ix not in y.columns and ix < len(y.columns):
                return y.columns[ix]
            else:
                return ix

        if isinstance(idx, (list, tuple)):
            return [self._get_indices(y, ix) for ix in idx]
        else:
            return _get_index(y, idx)

    def _check_forecasters(self, y):

        # if a single estimator is passed, replicate across columns
        if isinstance(self.forecasters, BaseForecaster):
            ycols = [str(col) for col in y.columns]
            colrange = range(len(ycols))
            forecaster_list = [self.forecasters.clone() for _ in colrange]
            return list(zip(ycols, forecaster_list, colrange))

        if (
            self.forecasters is None
            or len(self.forecasters) == 0
            or not isinstance(self.forecasters, list)
        ):
            raise ValueError(
                "Invalid 'forecasters' attribute, 'forecasters' should be a list"
                " of (string, estimator, int) tuples."
            )
        names, forecasters, indices = zip(*self.forecasters)

        # check names, defined by _HeterogenousEnsembleForecaster
        self._check_names(names)

        # coerce column names to indices in columns
        indices = self._get_indices(y, indices)

        for forecaster in forecasters:
            if not isinstance(forecaster, BaseForecaster):
                raise ValueError(
                    f"The estimator {forecaster.__class__.__name__} should be a "
                    f"Forecaster."
                )

        index_flat = flatten(indices)
        index_set = set(index_flat)
        not_in_y_idx = index_set.difference(y.columns)
        y_cols_not_found = set(y.columns).difference(index_set)

        if len(not_in_y_idx) > 0:
            raise ValueError(
                f"Column identifier must be indices in y.columns, or integers within "
                f"the range of the total number of columns, "
                f"but found column identifiers that are neither: {list(not_in_y_idx)}"
            )
        if len(y_cols_not_found) > 0:
            raise ValueError(
                f"All columns of y must be indexed by column identifiers, but "
                f"the following columns of y are not indexed: {list(y_cols_not_found)}"
            )

        if len(index_set) != len(index_flat):
            raise ValueError(
                f"One estimator per column required. Found {len(index_set)} unique"
                f" column names in forecasters arg, required {len(index_flat)}"
            )

        return self.forecasters

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # imports
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import TrendForecaster

        params1 = {"forecasters": NaiveForecaster()}
        params2 = {"forecasters": TrendForecaster()}

        return [params1, params2]
