#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from copy import deepcopy
from inspect import getfullargspec, isfunction, signature

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from sktime.datatypes import VectorizedDF, check_is_scitype, convert_to
from sktime.performance_metrics.base import BaseMetric
from sktime.performance_metrics.forecasting._coerce import (
    _coerce_to_1d_numpy,
    _coerce_to_df,
    _coerce_to_scalar,
    _coerce_to_series,
)
from sktime.performance_metrics.forecasting.sample_weight._types import (
    check_sample_weight_generator,
)
from sktime.utils.warnings import warn

__author__ = ["mloning", "tch", "RNKuhns", "fkiraly", "markussagen"]


def _is_average(multilevel_or_multioutput):
    """Check if multilevel is one of the inputs that lead to averaging.

    True if ``multilevel_or_multioutput`` is one of the strings ``"uniform_average"``,
    ``"uniform_average_time"``.

    False if ``multilevel_or_multioutput`` is the string ``"raw_values"``

    True otherwise
    """
    if isinstance(multilevel_or_multioutput, str):
        if multilevel_or_multioutput in ["uniform_average", "uniform_average_time"]:
            return True
        if multilevel_or_multioutput in ["raw_values"]:
            return False
    else:
        return True


class BaseForecastingErrorMetric(BaseMetric):
    """Base class for defining forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values.

    ``multioutput`` and ``multilevel`` parameters can be used to control averaging
    across variables (``multioutput``) and (non-temporal) hierarchy levels
    (``multilevel``).

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).

        * If 'uniform_average' (default), errors are mean-averaged across levels.
        * If 'uniform_average_time', metric is applied to all data,
          ignoring level index.
        * If 'raw_values', does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Determines averaging over time points in direct call to metric object.

        * If False, direct call to the metric object averages over time points,
          equivalent to a call of the``evaluate`` method.
        * If True, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.
    """

    _tags = {
        "object_type": ["metric_forecasting", "metric"],
        "scitype:y_pred": "pred",  # point forecasts
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
        "lower_is_better": True,
        # "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        "inner_implements_multilevel": False,
        "reserved_params": ["multioutput", "multilevel", "by_index"],
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        self.multioutput = multioutput
        self.multilevel = multilevel
        self.by_index = by_index

        if not hasattr(self, "name") or self.name is None:
            self.name = type(self).__name__

        super().__init__()

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) target values.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series, vanilla forecasting.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series, global/panel forecasting.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection, for
              hierarchical forecasting. ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        y_pred : time series in ``sktime`` compatible data container format
            Predicted values to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        y_pred_benchmark : optional, time series in ``sktime`` compatible data container format
            Benchmark predictions to compare ``y_pred`` to, used for relative metrics.
            Required only if metric requires benchmark predictions,
            as indicated by tag ``requires-y-pred-benchmark``.
            Otherwise, can be passed to ensure interface consistency, but is ignored.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        y_train : optional, time series in ``sktime`` compatible data container format
            Training data used to normalize the error metric.
            Required only if metric requires training data,
            as indicated by tag ``requires-y-train``.
            Otherwise, can be passed to ensure interface consistency, but is ignored.
            Must be of same format as ``y_true``, same columns if indexed,
            but not necessarily same indices.

        sample_weight : optional, 1D array-like, or callable, default=None
            Sample weights for each time point.

            * If ``None``, the time indices are considered equally weighted.
            * If an array, must be 1D.
              If ``y_true`` and ``y_pred``are a single time series,
              ``sample_weight`` must be of the same length as ``y_true``.
              If the time series are panel or hierarchical, the length of all
              individual time
              series must be the same, and equal to the length of ``sample_weight``,
              for all instances of time series passed.
            * If a callable, it must follow ``SampleWeightGenerator`` interface,
              or have one of the following signatures:
              ``y_true: pd.DataFrame -> 1D array-like``,
              or ``y_true: pd.DataFrame x y_pred: pd.DataFrame -> 1D array-like``.

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            Weighted by ``sample_weight`` if provided.

            * float if ``multioutput="uniform_average" or array-like,
              and ``multilevel="uniform_average"`` or "uniform_average_time"``.
              Value is metric averaged over variables and levels (see class docstring)
            * ``np.ndarray`` of shape ``(y_true.columns,)``
              if `multioutput="raw_values"``
              and ``multilevel="uniform_average"`` or ``"uniform_average_time"``.
              i-th entry is the, metric calculated for i-th variable
            * ``pd.DataFrame`` if ``multilevel="raw_values"``.
              of shape ``(n_levels, )``, if ``multioutput="uniform_average"``;
              of shape ``(n_levels, y_true.columns)`` if ``multioutput="raw_values"``.
              metric is applied per level, row averaging (yes/no) as in ``multioutput``.
        """  # noqa: E501
        if self.by_index:
            return self.evaluate_by_index(y_true, y_pred, **kwargs)
        return self.evaluate(y_true, y_pred, **kwargs)

    def _apply_sample_weight_to_kwargs(self, y_true, y_pred, **kwargs):
        """Apply sample weight to kwargs.

        Sample weight is updated to kwargs if it is a callable and follows the
        SampleWeightGenerator interface.
        """
        sample_weight = kwargs.get("sample_weight", None)
        if callable(sample_weight) and check_sample_weight_generator(sample_weight):
            kwargs["sample_weight"] = sample_weight(y_true, y_pred, **kwargs)

        return kwargs

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) target values.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series, vanilla forecasting.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series, global/panel forecasting.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection, for
              hierarchical forecasting. ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        y_pred : time series in ``sktime`` compatible data container format
            Predicted values to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        y_pred_benchmark : optional, time series in ``sktime`` compatible data container format
            Benchmark predictions to compare ``y_pred`` to, used for relative metrics.
            Required only if metric requires benchmark predictions,
            as indicated by tag ``requires-y-pred-benchmark``.
            Otherwise, can be passed to ensure interface consistency, but is ignored.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        y_train : optional, time series in ``sktime`` compatible data container format
            Training data used to normalize the error metric.
            Required only if metric requires training data,
            as indicated by tag ``requires-y-train``.
            Otherwise, can be passed to ensure interface consistency, but is ignored.
            Must be of same format as ``y_true``, same columns if indexed,
            but not necessarily same indices.

        sample_weight : optional, 1D array-like, or callable, default=None
            Sample weights or callable for each time point.

            * If ``None``, the time indices are considered equally weighted.
            * If an array, must be 1D.
              If ``y_true`` and ``y_pred``are a single time series,
              ``sample_weight`` must be of the same length as ``y_true``.
              If the time series are panel or hierarchical, the length of all
              individual time
              series must be the same, and equal to the length of ``sample_weight``,
              for all instances of time series passed.
            * If a callable, it must follow ``SampleWeightGenerator`` interface,
              or have one of the following signatures:
              ``y_true: pd.DataFrame -> 1D array-like``,
              or ``y_true: pd.DataFrame x y_pred: pd.DataFrame -> 1D array-like``.

        Returns
        -------
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            Weighted by ``sample_weight`` if provided.

            * float if ``multioutput="uniform_average" or array-like,
              and ``multilevel="uniform_average"`` or "uniform_average_time"``.
              Value is metric averaged over variables and levels (see class docstring)
            * ``np.ndarray`` of shape ``(y_true.columns,)``
              if `multioutput="raw_values"``
              and ``multilevel="uniform_average"`` or ``"uniform_average_time"``.
              i-th entry is the, metric calculated for i-th variable
            * ``pd.DataFrame`` if ``multilevel="raw_values"``.
              of shape ``(n_levels, )``, if ``multioutput="uniform_average"``;
              of shape ``(n_levels, y_true.columns)`` if ``multioutput="raw_values"``.
              metric is applied per level, row averaging (yes/no) as in ``multioutput``.
        """  # noqa: E501
        multioutput = self.multioutput
        multilevel = self.multilevel

        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
        )

        kwargs = self._apply_sample_weight_to_kwargs(
            y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
        )

        requires_vectorization = isinstance(y_true_inner, VectorizedDF)
        if not requires_vectorization:
            # pass to inner function
            out_df = self._evaluate(y_true=y_true_inner, y_pred=y_pred_inner, **kwargs)
        else:
            out_df = self._evaluate_vectorized(
                y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
            )

        if _is_average(multilevel) and not _is_average(multioutput):
            out_df = _coerce_to_1d_numpy(out_df)
        if _is_average(multilevel) and _is_average(multioutput):
            out_df = _coerce_to_scalar(out_df)
        if not _is_average(multilevel):
            out_df = _coerce_to_df(out_df)

        return out_df

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        By default this uses evaluate_by_index, taking arithmetic mean over time points.

        Parameters
        ----------
        y_true : time series in sktime compatible data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Panel scitype: pd.DataFrame with 2-level row MultiIndex,
                3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                value is metric averaged over variables (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                i-th entry is metric calculated for i-th variable
        """
        # multioutput = self.multioutput
        # multilevel = self.multilevel
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, **kwargs)
            return index_df.mean(axis=0)
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _evaluate_vectorized(self, y_true, y_pred, **kwargs):
        """Vectorized version of _evaluate.

        Runs _evaluate for all instances in y_true, y_pred,
        and returns results in a hierarchical pandas.DataFrame.

        Parameters
        ----------
        y_true : VectorizedDF
        y_pred : VectorizedDF
        non-time-like instances of y_true, y_pred must be identical
        """
        backend = dict()
        backend["backend"] = self.get_config()["backend:parallel"]
        backend["backend_params"] = self.get_config()["backend:parallel:params"]

        eval_result = y_true.vectorize_est(
            estimator=self.clone(),
            method="_evaluate",
            varname_of_self="y_true",
            args={**kwargs, "y_pred": y_pred},
            colname_default=self.name,
            **backend,
        )

        if isinstance(self.multioutput, str) and self.multioutput == "raw_values":
            eval_result = pd.DataFrame(
                eval_result.iloc[:, 0].to_list(),
                index=eval_result.index,
                columns=y_true.X.columns,
            )

        if self.multilevel == "uniform_average":
            eval_result = eval_result.mean(axis=0)

        return eval_result

    def _evaluate_by_index_vectorized(self, y_true, y_pred, **kwargs):
        """Vectorized version of _evaluate_by_index.

        Runs _evaluate for all instances in y_true, y_pred,
        and returns results in a hierarchical pandas.DataFrame.

        Parameters
        ----------
        y_true : VectorizedDF
        y_pred : VectorizedDF
        non-time-like instances of y_true, y_pred must be identical
        """
        backend = dict()
        backend["backend"] = self.get_config()["backend:parallel"]
        backend["backend_params"] = self.get_config()["backend:parallel:params"]

        eval_result = y_true.vectorize_est(
            estimator=self.clone().set_params(**{"multilevel": "uniform_average"}),
            method="_evaluate_by_index",
            varname_of_self="y_true",
            args={**kwargs, "y_pred": y_pred},
            colname_default=self.name,
            return_type="list",
            **backend,
        )

        eval_result = y_true.reconstruct(eval_result)
        return eval_result

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) target values.

            Individual data formats in ``sktime`` are so-called :term:`mtype`
            specifications, each mtype implements an abstract :term:`scitype`.

            * ``Series`` scitype = individual time series, vanilla forecasting.
              ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D)

            * ``Panel`` scitype = collection of time series, global/panel forecasting.
              ``pd.DataFrame`` with 2-level row ``MultiIndex`` ``(instance, time)``,
              ``3D np.ndarray`` ``(instance, variable, time)``,
              ``list`` of ``Series`` typed ``pd.DataFrame``

            * ``Hierarchical`` scitype = hierarchical collection, for
              hierarchical forecasting. ``pd.DataFrame`` with 3 or more level row
              ``MultiIndex`` ``(hierarchy_1, ..., hierarchy_n, time)``

            For further details on data format, see glossary on :term:`mtype`.
            For usage, see forecasting tutorial ``examples/01_forecasting.ipynb``

        y_pred : time series in ``sktime`` compatible data container format
            Predicted values to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        y_pred_benchmark : optional, time series in ``sktime`` compatible data container format
            Benchmark predictions to compare ``y_pred`` to, used for relative metrics.
            Required only if metric requires benchmark predictions,
            as indicated by tag ``requires-y-pred-benchmark``.
            Otherwise, can be passed to ensure interface consistency, but is ignored.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        y_train : optional, time series in ``sktime`` compatible data container format
            Training data used to normalize the error metric.
            Required only if metric requires training data,
            as indicated by tag ``requires-y-train``.
            Otherwise, can be passed to ensure interface consistency, but is ignored.
            Must be of same format as ``y_true``, same columns if indexed,
            but not necessarily same indices.

        sample_weight : optional, 1D array-like, or callable, default=None
            Sample weights or callable for each time point.

            * If ``None``, the time indices are considered equally weighted.
            * If an array, must be 1D.
              If ``y_true`` and ``y_pred``are a single time series,
              ``sample_weight`` must be of the same length as ``y_true``.
              If the time series are panel or hierarchical, the length of all
              individual time
              series must be the same, and equal to the length of ``sample_weight``,
              for all instances of time series passed.
            * If a callable, it must follow ``SampleWeightGenerator`` interface,
              or have one of the following signatures:
              ``y_true: pd.DataFrame -> 1D array-like``,
              or ``y_true: pd.DataFrame x y_pred: pd.DataFrame -> 1D array-like``.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            Weighted by ``sample_weight`` if provided.

            * ``pd.Series`` if ``multioutput="uniform_average"`` or array-like.
              index is equal to index of ``y_true``;
              entry at index i is metric at time i, averaged over variables
            * ``pd.DataFrame`` if ``multioutput="raw_values"``.
              index and columns equal to those of ``y_true``;
              i,j-th entry is metric at time i, at variable j
        """  # noqa: E501
        multioutput = self.multioutput
        multilevel = self.multilevel

        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
        )

        kwargs = self._apply_sample_weight_to_kwargs(
            y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
        )

        requires_vectorization = isinstance(y_true_inner, VectorizedDF)
        if not requires_vectorization:
            # pass to inner function
            out_df = self._evaluate_by_index(
                y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
            )
        else:
            out_df = self._evaluate_by_index_vectorized(
                y_true=y_true_inner, y_pred=y_pred_inner, **kwargs
            )

            if multilevel in ["uniform_average", "uniform_average_time"]:
                out_df = out_df.groupby(level=-1).mean()

        if isinstance(multioutput, str) and multioutput == "raw_values":
            out_df = _coerce_to_df(out_df)
        else:
            out_df = _coerce_to_series(out_df)
        return out_df

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        By default this uses _evaluate to find jackknifed pseudosamples.
        This yields estimates for the metric at each of the time points.
        Caution: this is only sensible for differentiable statistics,
        i.e., not for medians, quantiles or median/quantile based statistics.

        Parameters
        ----------
        y_true : time series in sktime compatible pandas based data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series scitype: pd.DataFrame
            Panel scitype: pd.DataFrame with 2-level row MultiIndex
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
        y_pred :time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput
        n = y_true.shape[0]
        if isinstance(multioutput, str) and multioutput == "raw_values":
            out_series = pd.DataFrame(
                index=y_true.index, columns=y_true.columns, dtype="float64"
            )
        else:
            out_series = pd.Series(index=y_true.index, dtype="float64")
        try:
            x_bar = self.evaluate(y_true, y_pred, **kwargs)
            for i in range(n):
                idx = y_true.index[i]
                kwargs_i = kwargs.copy()
                if "y_pred_benchmark" in kwargs.keys():
                    kwargs_i["y_pred_benchmark"] = kwargs["y_pred_benchmark"].drop(idx)
                pseudovalue = n * x_bar - (n - 1) * self.evaluate(
                    y_true.drop(idx),
                    y_pred.drop(idx),
                    **kwargs_i,
                )
                out_series.loc[idx] = pseudovalue
            return out_series
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _check_consistent_input(self, y_true, y_pred, multioutput, multilevel):
        y_true_orig = y_true
        y_pred_orig = y_pred

        # unwrap y_true, y_pred, if wrapped in VectorizedDF
        if isinstance(y_true, VectorizedDF):
            y_true = y_true.X
        if isinstance(y_pred, VectorizedDF):
            y_pred = y_pred.X

        # check row and column indices if y_true vs y_pred
        same_rows = y_true.index.equals(y_pred.index)
        same_row_num = len(y_true.index) == len(y_pred.index)
        same_cols = y_true.columns.equals(y_pred.columns)
        same_col_num = len(y_true.columns) == len(y_pred.columns)

        if not same_row_num:
            raise ValueError("y_pred and y_true do not have the same number of rows.")
        if not same_col_num:
            raise ValueError(
                "y_pred and y_true do not have the same number of columns."
            )

        if not same_rows:
            warn(
                "y_pred and y_true do not have the same row index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred.",
                obj=self,
            )
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig = deepcopy(y_pred_orig)
                y_pred_orig.X.index = y_true.index
            else:
                y_pred_orig = y_pred_orig.copy()
                y_pred_orig.index = y_true.index
        if not same_cols:
            warn(
                "y_pred and y_true do not have the same column index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred.",
                obj=self,
            )
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig = deepcopy(y_pred_orig)
                y_pred_orig.X.columns = y_true.columns
            else:
                y_pred_orig = y_pred_orig.copy()
                y_pred_orig.columns = y_true.columns
        # check multioutput arg
        # todo: add this back when variance_weighted is supported
        # ("raw_values", "uniform_average", "variance_weighted")
        allowed_multioutput_str = ("raw_values", "uniform_average")

        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    f"Allowed 'multioutput' values are {allowed_multioutput_str}, "
                    f"but found multioutput={multioutput}"
                )
        else:
            multioutput = check_array(multioutput, ensure_2d=False)
            if len(y_pred.columns) != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), len(y_pred.columns))
                )

        # check multilevel arg
        allowed_multilevel_str = (
            "raw_values",
            "uniform_average",
            "uniform_average_time",
        )

        if not isinstance(multilevel, str):
            raise ValueError(f"multilevel must be a str, but found {type(multilevel)}")
        if multilevel not in allowed_multilevel_str:
            raise ValueError(
                f"Allowed 'multilevel' values are {allowed_multilevel_str}, "
                f"but found multilevel={multilevel}"
            )

        return y_true_orig, y_pred_orig, multioutput, multilevel

    def _check_ys(self, y_true, y_pred, multioutput, multilevel, **kwargs):
        SCITYPES = ["Series", "Panel", "Hierarchical"]
        INNER_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]

        def _coerce_to_df(y, var_name="y"):
            if isinstance(y, VectorizedDF):
                return y.X_multiindex

            valid, msg, metadata = check_is_scitype(
                y, scitype=SCITYPES, return_metadata=[], var_name=var_name
            )
            if not valid:
                raise TypeError(msg)
            y_inner = convert_to(y, to_type=INNER_MTYPES)

            scitype = metadata["scitype"]
            ignore_index = multilevel == "uniform_average_time"
            if scitype in ["Panel", "Hierarchical"] and not ignore_index:
                y_inner = VectorizedDF(y_inner, is_scitype=scitype)
            return y_inner

        y_true = _coerce_to_df(y_true, var_name="y_true")
        y_pred = _coerce_to_df(y_pred, var_name="y_pred")
        if "y_train" in kwargs.keys():
            kwargs["y_train"] = _coerce_to_df(kwargs["y_train"], var_name="y_train")
        if "y_pred_benchmark" in kwargs.keys():
            kwargs["y_pred_benchmark"] = _coerce_to_df(
                kwargs["y_pred_benchmark"], var_name="y_pred_benchmark"
            )

        y_true, y_pred, multioutput, multilevel = self._check_consistent_input(
            y_true, y_pred, multioutput, multilevel
        )

        return y_true, y_pred, multioutput, multilevel, kwargs

    def _set_sample_weight_on_kwargs(self, **kwargs):
        """Get sample weights from kwargs.

        Assumes that either ``sample_weight`` is passed, or not.
        If ``sample_weight`` is passed, it is coerced to 1D numpy array and returned.
        Otherwise, returns None.

        Parameters
        ----------
        kwargs : dict
            Dictionary of keyword arguments passed to the metric.

        Returns
        -------
        sample_weight : 1D np.ndarray or None
            1D numpy array of sample weights, or None if not passed.
        """
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is not None:
            sample_weight = check_array(
                sample_weight, ensure_2d=False, input_name="sample_weight"
            )
        return sample_weight

    def _get_weighted_df(self, df, **kwargs):
        """Get weighted DataFrame.

        For n x m df, and kwargs containing sample_weight of length n,
        returns df * sample_weight.reshape(-1, 1), i.e., weights
        multiplied to each row of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be weighted.

        Returns
        -------
        df : pd.DataFrame
            Weighted DataFrame.
        """
        sample_weight = self._set_sample_weight_on_kwargs(**kwargs)
        if sample_weight is not None:
            df = df.mul(sample_weight, axis=0)
        return df

    def _handle_multioutput(self, df, multioutput):
        """Handle multioutput parameter.

        If multioutput is "raw_values", returns df unchanged.
        If multioutput is "uniform_average", returns df.mean(axis=1) for pd.DataFrame,
        or df.mean() for pd.Series.
        If multioutput is array-like, returns df.dot(multioutput).

        Parameters
        ----------
        df : pd.DataFrame or pd.Series
            DataFrame to be handled, assumed result of metric calculation.
        multioutput : str or array-like
            Multioutput parameter.
        """
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return df

            if multioutput == "uniform_average":
                if isinstance(df, pd.Series):
                    return df.mean()
                else:
                    return df.mean(axis=1)

        # else, we expect multioutput to be array-like
        return df.dot(multioutput)


class BaseForecastingErrorMetricFunc(BaseForecastingErrorMetric):
    """Adapter for numpy metrics."""

    # all descendants should have a func class attribute
    #   of signature func(y_true: np.ndarray, y_pred: np.darray, multioutput: bool)
    #   additional optional args: y_train: np.darray, y_pred_benchmark: np.darray
    #                       further args that are parameters
    #       all np.ndarray should be 2D
    # func should return 1D np.ndarray if multioutput="raw_values", otherwise float

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs."""
        # this dict should contain all parameters
        params = self.get_params()

        # adding kwargs to the metric, should not overwrite params (but does if clashes)
        params.update(kwargs)

        # aliasing of sample_weight and horizon_weight keys
        # this is for downwards compatibility with earlier sktime versions
        if "sample_weight" in params.keys() and "horizon_weight" not in params.keys():
            params["horizon_weight"] = params["sample_weight"]
        if "horizon_weight" in params.keys() and "sample_weight" not in params.keys():
            params["sample_weight"] = params["horizon_weight"]

        # calls class variable func, if available, or dynamic (object) variable
        # we need to call type since we store func as a class attribute
        if hasattr(type(self), "func") and isfunction(type(self).func):
            func = type(self).func
        else:
            func = self.func

        return self._evaluate_func(func=func, y_true=y_true, y_pred=y_pred, **params)

    def _evaluate_func(self, func, y_true, y_pred, **params):
        """Call func with kwargs subset to func parameters."""
        # import here for now to avoid interaction with getmembers in tests
        # todo: clean up ancient getmembers in test_metrics_classes
        from functools import partial

        # if func does not catch kwargs, subset to args of func
        if getfullargspec(func).varkw is None or isinstance(func, partial):
            func_params = signature(func).parameters.keys()
            func_params = set(func_params).difference(["y_true", "y_pred"])
            func_params = func_params.intersection(params.keys())
            params = {key: params[key] for key in func_params}

        # deal with sklearn specific parameter constraints
        # as these are a decorator, they obfuscate python native inspection
        # via signature, so have to be dealt with separately
        if hasattr(func, "_skl_parameter_constraints"):
            constr = func._skl_parameter_constraints
            if isinstance(constr, dict):
                constr_params = set(constr.keys()).intersection(params.keys())
                params = {key: params[key] for key in constr_params}

        res = func(y_true=y_true, y_pred=y_pred, **params)
        return res


class _DynamicForecastingErrorMetric(BaseForecastingErrorMetricFunc):
    """Class for defining forecasting error metrics from a function dynamically."""

    def __init__(
        self,
        func,
        name=None,
        multioutput="uniform_average",
        multilevel="uniform_average",
        lower_is_better=True,
        by_index=False,
    ):
        self.multioutput = multioutput
        self.multilevel = multilevel
        self.func = func
        self.name = name
        self.lower_is_better = lower_is_better
        super().__init__(
            multioutput=multioutput, multilevel=multilevel, by_index=by_index
        )

        self.set_tags(**{"lower_is_better": lower_is_better})

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs."""
        # this dict should contain all parameters
        params = kwargs
        params.update({"multioutput": self.multioutput, "multilevel": self.multilevel})

        func = self.func

        score = self._evaluate_func(func=func, y_true=y_true, y_pred=y_pred, **params)

        if _is_average(self.multioutput) and not isinstance(score, float):
            if isinstance(self.multioutput, np.ndarray):
                score = np.dot(score, self.multioutput)
            elif self.multioutput == "uniform_average":
                score = np.mean(score)

        return score

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """

        def custom_mape(y_true, y_pred) -> float:
            eps = np.finfo(np.float64).eps

            result = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps))

            return float(result)

        params1 = {"func": custom_mape, "name": "custom_mape", "lower_is_better": False}

        def custom_mae(y_true, y_pred) -> float:
            result = np.mean(np.abs(y_true - y_pred))

            return float(result)

        params2 = {"func": custom_mae, "name": "custom_mae", "lower_is_better": True}

        return [params1, params2]


class _ScaledMetricTags:
    """Tags for metrics that are scaled on training data y_train."""

    _tags = {
        "requires-y-train": True,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
    }


def make_forecasting_scorer(
    func,
    name=None,
    greater_is_better=False,
    multioutput="uniform_average",
    multilevel="uniform_average",
):
    """Create a metric class from a metric function.

    Parameters
    ----------
    func : callable
        Callable to convert to a forecasting scorer class.
        Score function (or loss function) with signature ``func(y, y_pred, **kwargs)``.

    name : str, default=None
        Name to use for the forecasting scorer loss class.

    greater_is_better : bool, default=False
        If True then maximizing the metric is better.
        If False then minimizing the metric is better.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.

        * If array-like, values used as weights to average the errors.
        * If ``'raw_values'``,
          returns a full set of errors in case of multioutput input.
        * If ``'uniform_average'``,
          errors of all outputs are averaged with uniform weight.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    Returns
    -------
    scorer:
        Metric class that can be used as forecasting scorer.
    """
    lower_is_better = not greater_is_better
    return _DynamicForecastingErrorMetric(
        func,
        name=name,
        multioutput=multioutput,
        multilevel=multilevel,
        lower_is_better=lower_is_better,
    )
