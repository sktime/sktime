#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""
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
from sktime.performance_metrics.forecasting._functions import (
    geometric_mean_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    geometric_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_asymmetric_error,
    mean_linex_error,
    mean_relative_absolute_error,
    mean_squared_error,
    mean_squared_percentage_error,
    mean_squared_scaled_error,
    median_absolute_error,
    median_absolute_percentage_error,
    median_absolute_scaled_error,
    median_relative_absolute_error,
    median_squared_error,
    median_squared_percentage_error,
    median_squared_scaled_error,
    relative_loss,
)
from sktime.utils.warnings import warn

__author__ = ["mloning", "tch", "RNKuhns", "fkiraly"]
__all__ = [
    "make_forecasting_scorer",
    "MeanAbsoluteScaledError",
    "MedianAbsoluteScaledError",
    "MeanSquaredScaledError",
    "MedianSquaredScaledError",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "MedianAbsoluteError",
    "MedianSquaredError",
    "GeometricMeanAbsoluteError",
    "GeometricMeanSquaredError",
    "MeanAbsolutePercentageError",
    "MedianAbsolutePercentageError",
    "MeanSquaredPercentageError",
    "MedianSquaredPercentageError",
    "MeanRelativeAbsoluteError",
    "MedianRelativeAbsoluteError",
    "GeometricMeanRelativeAbsoluteError",
    "GeometricMeanRelativeSquaredError",
    "MeanAsymmetricError",
    "MeanLinexError",
    "RelativeLoss",
]


def _is_average(multilevel_or_multioutput):
    """Check if multilevel is one of the inputs that lead to averaging.

    True if `multilevel_or_multioutput` is one of the strings `"uniform_average"`,
    `"uniform_average_time"`.

    False if `multilevel_or_multioutput` is the string `"raw_values"`

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

    `multioutput` and `multilevel` parameters can be used to control averaging
    across variables (`multioutput`) and (non-temporal) hierarchy levels (`multilevel`).

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": False,
        "univariate-only": False,
        "lower_is_better": True,
        # "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        "inner_implements_multilevel": False,
        "reserved_params": ["multioutput", "multilevel"],
    }

    def __init__(self, multioutput="uniform_average", multilevel="uniform_average"):
        self.multioutput = multioutput
        self.multilevel = multilevel

        if not hasattr(self, "name") or self.name is None:
            self.name = type(self).__name__

        super().__init__()

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

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
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                and self.multilevel="uniform_average" or "uniform_average_time"
                value is metric averaged over variables and levels (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                and self.multilevel="uniform_average" or "uniform_average_time"
                i-th entry is metric calculated for i-th variable
            pd.DataFrame if self.multilevel=raw.values
                of shape (n_levels, ) if self.multioutput = "uniform_average" or array
                of shape (n_levels, y_true.columns) if self.multioutput="raw_values"
                metric is applied per level, row averaging (yes/no) as in multioutput
        """
        return self.evaluate(y_true, y_pred, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

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
        loss : float, np.ndarray, or pd.DataFrame
            Calculated metric, averaged or by variable.
            float if self.multioutput="uniform_average" or array-like
                and self.multilevel="uniform_average" or "uniform_average_time"
                value is metric averaged over variables and levels (see class docstring)
            np.ndarray of shape (y_true.columns,) if self.multioutput="raw_values"
                and self.multilevel="uniform_average" or "uniform_average_time"
                i-th entry is metric calculated for i-th variable
            pd.DataFrame if self.multilevel=raw.values
                of shape (n_levels, ) if self.multioutput = "uniform_average" or array
                of shape (n_levels, y_true.columns) if self.multioutput="raw_values"
                metric is applied per level, row averaging (yes/no) as in multioutput
        """
        multioutput = self.multioutput
        multilevel = self.multilevel
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
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
        eval_result = y_true.vectorize_est(
            estimator=self.clone(),
            method="_evaluate",
            varname_of_self="y_true",
            args={**kwargs, "y_pred": y_pred},
            colname_default=self.name,
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
        eval_result = y_true.vectorize_est(
            estimator=self.clone().set_params(**{"multilevel": "uniform_average"}),
            method="_evaluate_by_index",
            varname_of_self="y_true",
            args={**kwargs, "y_pred": y_pred},
            colname_default=self.name,
            return_type="list",
        )

        eval_result = y_true.reconstruct(eval_result)
        return eval_result

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

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
        multilevel = self.multilevel
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput, multilevel, kwargs = self._check_ys(
            y_true, y_pred, multioutput, multilevel, **kwargs
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
            y_pred_orig = y_pred_orig.copy()
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig.X.index = y_true.index
            else:
                y_pred_orig.index = y_true.index
        if not same_cols:
            warn(
                "y_pred and y_true do not have the same column index. "
                "This may indicate incorrect objects passed to the metric. "
                "Indices of y_true will be used for y_pred.",
                obj=self,
            )
            y_pred_orig = y_pred_orig.copy()
            if isinstance(y_pred_orig, VectorizedDF):
                y_pred_orig.X.columns = y_true.columns
            else:
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
                y, scitype=SCITYPES, return_metadata=True, var_name=var_name
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

        # calls class variable func, if available, or dynamic (object) variable
        # we need to call type since we store func as a class attribute
        if hasattr(type(self), "func") and isfunction(type(self).func):
            func = type(self).func
        else:
            func = self.func

        # import here for now to avoid interaction with getmembers in tests
        # todo: clean up ancient getmembers in test_metrics_classes
        from functools import partial

        # if func does not catch kwargs, subset to args of func
        if getfullargspec(func).varkw is None or isinstance(func, partial):
            func_params = signature(func).parameters.keys()
            func_params = set(func_params).difference(["y_true", "y_pred"])
            func_params = func_params.intersection(params.keys())
            params = {key: params[key] for key in func_params}

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
    ):
        self.multioutput = multioutput
        self.multilevel = multilevel
        self.func = func
        self.name = name
        self.lower_is_better = lower_is_better

        super().__init__(multioutput=multioutput, multilevel=multilevel)

        self.set_tags(**{"lower_is_better": lower_is_better})

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
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
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

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


class MeanAbsoluteScaledError(_ScaledMetricTags, BaseForecastingErrorMetricFunc):
    """Mean absolute scaled error (MASE).

    MASE output is non-negative floating point. The best value is 0.0.

    Like other scaled performance metrics, this scale-free error metric can be
    used to compare forecast methods on a single series and also to compare
    forecast accuracy between series.

    This metric is well suited to intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of the data
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MedianAbsoluteScaledError
    MeanSquaredScaledError
    MedianSquaredScaledError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
    for intermittent demand", Foresight, Issue 4.

    Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
    "The M4 Competition: 100,000 time series and 61 forecasting methods",
    International Journal of Forecasting, Volume 3.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mase = MeanAbsoluteScaledError()
    >>> mase(y_true, y_pred, y_train=y_train)
    0.18333333333333335
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mase(y_true, y_pred, y_train=y_train)
    0.18181818181818182
    >>> mase = MeanAbsoluteScaledError(multioutput='raw_values')
    >>> mase(y_true, y_pred, y_train=y_train)
    array([0.10526316, 0.28571429])
    >>> mase = MeanAbsoluteScaledError(multioutput=[0.3, 0.7])
    >>> mase(y_true, y_pred, y_train=y_train)
    0.21935483870967742
    """

    func = mean_absolute_scaled_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
    ):
        self.sp = sp
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"sp": 2}
        return [params1, params2]


class MedianAbsoluteScaledError(_ScaledMetricTags, BaseForecastingErrorMetricFunc):
    """Median absolute scaled error (MdASE).

    MdASE output is non-negative floating point. The best value is 0.0.

    Taking the median instead of the mean of the test and train absolute errors
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Like MASE and other scaled performance metrics this scale-free metric can be
    used to compare forecast methods on a single series or between series.

    Also like MASE, this metric is well suited to intermittent-demand series
    because it will not give infinite or undefined values unless the training
    data is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of data.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsoluteScaledError
    MeanSquaredScaledError
    MedianSquaredScaledError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
    for intermittent demand", Foresight, Issue 4.

    Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
    "The M4 Competition: 100,000 time series and 61 forecasting methods",
    International Journal of Forecasting, Volume 3.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianAbsoluteScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mdase = MedianAbsoluteScaledError()
    >>> mdase(y_true, y_pred, y_train=y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdase(y_true, y_pred, y_train=y_train)
    0.18181818181818182
    >>> mdase = MedianAbsoluteScaledError(multioutput='raw_values')
    >>> mdase(y_true, y_pred, y_train=y_train)
    array([0.10526316, 0.28571429])
    >>> mdase = MedianAbsoluteScaledError(multioutput=[0.3, 0.7])
    >>> mdase( y_true, y_pred, y_train=y_train)
    0.21935483870967742
    """

    func = median_absolute_scaled_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
    ):
        self.sp = sp
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"sp": 2}
        return [params1, params2]


class MeanSquaredScaledError(_ScaledMetricTags, BaseForecastingErrorMetricFunc):
    """Mean squared scaled error (MSSE) or root mean squared scaled error (RMSSE).

    If `square_root` is False then calculates MSSE, otherwise calculates RMSSE if
    `square_root` is True. Both MSSE and RMSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared variant of the MASE loss metric.  Like MASE and other
    scaled performance metrics this scale-free metric can be used to compare
    forecast methods on a single series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of data.
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MedianSquaredScaledError

    References
    ----------
    M5 Competition Guidelines.
    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanSquaredScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> rmsse = MeanSquaredScaledError(square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    0.20568833780186058
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> rmsse(y_true, y_pred, y_train=y_train)
    0.15679361328058636
    >>> rmsse = MeanSquaredScaledError(multioutput='raw_values', square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    array([0.11215443, 0.20203051])
    >>> rmsse = MeanSquaredScaledError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmsse(y_true, y_pred, y_train=y_train)
    0.17451891814894502
    """

    func = mean_squared_scaled_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
        square_root=False,
    ):
        self.sp = sp
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"sp": 2, "square_root": True}
        return [params1, params2]


class MedianSquaredScaledError(_ScaledMetricTags, BaseForecastingErrorMetricFunc):
    """Median squared scaled error (MdSSE) or root median squared scaled error (RMdSSE).

    If `square_root` is False then calculates MdSSE, otherwise calculates RMdSSE if
    `square_root` is True. Both MdSSE and RMdSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared variant of the MdASE loss metric. Like MASE and other
    scaled performance metrics this scale-free metric can be used to compare
    forecast methods on a single series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    sp : int, default = 1
        Seasonal periodicity of data.
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MedianSquaredScaledError

    References
    ----------
    M5 Competition Guidelines.
    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianSquaredScaledError
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> rmdsse = MedianSquaredScaledError(square_root=True)
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    0.1472819539849714
    >>> rmdsse = MedianSquaredScaledError(multioutput='raw_values', square_root=True)
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    array([0.08687445, 0.20203051])
    >>> rmdsse = MedianSquaredScaledError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmdsse(y_true, y_pred, y_train=y_train)
    0.16914781383660782
    """

    func = median_squared_scaled_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
        square_root=False,
    ):
        self.sp = sp
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"sp": 2, "square_root": True}
        return [params1, params2]


class MeanAbsoluteError(BaseForecastingErrorMetric):
    r"""Mean absolute error (MAE).

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n` (in :math:`mathbb{R}`),
    at time indices :math:`t_1, \dots, t_n`,
    `evaluate` or call returns the Mean Absolute Error,
    :math:`\frac{1}{n}\sum_{i=1}^n |y_i - \widehat{y}_i|`.
    (the time indices are not used)

    `multioutput` and `multilevel` control averaging across variables and
    hierarchy indices, see below.

    `evaluate_by_index` returns, at a time index :math:`t_i`,
    the absolute error at that time index, :math:`|y_i - \widehat{y}_i|`,
    for all time indices :math:`t_1, \dots, t_n` in the input.

    MAE output is non-negative floating point. The best value is 0.0.

    MAE is on the same scale as the data. Because MAE takes the absolute value
    of the forecast error rather than squaring it, MAE penalizes large errors
    to a lesser degree than MSE or RMSE.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MedianAbsoluteError
    MeanSquaredError
    MedianSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mae = MeanAbsoluteError()
    >>> mae(y_true, y_pred)
    0.55
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mae(y_true, y_pred)
    0.75
    >>> mae = MeanAbsoluteError(multioutput='raw_values')
    >>> mae(y_true, y_pred)
    array([0.5, 1. ])
    >>> mae = MeanAbsoluteError(multioutput=[0.3, 0.7])
    >>> mae(y_true, y_pred)
    0.85
    """

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

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

        raw_values = (y_true - y_pred).abs()

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return raw_values

            if multioutput == "uniform_average":
                return raw_values.mean(axis=1)

        # else, we expect multioutput to be array-like
        return raw_values.dot(multioutput)


class MedianAbsoluteError(BaseForecastingErrorMetricFunc):
    """Median absolute error (MdAE).

    MdAE output is non-negative floating point. The best value is 0.0.

    Like MAE, MdAE is on the same scale as the data. Because MAE takes the
    absolute value of the forecast error rather than squaring it, MAE penalizes
    large errors to a lesser degree than MdSE or RdMSE.

    Taking the median instead of the mean of the absolute errors also makes
    this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsoluteError
    MeanSquaredError
    MedianSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdae = MedianAbsoluteError()
    >>> mdae(y_true, y_pred)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdae(y_true, y_pred)
    0.75
    >>> mdae = MedianAbsoluteError(multioutput='raw_values')
    >>> mdae(y_true, y_pred)
    array([0.5, 1. ])
    >>> mdae = MedianAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdae(y_true, y_pred)
    0.85
    """

    func = median_absolute_error


class MeanSquaredError(BaseForecastingErrorMetricFunc):
    """Mean squared error (MSE) or root mean squared error (RMSE).

    If `square_root` is False then calculates MSE and if `square_root` is True
    then RMSE is calculated.  Both MSE and RMSE are both non-negative floating
    point. The best value is 0.0.

    MSE is measured in squared units of the input data, and RMSE is on the
    same scale as the data. Because MSE and RMSE square the forecast error
    rather than taking the absolute value, they penalize large errors more than
    MAE.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsoluteError
    MedianAbsoluteError
    MedianSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mse = MeanSquaredError()
    >>> mse(y_true, y_pred)
    0.4125
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mse(y_true, y_pred)
    0.7083333333333334
    >>> rmse = MeanSquaredError(square_root=True)
    >>> rmse(y_true, y_pred)
    0.8227486121839513
    >>> rmse = MeanSquaredError(multioutput='raw_values')
    >>> rmse(y_true, y_pred)
    array([0.41666667, 1.        ])
    >>> rmse = MeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rmse(y_true, y_pred)
    array([0.64549722, 1.        ])
    >>> rmse = MeanSquaredError(multioutput=[0.3, 0.7])
    >>> rmse(y_true, y_pred)
    0.825
    >>> rmse = MeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmse(y_true, y_pred)
    0.8936491673103708
    """

    func = mean_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
    ):
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"square_root": True}
        return [params1, params2]


class MedianSquaredError(BaseForecastingErrorMetricFunc):
    """Median squared error (MdSE) or root median squared error (RMdSE).

    If `square_root` is False then calculates MdSE and if `square_root` is True
    then RMdSE is calculated. Both MdSE and RMdSE return non-negative floating
    point. The best value is 0.0.

    Like MSE, MdSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE. Because MdSE and RMdSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than MAE or MdAE.

    Taking the median instead of the mean of the squared errors makes
    this metric more robust to error outliers relative to a meean based metric
    since the median tends to be a more robust measure of central tendency in
    the presence of outliers.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsoluteError
    MedianAbsoluteError
    MeanSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdse = MedianSquaredError()
    >>> mdse(y_true, y_pred)
    0.25
    >>> rmdse = MedianSquaredError(square_root=True)
    >>> rmdse(y_true, y_pred)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdse(y_true, y_pred)
    0.625
    >>> rmdse(y_true, y_pred)
    0.75
    >>> mdse = MedianSquaredError(multioutput='raw_values')
    >>> mdse(y_true, y_pred)
    array([0.25, 1.  ])
    >>> rmdse = MedianSquaredError(multioutput='raw_values', square_root=True)
    >>> rmdse(y_true, y_pred)
    array([0.5, 1. ])
    >>> mdse = MedianSquaredError(multioutput=[0.3, 0.7])
    >>> mdse(y_true, y_pred)
    0.7749999999999999
    >>> rmdse = MedianSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmdse(y_true, y_pred)
    0.85
    """

    func = median_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
    ):
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"square_root": True}
        return [params1, params2]


class GeometricMeanAbsoluteError(BaseForecastingErrorMetricFunc):
    """Geometric mean absolute error (GMAE).

    GMAE output is non-negative floating point. The best value is approximately
    zero, rather than zero.

    Like MAE and MdAE, GMAE is measured in the same units as the input data.
    Because GMAE takes the absolute value of the forecast error rather than
    squaring it, MAE penalizes large errors to a lesser degree than squared error
    variants like MSE, RMSE or GMSE or RGMSE.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_squared_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of `y_true` and `y_pred` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when `y_true` equals `y_pred`)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import GeometricMeanAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmae = GeometricMeanAbsoluteError()
    >>> gmae(y_true, y_pred)
    0.000529527232030127
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmae(y_true, y_pred)
    0.5000024031086919
    >>> gmae = GeometricMeanAbsoluteError(multioutput='raw_values')
    >>> gmae(y_true, y_pred)
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmae = GeometricMeanAbsoluteError(multioutput=[0.3, 0.7])
    >>> gmae(y_true, y_pred)
    0.7000014418652152
    """

    func = geometric_mean_absolute_error


class GeometricMeanSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean squared error (GMSE) or Root geometric mean squared error (RGMSE).

    If `square_root` is False then calculates GMSE and if `square_root` is True
    then RGMSE is calculated. Both GMSE and RGMSE return non-negative floating
    point. The best value is approximately zero, rather than zero.

    Like MSE and MdSE, GMSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE and RdMSE. Because GMSE and RGMSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than GMAE.

    Parameters
    ----------
    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root geometric mean squared error (RGMSE)
        If False, returns geometric mean squared error (GMSE)
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_absolute_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of `y_true` and `y_pred` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when `y_true` equals `y_pred`)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import GeometricMeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    2.80399089461488e-07
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    0.000529527232030127
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    0.5000000000115499
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    0.5000024031086919
    >>> gmse = GeometricMeanSquaredError(multioutput='raw_values')
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    array([2.30997255e-11, 1.00000000e+00])
    >>> rgmse = GeometricMeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rgmse(y_true, y_pred)# doctest: +SKIP
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7])
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    0.7000000000069299
    >>> rgmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    0.7000014418652152
    """

    func = geometric_mean_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
    ):
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"square_root": True}
        return [params1, params2]


class MeanAbsolutePercentageError(BaseForecastingErrorMetricFunc):
    r"""Mean absolute percentage error (MAPE) or symmetric version.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    at time indices :math:`t_1, \dots, t_n`,
    `evaluate` or call returns the Mean Absolute Percentage Error,
    :math:`\frac{1}{n} \sum_{i=1}^n \left|\frac{y_i-\widehat{y}_i}{y_i} \right|`.
    (the time indices are not used)

    if `symmetric` is True then calculates
    symmetric mean absolute percentage error (sMAPE), defined as
    :math:`\frac{2}{n} \sum_{i=1}^n \frac{|y_i - \widehat{y}_i|}
    {|y_i| + |\widehat{y}_i|}`.

    Both MAPE and sMAPE output non-negative floating point which is in fractional units
    rather than percentage. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    MAPE has no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`. While sMAPE is bounded at 2.

    `multioutput` and `multilevel` control averaging across variables and
    hierarchy indices, see below.

    `evaluate_by_index` returns, at a time index :math:`t_i`,
    the absolute percentage error at that time index,
    :math:`\left| \frac{y_i - \widehat{y}_i}{y_i} \right|`,
    or :math:`\frac{2|y_i - \widehat{y}_i|}{|y_i| + |\widehat{y}_i|}`,
    the symmetric version, if `symmetric` is True, for all time indices
    :math:`t_1, \dots, t_n` in the input.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
    multioutput : str or 1D array-like (n_outputs,), default='uniform_average'
        if str, must be one of {'raw_values', 'uniform_average'}
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MedianAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mape = MeanAbsolutePercentageError(symmetric=False)
    >>> mape(y_true, y_pred)
    0.33690476190476193
    >>> smape = MeanAbsolutePercentageError(symmetric=True)
    >>> smape(y_true, y_pred)
    0.5553379953379953
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mape(y_true, y_pred)
    0.5515873015873016
    >>> smape(y_true, y_pred)
    0.6080808080808081
    >>> mape = MeanAbsolutePercentageError(multioutput='raw_values', symmetric=False)
    >>> mape(y_true, y_pred)
    array([0.38095238, 0.72222222])
    >>> smape = MeanAbsolutePercentageError(multioutput='raw_values', symmetric=True)
    >>> smape(y_true, y_pred)
    array([0.71111111, 0.50505051])
    >>> mape = MeanAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mape(y_true, y_pred)
    0.6198412698412699
    >>> smape = MeanAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=True)
    >>> smape(y_true, y_pred)
    0.5668686868686869
    """

    func = mean_absolute_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
    ):
        self.symmetric = symmetric
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"symmetric": True}
        return [params1, params2]


class MedianAbsolutePercentageError(BaseForecastingErrorMetricFunc):
    r"""Median absolute percentage error (MdAPE) or symmetric version.

    For a univariate, non-hierarchical sample of true values :math:`y_1, \dots, y_n`
    and predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    at time indices :math:`t_1, \dots, t_n`,
    `evaluate` or call returns the Median Absolute Percentage Error,
    :math:`median(\left|\frac{y_i - \widehat{y}_i}{y_i} \right|)`.
    (the time indices are not used)

    if `symmetric` is True then calculates
    symmetric Median Absolute Percentage Error (sMdAPE), defined as
    :math:`median(\frac{2|y_i-\widehat{y}_i|}{|y_i|+|\widehat{y}_i|})`.

    Both MdAPE and sMdAPE output non-negative floating point which is in fractional
    units rather than percentage. The best value is 0.0.

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because it takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    MAPE has no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`. While sMAPE is bounded at 2.

    `multioutput` and `multilevel` control averaging across variables and
    hierarchy indices, see below.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
    multioutput : str or 1D array-like (n_outputs,), default='uniform_average'
        if str, must be one of {'raw_values', 'uniform_average'}
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianAbsolutePercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdape = MedianAbsolutePercentageError(symmetric=False)
    >>> mdape(y_true, y_pred)
    0.16666666666666666
    >>> smdape = MedianAbsolutePercentageError(symmetric=True)
    >>> smdape(y_true, y_pred)
    0.18181818181818182
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdape(y_true, y_pred)
    0.5714285714285714
    >>> smdape(y_true, y_pred)
    0.39999999999999997
    >>> mdape = MedianAbsolutePercentageError(multioutput='raw_values', symmetric=False)
    >>> mdape(y_true, y_pred)
    array([0.14285714, 1.        ])
    >>> smdape = MedianAbsolutePercentageError(multioutput='raw_values', symmetric=True)
    >>> smdape(y_true, y_pred)
    array([0.13333333, 0.66666667])
    >>> mdape = MedianAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mdape(y_true, y_pred)
    0.7428571428571428
    >>> smdape = MedianAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=True)
    >>> smdape(y_true, y_pred)
    0.5066666666666666
    """  # noqa: E501

    func = median_absolute_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
    ):
        self.symmetric = symmetric
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"symmetric": True}
        return [params1, params2]


class MeanSquaredPercentageError(BaseForecastingErrorMetricFunc):
    """Mean squared percentage error (MSPE)  or square root version.

    If `square_root` is False then calculates MSPE and if `square_root` is True
    then calculates root mean squared percentage error (RMSPE). If `symmetric`
    is True then calculates sMSPE or sRMSPE. Output is non-negative floating
    point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MeanSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mspe = MeanSquaredPercentageError(symmetric=False)
    >>> mspe(y_true, y_pred)
    0.23776218820861678
    >>> smspe = MeanSquaredPercentageError(square_root=True, symmetric=False)
    >>> smspe(y_true, y_pred)
    0.48760864246710883
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mspe(y_true, y_pred)
    0.5080309901738473
    >>> smspe(y_true, y_pred)
    0.7026794936195895
    >>> mspe = MeanSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mspe(y_true, y_pred)
    array([0.34013605, 0.67592593])
    >>> smspe = MeanSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    array([0.58321184, 0.82214714])
    >>> mspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mspe(y_true, y_pred)
    0.5751889644746787
    >>> smspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    0.7504665536595034
    """

    func = mean_squared_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        square_root=False,
    ):
        self.symmetric = symmetric
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"symmetric": True, "square_root": True}
        return [params1, params2]


class MedianSquaredPercentageError(BaseForecastingErrorMetricFunc):
    """Median squared percentage error (MdSPE)  or square root version.

    If `square_root` is False then calculates MdSPE and if `square_root` is True
    then calculates root median squared percentage error (RMdSPE). If `symmetric`
    is True then calculates sMdSPE or sRMdSPE. Output is non-negative floating
    point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    RMdSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MeanSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MedianSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdspe = MedianSquaredPercentageError(symmetric=False)
    >>> mdspe(y_true, y_pred)
    0.027777777777777776
    >>> smdspe = MedianSquaredPercentageError(square_root=True, symmetric=False)
    >>> smdspe(y_true, y_pred)
    0.16666666666666666
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdspe(y_true, y_pred)
    0.5102040816326531
    >>> smdspe(y_true, y_pred)
    0.5714285714285714
    >>> mdspe = MedianSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mdspe(y_true, y_pred)
    array([0.02040816, 1.        ])
    >>> smdspe = MedianSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smdspe(y_true, y_pred)
    array([0.14285714, 1.        ])
    >>> mdspe = MedianSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mdspe(y_true, y_pred)
    0.7061224489795918
    >>> smdspe = MedianSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smdspe(y_true, y_pred)
    0.7428571428571428
    """

    func = median_squared_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        square_root=False,
    ):
        self.symmetric = symmetric
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"symmetric": True, "square_root": True}
        return [params1, params2]


class MeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Mean relative absolute error (MRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MRAE applies mean absolute error (MAE) to the resulting relative errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae = MeanRelativeAbsoluteError()
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.9511111111111111
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8703703703703702
    >>> mrae = MeanRelativeAbsoluteError(multioutput='raw_values')
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.51851852, 1.22222222])
    >>> mrae = MeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    1.0111111111111108
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = mean_relative_absolute_error


class MedianRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Median relative absolute error (MdRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MdRAE applies medan absolute error (MdAE) to the resulting relative errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae = MedianRelativeAbsoluteError()
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    1.0
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.6944444444444443
    >>> mdrae = MedianRelativeAbsoluteError(multioutput='raw_values')
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.55555556, 0.83333333])
    >>> mdrae = MedianRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.7499999999999999
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = median_relative_absolute_error


class GeometricMeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Geometric mean relative absolute error (GMRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRAE applies geometric mean absolute error (GMAE) to the resulting relative
    errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae = GeometricMeanRelativeAbsoluteError()
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.0007839273064064755
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.5578632807409556
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput='raw_values')
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.97801163e-06, 1.11572158e+00])
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.7810066018326863
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = geometric_mean_relative_absolute_error


class GeometricMeanRelativeSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean relative squared error (GMRSE).

    If `square_root` is False then calculates GMRSE and if `square_root` is True
    then calculates root geometric mean relative squared error (RGMRSE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRSE applies geometric mean squared error (GMSE) to the resulting relative
    errors. RGMRSE applies root geometric mean squared error (RGMSE) to the
    resulting relative errors.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse = GeometricMeanRelativeSquaredError()
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.0008303544925949156
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.622419372049448
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput='raw_values')
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.09227746e-06, 1.24483465e+00])
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput=[0.3, 0.7])
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8713854839582426
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = geometric_mean_relative_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
    ):
        self.square_root = square_root
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"square_root": True}
        return [params1, params2]


class MeanAsymmetricError(BaseForecastingErrorMetricFunc):
    """Calculate mean of asymmetric loss function.

    Output is non-negative floating point. The best value is 0.0.

    Error values that are less than the asymmetric threshold have
    `left_error_function` applied. Error values greater than or equal to
    asymmetric threshold  have `right_error_function` applied.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    Setting `asymmetric_threshold` to zero, `left_error_function` to 'squared'
    and `right_error_function` to 'absolute` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for `left_error_function` set to 'absolute' and `right_error_function`
    set to 'squared`.

    The left_error_penalty and right_error_penalty can be used to add differing
    multiplicative penalties to over-predictions and under-predictions.

    Parameters
    ----------
    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have `left_error_function`
        applied. Error values greater than or equal to asymmetric threshold
        have `right_error_function` applied.
    left_error_function : {'squared', 'absolute'}, default='squared'
        Loss penalty to apply to error values less than the asymmetric threshold.
    right_error_function : {'squared', 'absolute'}, default='absolute'
        Loss penalty to apply to error values greater than or equal to the
        asymmetric threshold.
    left_error_penalty : int or float, default=1.0
        An additional multiplicative penalty to apply to error values less than
        the asymmetric threshold.
    right_error_penalty : int or float, default=1.0
        An additional multiplicative penalty to apply to error values greater
        than the asymmetric threshold.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    mean_linex_error

    Notes
    -----
    Setting `left_error_function` and `right_error_function` to "absolute", but
    choosing different values for `left_error_penalty` and `right_error_penalty`
    results in the "lin-lin" error function discussed in [2]_.

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [2] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
       Thomson, South-Western: Ohio, US.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAsymmetricError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> asymmetric_error = MeanAsymmetricError()
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    0.5
    >>> asymmetric_error = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    0.4625
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> asymmetric_error = MeanAsymmetricError()
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    0.75
    >>> asymmetric_error = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    0.7083333333333334
    >>> asymmetric_error = MeanAsymmetricError(multioutput='raw_values')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    array([0.5, 1. ])
    >>> asymmetric_error = MeanAsymmetricError(multioutput=[0.3, 0.7])
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    0.85
    """

    func = mean_asymmetric_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
        left_error_penalty=1.0,
        right_error_penalty=1.0,
    ):
        self.asymmetric_threshold = asymmetric_threshold
        self.left_error_function = left_error_function
        self.right_error_function = right_error_function
        self.left_error_penalty = left_error_penalty
        self.right_error_penalty = right_error_penalty

        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {
            "asymmetric_threshold": 0.1,
            "left_error_function": "absolute",
            "right_error_function": "squared",
            "left_error_penalty": 2.0,
            "right_error_penalty": 0.5,
        }
        return [params1, params2]


class MeanLinexError(BaseForecastingErrorMetricFunc):
    """Calculate mean linex error.

    Output is non-negative floating point. The best value is 0.0.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    The linex error function accounts for this by penalizing errors on one side
    of a threshold approximately linearly, while penalizing errors on the other
    side approximately exponentially. If `a` > 0 then negative errors
    (over-predictions) are penalized approximately linearly and positive errors
    (under-predictions) are penalized approximately exponentially. If `a` < 0
    the reverse is true.

    Parameters
    ----------
    a : int or float
        Controls whether over- or under- predictions receive an approximately
        linear or exponential penalty. If `a` > 0 then negative errors
        (over-predictions) are penalized approximately linearly and positive errors
        (under-predictions) are penalized approximately exponentially. If `a` < 0
        the reverse is true.
    b : int or float
        Multiplicative penalty to apply to calculated errors.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    See Also
    --------
    mean_asymmetric_error

    Notes
    -----
    Calculated as b * (np.exp(a * error) - a * error - 1), where a != 0 and b > 0
    according to formula in [2]_.

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [1] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
       Thomson, South-Western: Ohio, US.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanLinexError
    >>> linex_error = MeanLinexError()
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    0.19802627763937575
    >>> linex_error = MeanLinexError(b=2)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    0.3960525552787515
    >>> linex_error = MeanLinexError(a=-1)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    0.2391800623225643
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> linex_error = MeanLinexError()
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    0.2700398392309829
    >>> linex_error = MeanLinexError(a=-1)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    0.49660966225813563
    >>> linex_error = MeanLinexError(multioutput='raw_values')
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    array([0.17220024, 0.36787944])
    >>> linex_error = MeanLinexError(multioutput=[0.3, 0.7])
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    0.30917568000716666
    """

    func = mean_linex_error

    def __init__(
        self,
        a=1.0,
        b=1.0,
        multioutput="uniform_average",
        multilevel="uniform_average",
    ):
        self.a = a
        self.b = b
        super().__init__(multioutput=multioutput, multilevel=multilevel)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}
        params2 = {"a": 0.5, "b": 2}
        return [params1, params2]


class RelativeLoss(BaseForecastingErrorMetricFunc):
    """Calculate relative loss of forecast versus benchmark forecast.

    Applies a forecasting performance metric to a set of forecasts and
    benchmark forecasts and reports ratio of the metric from the forecasts to
    the the metric from the benchmark forecasts. Relative loss output is
    non-negative floating point. The best value is 0.0.

    If the score of the benchmark predictions for a given loss function is zero
    then a large value is returned.

    This function allows the calculation of scale-free relative loss metrics.
    Unlike mean absolute scaled error (MASE) the function calculates the
    scale-free metric relative to a defined loss function on a benchmark
    method instead of the in-sample training data. Like MASE, metrics created
    using this function can be used to compare forecast methods on a single
    series and also to compare forecast accuracy between series.

    This is useful when a scale-free comparison is beneficial but the training
    data used to generate some (or all) predictions is unknown such as when
    comparing the loss of 3rd party forecasts or surveys of professional
    forecasters.

    Only metrics that do not require y_train are currently supported.

    Parameters
    ----------
    relative_loss_function : function
        Function to use in calculation relative loss.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).
        If 'uniform_average' (default), errors are mean-averaged across levels.
        If 'uniform_average_time', metric is applied to all data, ignoring level index.
        If 'raw_values', does not average errors across levels, hierarchy is retained.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import RelativeLoss
    >>> from sktime.performance_metrics.forecasting import mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8148148148148147
    >>> relative_mse = RelativeLoss(relative_loss_function=mean_squared_error)
    >>> relative_mse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.5178095088655261
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8490566037735847
    >>> relative_mae = RelativeLoss(multioutput='raw_values')
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.625     , 1.03448276])
    >>> relative_mae = RelativeLoss(multioutput=[0.3, 0.7])
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.927272727272727
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = relative_loss

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        relative_loss_function=mean_absolute_error,
    ):
        self.relative_loss_function = relative_loss_function
        super().__init__(multioutput=multioutput, multilevel=multilevel)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"relative_loss_function": mean_squared_error}
        return [params1, params2]
