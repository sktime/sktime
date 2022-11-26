#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
from logging import warning

import numpy as np
import pandas as pd
from sklearn.utils import check_array, check_consistent_length

from sktime.datatypes import check_is_scitype, convert
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

# TODO: Rework tests now


class _BaseProbaForecastingErrorMetric(BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    score_average : bool, optional, default=True
        for interval and quantile losses only
            if True, metric/loss is averaged by upper/lower and/or quantile
            if False, metric/loss is not averaged by upper/lower and/or quantile
    """

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.multioutput = multioutput
        self.score_average = score_average
        super().__init__(multioutput=multioutput)

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must be at fh and for variables equal to those in y_true.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        return self.evaluate(y_true, y_pred, multioutput=self.multioutput, **kwargs)

    def evaluate(self, y_true, y_pred, multioutput=None, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must be at fh and for variables equal to those in y_true

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warning(
                "Dropping 0 width interval, don't include 0.5 quantile\
            for interval metrics."
            )

        # pass to inner function
        out = self._evaluate(y_true_inner, y_pred_inner, multioutput, **kwargs)

        if self.score_average and multioutput == "uniform_average":
            out = float(out.mean(axis=1))  # average over all
        if self.score_average and multioutput == "raw_values":
            out = out.groupby(axis=1, level=0).mean()  # average over scores
        if not self.score_average and multioutput == "uniform_average":
            out = out.groupby(axis=1, level=1).mean()  # average over variables
        if not self.score_average and multioutput == "raw_values":
            out = out  # don't average

        if isinstance(out, pd.DataFrame):
            out = out.squeeze(axis=0)

        return out

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        """Evaluate the desired metric on given inputs.

        Parameters
        ----------
        y_true : pd.DataFrame or of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.DataFrame of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : pd.DataFrame of shape (, n_outputs), calculated loss metric.
        """
        # Default implementation relies on implementation of evaluate_by_index
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, multioutput)
            out_df = pd.DataFrame(index_df.mean(axis=0)).T
            out_df.columns = index_df.columns
            return out_df
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def evaluate_by_index(self, y_true, y_pred, multioutput=None, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must be at fh and for variables equal to those in y_true

        multioutput : string "uniform_average" or "raw_values" determines how\
            multioutput results will be treated.

        Returns
        -------
        loss : pd.DataFrame of length len(fh), with calculated metric value(s)
            i-th column contains metric value(s) for prediction at i-th fh element
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warning(
                "Dropping 0 width interval, don't include 0.5 quantile\
            for interval metrics."
            )

        # pass to inner function
        out = self._evaluate_by_index(y_true_inner, y_pred_inner, multioutput, **kwargs)

        if self.score_average and multioutput == "uniform_average":
            out = out.mean(axis=1)  # average over all
        if self.score_average and multioutput == "raw_values":
            out = out.groupby(axis=1, level=0).mean()  # average over scores
        if not self.score_average and multioutput == "uniform_average":
            out = out.groupby(axis=1, level=1).mean()  # average over variables
        if not self.score_average and multioutput == "raw_values":
            out = out  # don't average

        return out

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        By default this uses _evaluate to find jackknifed pseudosamples. This
        estimates the error at each of the time points.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        n = y_true.shape[0]
        out_series = pd.Series(index=y_pred.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
            for i in range(n):
                out_series[i] = n * x_bar - (n - 1) * self.evaluate(
                    np.vstack((y_true[:i, :], y_true[i + 1 :, :])),  # noqa
                    np.vstack((y_pred[:i, :], y_pred[i + 1 :, :])),  # noqa
                    multioutput,
                )
            return out_series
        except RecursionError:
            raise RecursionError(
                "Must implement one of _evaluate or _evaluate_by_index"
            )

    def _check_consistent_input(self, y_true, y_pred, multioutput):
        check_consistent_length(y_true, y_pred)

        y_true = check_array(y_true, ensure_2d=False)

        if not isinstance(y_pred, pd.DataFrame):
            raise ValueError("y_pred should be a dataframe.")

        if not all(y_pred.dtypes == float):
            raise ValueError("Data should be numeric.")

        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))

        n_outputs = y_true.shape[1]

        allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    "Allowed 'multioutput' string values are {}. "
                    "You provided multioutput={!r}".format(
                        allowed_multioutput_str, multioutput
                    )
                )
        elif multioutput is not None:
            multioutput = check_array(multioutput, ensure_2d=False)
            if n_outputs == 1:
                raise ValueError("Custom weights are useful only in multi-output case.")
            elif n_outputs != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), n_outputs)
                )

        return y_true, y_pred, multioutput

    def _check_ys(self, y_true, y_pred, multioutput):
        if multioutput is None:
            multioutput = self.multioutput
        valid, msg, metadata = check_is_scitype(
            y_pred, scitype="Proba", return_metadata=True, var_name="y_pred"
        )

        if not valid:
            raise TypeError(msg)

        y_pred_mtype = metadata["mtype"]
        inner_y_pred_mtype = self.get_tag("scitype:y_pred")

        y_pred_inner = convert(
            y_pred,
            from_type=y_pred_mtype,
            to_type=inner_y_pred_mtype,
            as_scitype="Proba",
        )

        if inner_y_pred_mtype == "pred_interval":
            if 0.0 in y_pred_inner.columns.get_level_values(1):
                for var in y_pred_inner.columns.get_level_values(0):
                    y_pred_inner[var, 0.0, "upper"] = y_pred_inner[var, 0.0, "lower"]

        y_pred_inner.sort_index(axis=1, level=[0, 1], inplace=True)

        y_true, y_pred, multioutput = self._check_consistent_input(
            y_true, y_pred, multioutput
        )

        return y_true, y_pred_inner, multioutput

    def _get_alpha_from(self, y_pred):
        """Fetch the alphas present in y_pred."""
        alphas = np.unique(list(y_pred.columns.get_level_values(1)))
        if not all(((alphas > 0) & (alphas < 1))):
            raise ValueError("Alpha must be between 0 and 1.")

        return alphas

    def _check_alpha(self, alpha):
        """Check alpha input and coerce to np.ndarray."""
        if alpha is None:
            return None

        if isinstance(alpha, float):
            alpha = [alpha]

        if not isinstance(alpha, np.ndarray):
            alpha = np.asarray(alpha)

        if not all(((alpha > 0) & (alpha < 1))):
            raise ValueError("Alpha must be between 0 and 1.")

        return alpha

    def _handle_multioutput(self, loss, multioutput):
        """Specificies how multivariate outputs should be handled.

        Parameters
        ----------
        loss : float, np.ndarray the evaluated metric value.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return loss
            elif multioutput == "uniform_average":
                # pass None as weights to np.average: uniform mean
                multioutput = None
            else:
                raise ValueError(
                    "multioutput is expected to be 'raw_values' "
                    "or 'uniform_average' but we got %r"
                    " instead." % multioutput
                )

        if loss.ndim > 1:
            out = np.average(loss, weights=multioutput, axis=1)
        else:
            out = np.average(loss, weights=multioutput)
        return out


class PinballLoss(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.

    alpha (optional) : float, list or np.ndarray, specifies what quantiles to \
        evaluate metric at.
    """

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True, alpha=None):
        self.score_average = score_average
        self.alpha = alpha
        self._alpha = self._check_alpha(alpha)
        self.metric_args = {"alpha": self._alpha}
        super().__init__(multioutput=multioutput, score_average=score_average)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target value`s.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values"
            Determines how multioutput results will be treated.
        """
        alpha = self._alpha
        y_pred_alphas = self._get_alpha_from(y_pred)
        if alpha is None:
            alphas = y_pred_alphas
        else:
            # if alpha was provided, check whether  they are predicted
            #   if not all alpha are observed, raise a ValueError
            if not np.isin(alpha, y_pred_alphas).all():
                # todo: make error msg more informative
                #   which alphas are missing
                msg = "not all quantile values in alpha are available in y_pred"
                raise ValueError(msg)
            else:
                alphas = alpha

        alphas = self._check_alpha(alphas)

        alpha_preds = y_pred.iloc[
            :, [x in alphas for x in y_pred.columns.get_level_values(1)]
        ]

        alpha_preds_np = alpha_preds.to_numpy()
        alpha_mat = np.repeat(
            (alpha_preds.columns.get_level_values(1).to_numpy().reshape(1, -1)),
            repeats=y_true.shape[0],
            axis=0,
        )

        y_true_np = np.repeat(y_true, axis=1, repeats=len(alphas))
        diff = y_true_np - alpha_preds_np
        sign = (diff >= 0).astype(diff.dtype)
        loss = alpha_mat * sign * diff - (1 - alpha_mat) * (1 - sign) * diff

        out_df = pd.DataFrame(loss, columns=alpha_preds.columns)

        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"alpha": [0.1, 0.5, 0.9]}
        return [params1, params2]


class EmpiricalCoverage(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": False,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.score_average = score_average
        self.multioutput = multioutput
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true
        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        truth_array = (y_true_np > lower).astype(int) * (y_true_np < upper).astype(int)

        out_df = pd.DataFrame(
            truth_array, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        return [params1]


class ConstraintViolation(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.score_average = score_average
        self.multioutput = multioutput
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true

        if y_true_np.ndim == 1:
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        int_distance = ((y_true_np < lower).astype(int) * abs(lower - y_true_np)) + (
            (y_true_np > upper).astype(int) * abs(y_true_np - upper)
        )

        out_df = pd.DataFrame(
            int_distance, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        return [params1]
