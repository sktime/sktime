#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
from sklearn.utils import check_array, check_consistent_length

from sktime.datatypes import check_is_scitype, convert
from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric

# TODO: add formal tests
# TODO: Adapt score_average to ensure doesn't interfere with multioutput
#        treatment (both average across columns)


class _BaseProbaForecastingErrorMetric(_BaseForecastingErrorMetric):
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

    def __init__(
        self,
        func=None,
        name=None,
        multioutput="uniform_average",
        score_average=True,
    ):
        self.multioutput = multioutput
        self.score_average = score_average
        super().__init__(func, name=name)

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must be at fh and for variables equal to those in y_true

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
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
        # pass to inner function
        out_df = self._evaluate(y_true_inner, y_pred_inner, multioutput, **kwargs)
        if self.score_average:
            out_df = out_df.mean(axis=1)

        # unwrap the case where resulting data frame has one entry
        if self.score_average and multioutput in "uniform_average":
            out_df = out_df.iloc[0]
        return out_df

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        # Default implementation relies on implementation of evaluate_by_index
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, multioutput)
            return index_df.mean(axis=0)
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
        # pass to inner function
        out_df = self._evaluate_by_index(
            y_true_inner, y_pred_inner, multioutput, **kwargs
        )

        if self.score_average:
            out_df = out_df.mean(axis=1)
        return out_df

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Logic for finding the metric evaluated at each index.

        By default this uses _evaluate to find jackknifed pseudosamples. This
        estimates the error at each of the time points.
        """
        n = y_true.shape[0]
        out_series = pd.Series(index=y_pred.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
            for i in range(n):
                out_series[i] = n * x_bar - (n - 1) * self.evaluate(
                    np.vstack((y_true[:i, :], y_true[i + 1 :, :])),  # noqa
                    np.vstack((y_pred[:i, :], y_pred[i + 1 :, :])),
                    multioutput,
                )
            return out_series
        except RecursionError:
            RecursionError("Must implement one of _evaluate or _evaluate_by_index")

    def _check_consistent_input(self, y_true, y_pred, multioutput):
        check_consistent_length(y_true, y_pred)

        y_true = check_array(y_true, ensure_2d=False)

        if not isinstance(y_pred, pd.DataFrame):
            ValueError("y_pred should be a dataframe.")

        if not all(y_pred.dtypes == float):
            ValueError("Data should be numeric.")

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
        """Check that alpha input is valid."""
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
    multioutput : string "uniform_average" or "raw_values" determines how multioutput
    results will be treated
    """

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(
        self,
        multioutput="uniform_average",
        score_average=True,
        alpha=None,
    ):
        name = "PinballLoss"
        self.score_average = score_average
        self.alpha = self._check_alpha(alpha)
        self.metric_args = {"alpha": alpha}
        super().__init__(
            name=name, multioutput=multioutput, score_average=score_average
        )

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        alpha = self.alpha
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

        out = [None] * len(alphas)
        for i, alpha in enumerate(alphas):
            alpha_preds = y_pred.iloc[
                :, y_pred.columns.get_level_values(1) == alpha
            ].to_numpy()
            diff = y_true - alpha_preds
            sign = (diff >= 0).astype(diff.dtype)
            loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff

            loss = np.average(loss, axis=0)

            out[i] = self._handle_multioutput(loss, multioutput)

        out_df = pd.DataFrame([out], columns=alphas)
        return out_df

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        alphas = self._get_alpha_from(y_pred)

        n = len(y_true)
        out = np.full([n, len(alphas)], None)
        for i, alpha in enumerate(alphas):
            alpha_preds = y_pred.iloc[
                :, y_pred.columns.get_level_values(1) == alpha
            ].to_numpy()
            diff = y_true - alpha_preds
            sign = (diff >= 0).astype(diff.dtype)
            loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff

            out[:, i] = self._handle_multioutput(loss, multioutput)

        out_df = pd.DataFrame(out, index=y_pred.index, columns=alphas)
        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"alpha": [0.1, 0.5, 0.9]}
        return [params1, params2]
