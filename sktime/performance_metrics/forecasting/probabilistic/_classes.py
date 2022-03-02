#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

# TODO: add formal tests
import pandas as pd
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils import check_array, check_consistent_length

from sktime.datatypes import check_is_scitype, convert
from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric


class _BaseProbaForecastingErrorMetric(_BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.
    """

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, func=None, name=None, multioutput="uniform_average"):
        self.multioutput = multioutput
        super().__init__(func, name=name)

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
                (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
                (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        return self.evaluate(y_true, y_pred, multioutput=self.multioutput, **kwargs)

    def evaluate(self, y_true, y_pred, multioutput=None, **kwargs):
        """Evaluate the desired metric on given inputs."""
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )
        # pass to inner function
        return self._evaluate(y_true_inner, y_pred_inner, multioutput, **kwargs)

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        # Default implementation relies on implementation of evaluate_by_index
        try:
            index_df = self._evaluate_by_index(y_true, y_pred, multioutput)
            return index_df.mean(axis=0)
        except RecursionError:
            print("Must implement one of _evaluate or _evaluate_by_index")

    def evaluate_by_index(self, y_true, y_pred, multioutput=None, **kwargs):
        """Return the metric evaluated at each time point."""
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )
        # pass to inner function
        return self._evaluate_by_index(y_true_inner, y_pred_inner, multioutput)

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
                    np.vstack((y_true[:i, :], y_true[i + 1:, :])),
                    np.vstack((y_pred[:i, :], y_pred[i + 1:, :])),
                    multioutput,
                )
            return out_series
        except RecursionError:
            print("Must implement one of _evaluate or _evaluate_by_index")

    def _check_alpha(self, alpha):
        if isinstance(alpha, list):
            alpha = np.array(alpha)
        if isinstance(alpha, np.ndarray):
            self.multiple_alphas = True
            if not all(((alpha > 0) & (alpha < 1))):
                raise ValueError("Alpha must be between 0 and 1.")
            return alpha
        if isinstance(alpha, float):
            self.multiple_alphas = False
            if not ((alpha > 0) & (alpha < 1)):
                raise ValueError("Alpha must be between 0 and 1.")
            return [alpha]
        raise TypeError("Alpha should be a float or numpy array")

    def _check_consistent_input(self, y_true, y_pred, multioutput):
        check_consistent_length(y_true, y_pred)

        check_array(y_true, ensure_2d=False)

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
            y_true,
            y_pred,
            multioutput)

        return y_true, y_pred_inner, multioutput


class PinballLoss(_BaseProbaForecastingErrorMetric):
    """Evaluate the pinball loss at the alpha quantile.

    Parameters
    ----------
    alpha : The significance level required for the pinball loss

    multioutput : string "uniform_average" or "raw_values" determines how multioutput
    results will be treated
    """

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average"):
        name = "PinballLoss"
        # func = pinball_loss
        super().__init__(name=name, multioutput=multioutput)

    def get_alpha_from(y_pred):
        """Fetch the alphas present in y_pred."""
        return

    def _evaluate(self, y_true, y_pred, multioutput):
        # implement this function, or write af ew lines
        alpha = self.get_alpha_from(y_pred)
        # from old evaluate
        out = [None] * len(self.alphas)
        for i, alpha in enumerate(self.alphas):
            cur_preds = y_pred.iloc[
                :, y_pred.columns.get_level_values(1) == alpha
            ].to_numpy()
            # input checks should have already been done in "evaluate"
            _, y_true, cur_preds, multioutput = _check_reg_targets(
                y_true, cur_preds, multioutput
            )
            out[i] = pinball_loss(y_true, cur_preds, alpha, multioutput)
        out_df = pd.DataFrame([out], columns=self.alphas)
        return out_df

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):

        alpha = self.get_alpha_from(y_pred)

        n = len(y_true)
        out = np.full([n, len(self.alphas)], None)
        for i, alpha in enumerate(self.alphas):
            cur_preds = y_pred.iloc[
                :, y_pred.columns.get_level_values(1) == alpha
            ].to_numpy()
            _, y_true, cur_preds, multioutput = _check_reg_targets(
                y_true, cur_preds, multioutput
            )
            out[:, i] = pinball_loss(y_true, cur_preds, alpha, multioutput, av=False)[
                :, 0
            ]

        out_df = pd.DataFrame(out, index=y_pred.index, columns=self.alphas)
        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        return {}


def pinball_loss(y_true, y_pred, alpha, multioutput, av=True):
    """Evaluate the pinball loss at the alpha quantile.

    Parameters
    ----------
    y_true : np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted alpha quantiles

    alpha : The quantile at which the pinball loss is to be evaluated.

    av : Boolean determining whether average pinball loss should be returned.

    Returns
    -------
    out : np.ndarray of evaluated metric values.
    """
    # check_alpha(alpha) # checks alpha is scalar between 0 and 1
    # check_quantile(y_pred, alpha) # checks that y_pred is quantile at correct

    # Return pinball loss, this was taken from sklearn.mean_pinball_loss()
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    # (((sign * diff).T * alpha) - (((1-sign) * diff).T * (1-alpha)))
    # Works as vectorization for multiple alpha
    if not av:
        return loss

    output_errors = np.average(loss, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
        else:
            raise ValueError(
                "multioutput is expected to be 'raw_values' "
                "or 'uniform_average' but we got %r"
                " instead." % multioutput
            )
    out = np.average(output_errors, weights=multioutput)
    return out
