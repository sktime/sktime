#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

# TODO: add formal tests
import pandas as pd
from sklearn.metrics._regression import _check_reg_targets

from sktime.datatypes import check_is_scitype, convert
from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric


def _check_y_pred_type(metric_object, y_pred):
    # Checks y_pred is of correct type to work with metric.
    # If it isn't it will convert it if possible
    input_type = metric_object.get_tag("scitype:y_pred")
    if input_type == "quantiles":
        if isinstance(y_pred, pd.DataFrame):
            return y_pred
        # if distribution object then get the relevant quantiles and return.


class _BaseProbaForecastingErrorMetric(_BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.
    """

    _tags = {
        "scitype:y_pred": "quantiles",
        "lower_is_better": True,
    }

    def __init__(self, func, name=None, multioutput="uniform_average"):
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
        # todo: should have a default implementation, which is just the mean over
        #       _evaluate_by_index
        raise NotImplementedError

    def evaluate_by_index(self, y_true, y_pred, multioutput=None, **kwargs):
        """Return the metric evaluated at each time point."""
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, multioutput
        )
        # pass to inner function
        return self._evaluate_by_index(y_true_inner, y_pred_inner, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        # docstring/comments should explain that default is jackknife pseudosamples
        n = y_true.shape[0]
        out_series = pd.Series(index=y_pred.index)
        x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
        for i in range(n):
            out_series[i] = n * x_bar - (n - 1) * self.evaluate(
                np.vstack((y_true[:i, :], y_true[i + 1 :, :])),
                np.vstack((y_pred[:i, :], y_pred[i + 1 :, :])),
                multioutput,
            )
        return out_series

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

    def _check_ys(self, y_true, y_pred, multioutput):
        if multioutput is None:
            multioutput = self.multioutput
        _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
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
            as_scitype="Proba"
        )
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
        "scitype:y_pred": "quantiles",
        "lower_is_better": True,
    }

    def __init__(self, alphas, multioutput="uniform_average"):
        name = "PinballLoss"
        # _tags = {"scitype:pred": "quantile"}
        func = pinball_loss
        super().__init__(func=func, name=name, multioutput=multioutput)

    def _evaluate(self, y_true, y_pred, multioutput):
        # implement this function, or write af ew lines
        alpha = get_alpha_from(y_pred)
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

        alpha = get_alpha_from(y_pred)

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
        return {"alphas": 0.5}


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
