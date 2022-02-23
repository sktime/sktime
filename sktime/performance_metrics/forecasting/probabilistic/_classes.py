#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

# TODO: add formal tests
import pandas as pd
from sklearn.metrics._regression import _check_reg_targets

from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric


def _check_y_pred_type(metric_object, y_pred):
    # Checks y_pred is of correct type to work with metric.
    # If it isn't it will convert it if possible
    input_type = metric_object.get_tag("scitype:y_pred")
    if input_type == "quantiles":
        if isinstance(y_pred, pd.DataFrame):
            return y_pred
        # if distribution object then get the relevant quantiles and return.


class _BaseProbForecastingErrorMetric(_BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.
    """

    _tags = {"scitype:y_pred": "quantiles"}

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

    def evaluate(self, y_true, y_pred, multioutput, **kwargs):
        """Evaluate the desired metric on given inputs."""
        # Input checks
        _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
        # y_pred = _check_y_pred_type()

        return self._evaluate(y_true, y_pred, multioutput, **kwargs)

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        raise NotImplementedError

    def evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Return the metric evaluated at each time point."""
        _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
        # y_pred = _check_y_pred_type(self)

        return self._evaluate_by_index(y_true, y_pred, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
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


class PinballLoss(_BaseProbForecastingErrorMetric):
    """Evaluate the pinball loss at the alpha quantile.

    Parameters
    ----------
    alpha : The significance level required for the pinball loss

    multioutput : string "uniform_average" or "raw_values" determines how multioutput
    results will be treated
    """

    def __init__(self, alphas, multioutput="uniform_average"):
        name = "PinballLoss"
        # _tags = {"scitype:pred": "quantile"}
        func = pinball_loss
        self.alphas = self._check_alpha(alphas)
        super().__init__(func=func, name=name, multioutput=multioutput)

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

    def _check_y_pred_quantiles(self, y_pred):
        assert isinstance(y_pred, pd.DataFrame)
        assert y_pred.columns.get_level_values(0)[0] == "Quantiles"
        pred_alphas = y_pred.columns.get_level_values(1)
        if not all([alph in pred_alphas for alph in self.alphas]):
            raise ValueError(f"Predictions must include the {self.alphas} quantile.")

    def evaluate(self, y_true, y_pred, multioutput="uniform_average", **kwargs):
        """Evaluate the desired metric on given inputs."""
        # Input checks
        self._check_y_pred_quantiles(y_pred)
        out = [None] * len(self.alphas)
        for i, alpha in enumerate(self.alphas):
            cur_preds = y_pred.iloc[
                :, y_pred.columns.get_level_values(1) == alpha
            ].to_numpy()
            _, y_true, cur_preds, multioutput = _check_reg_targets(
                y_true, cur_preds, multioutput
            )
            out[i] = self._evaluate(y_true, cur_preds, alpha, multioutput)
        out_df = pd.DataFrame([out], columns=self.alphas)
        return out_df

    def _evaluate(self, y_true, y_pred, alpha, multioutput):
        # Return pinball loss
        return pinball_loss(y_true, y_pred, alpha, multioutput)

    def evaluate_by_index(self, y_true, y_pred, multioutput="uniform_average"):
        """Return the metric evaluated at each time point."""
        # Input checks
        n = len(y_true)
        self._check_y_pred_quantiles(y_pred)
        out = np.full([n, len(self.alphas)], None)
        for i, alpha in enumerate(self.alphas):
            cur_preds = y_pred.iloc[
                :, y_pred.columns.get_level_values(1) == alpha
            ].to_numpy()
            _, y_true, cur_preds, multioutput = _check_reg_targets(
                y_true, cur_preds, multioutput
            )
            out[:, i] = self._evaluate_by_index(y_true, cur_preds, alpha, multioutput)[
                :, 0
            ]

        out_df = pd.DataFrame(out, index=y_pred.index, columns=self.alphas)
        return out_df

    def _evaluate_by_index(self, y_true, y_pred, alpha, multioutput, **kwargs):
        return pinball_loss(y_true, y_pred, alpha, multioutput, av=False)

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
