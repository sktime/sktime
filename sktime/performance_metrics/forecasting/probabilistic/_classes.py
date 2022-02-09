# -*- coding: utf-8 -*-
# from imp import get_tag

import numpy as np

# TODO: add formal tests
import pandas as pd
from sklearn.metrics._regression import _check_reg_targets

from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric


class _BaseProbForecastingErrorMetric(_BaseForecastingErrorMetric):
    """Base class for probabilistic forecasting error metrics in sktime.

    Extends sktime's BaseMetric to the forecasting interface. Forecasting error
    metrics measure the error (loss) between forecasts and true values. Lower
    values are better.
    """

    # _tags = {"scitype:y_pred": "quantiles"}

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

        return self._evaluate(y_true, y_pred, multioutput, **kwargs)

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        raise NotImplementedError

    def evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Return the metric evaluated at each time point."""
        _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)

        return self._evaluate_by_index(y_true, y_pred, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        n = y_true.shape[0]
        out_series = pd.Series(index=pd.RangeIndex(0, n))
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
    alpha : The significance level that you want to find pinball loss for

    multioutput : string "uniform_average" or "raw_values" determines how multioutput
    results will be treated


    """

    def __init__(self, alpha, multioutput="uniform_average"):
        name = "PinballLoss"
        # _tags = {"scitype:pred": "quantile"}
        func = pinball_loss
        self.alpha = alpha
        super().__init__(func=func, name=name, multioutput=multioutput)

    def evaluate(self, y_true, y_pred, multioutput, **kwargs):
        """Evaluate the desired metric on given inputs."""
        # Input checks
        if isinstance(y_pred, pd.DataFrame):
            if y_pred.columns.get_level_values(0)[0] == "Quantiles":
                # Need to catch error if alpha not in dataframe
                y_pred = y_pred.iloc[
                    :, y_pred.columns.get_level_values(1) == self.alpha
                ].to_numpy()

        _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
        return self._evaluate(y_true, y_pred, multioutput)

    def _evaluate(self, y_true, y_pred, multioutput):
        # Return pinball loss
        alpha = self.alpha
        return pinball_loss(y_true, y_pred, alpha, multioutput)

    def evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Return the metric evaluated at each time point."""
        # Input checks
        if isinstance(y_pred, pd.DataFrame):
            if y_pred.columns.get_level_values(0)[0] == "Quantiles":
                # Need to catch error if alpha not in dataframe
                y_pred = y_pred.iloc[
                    :, y_pred.columns.get_level_values(1) == self.alpha
                ].to_numpy()

        return super().evaluate_by_index(y_true, y_pred, multioutput, **kwargs)


def pinball_loss(y_true, y_pred, alpha, multioutput):
    """Evaluate the pinball loss at the alpha quantile.

    Parameters
    ----------
    y_true : np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted alpha quantiles

    alpha : The significance
    """
    # check_alpha(alpha) # checks alpha is scalar between 0 and 1
    # check_quantile(y_pred, alpha) # checks that y_pred is quantile at correct

    # Return pinball loss, this was taken from sklearn.mean_pinball_loss()
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
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

    return np.average(output_errors, weights=multioutput)
