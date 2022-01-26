# -*- coding: utf-8 -*-
# from imp import get_tag
import pandas as pd

from sktime.performance_metrics.forecasting._classes import _BaseForecastingErrorMetric


class _BaseProbForecastingErrorMetric(_BaseForecastingErrorMetric):
    """Base class for defining probabilistic forecasting error metrics in sktime.

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
        # input_check()
        return self._evaluate(y_true, y_pred, multioutput, **kwargs)

    def _evaluate(self, y_true, y_pred, multioutput, **kwargs):
        raise NotImplementedError

    def evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Return the metric evaluated at each time point."""
        # input_check()
        n = len(y_true)
        out_series = pd.series()
        x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
        for _i in range(n):
            out_series[_i] = n * x_bar - (n - 1) * self.evaluate(
                y_true[-_i], y_pred[-_i]
            )
        return out_series


class PinballLoss(_BaseProbForecastingErrorMetric):
    """Evaluate the pinball loss at the alpha quantile.

    Describe parameters/output

    Give some examples of use
    """

    def __init__(self, alpha, multioutput="uniform_average"):
        name = "PinballLoss"
        # _tags = {"scitype:pred": "quantile"}
        func = pinball_loss
        self.alpha = alpha
        super().__init__(func=func, name=name, multioutput=multioutput)

    def evaluate(self, y_true, y_pred, multioutput, **kwargs):
        """Evaluate the desired metric on given inputs."""
        # input_check()
        n = y_true.shape[1]
        for _i in range(n):
            # need to change this
            return self._evaluate(y_true, y_pred, **kwargs)

    def _evaluate(self, y_true, y_pred, **kwargs):
        # Return pinball loss
        alpha = self.alpha
        if y_true >= y_pred:
            return alpha * (y_true - y_pred)
        else:
            return (1 - alpha) * (y_true - y_pred)

    def evaluate_by_index(self, y_true, y_pred, multioutput, **kwargs):
        """Return the metric evaluated at each time point."""
        # input_check()
        n = len(y_true)
        out_series = pd.series()  #
        x_bar = self.evaluate(y_true, y_pred, multioutput, **kwargs)
        for _i in range(n):
            out_series[_i] = n * x_bar - (n - 1) * self.evaluate(
                y_true[-_i], y_pred[-_i]
            )
        return out_series


def pinball_loss(y_true, y_pred, alpha):
    """Evaluate the pinball loss at the alpha quantile.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted alpha quantiles

    alpha : The significance
    """
    # check_alpha(alpha) # checks alpha is scalar between 0 and 1
    # check_quantile(y_pred, alpha) # checks that y_pred is quantile at correct

    # Return pinball loss
    if y_true >= y_pred:
        return alpha * (y_true - y_pred)
    else:
        return (1 - alpha) * (y_true - y_pred)
