#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Theil's U2 statistic for forecast evaluation."""

__author__ = ["michaelellis003"]

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import (
    BaseForecastingErrorMetric,
    _ScaledMetricTags,
)


class TheilU2(_ScaledMetricTags, BaseForecastingErrorMetric):
    r"""Theil's U2 statistic comparing forecast to naive (random walk) forecast.

    Theil's U2 compares the root mean squared error of the forecast to the
    root mean squared error of a naive (random walk) forecast. Output is
    non-negative floating point, lower is better, with 0.0 indicating a
    perfect forecast and 1.0 indicating performance equal to the naive
    forecast.

    For a univariate, non-hierarchical sample of
    true values :math:`y_1, \dots, y_n`,
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    and in-sample training values
    :math:`y_1^{\text{train}}, \dots, y_m^{\text{train}}`,
    ``evaluate`` or call returns

    .. math::
        U_2 = \sqrt{
        \frac{
            \frac{1}{n}\sum_{i=1}^{n}(y_i - \widehat{y}_i)^2
        }{
            \frac{1}{n}\sum_{i=1}^{n}(y_i - y_{i-s})^2
        }
        }

    where :math:`s` is the seasonal periodicity (``sp``, default 1), and
    :math:`y_{i-s}` is the naive seasonal forecast: the actual value
    :math:`s` periods before time :math:`i`. For the first :math:`s`
    test-period values, the naive forecast uses the last :math:`s` values
    from ``y_train``.

    To avoid division by zero, the denominator is clamped to ``eps``.

    ``evaluate_by_index`` returns jackknife pseudo-values since U2 is
    ``sqrt(mean/mean)``, not a simple per-index mean.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity of the data. ``sp=1`` corresponds to the
        standard random-walk naive forecast.

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable errors are
          returned.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call to the metric object averages over time points,
          equivalent to a call of the ``evaluate`` method.
        * If ``True``, direct call to the metric object evaluates the metric at
          each time point, equivalent to a call of the ``evaluate_by_index``
          method.

    See Also
    --------
    RelativeLoss
    MeanAbsoluteScaledError

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Theil, H. (1966). "Applied Economic Forecasting", North-Holland,
    Amsterdam.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import TheilU2
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> theilu2 = TheilU2()
    >>> theilu2(y_true, y_pred, y_train=y_train)
    np.float64(0.17226798597767884)
    """

    _tags = {
        "authors": ["michaelellis003"],
    }

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        sp=1,
        by_index=False,
        eps=None,
    ):
        self.sp = sp
        self.eps = eps
        super().__init__(
            multioutput=multioutput, multilevel=multilevel, by_index=by_index
        )

    def _build_naive_pred(self, y_true, y_train):
        """Build naive seasonal forecast for the test period.

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth test values.
        y_train : pandas.DataFrame
            Training data.

        Returns
        -------
        y_naive : np.ndarray, same shape as y_true
            Naive seasonal forecast values.
        """
        sp = self.sp
        n = y_true.shape[0]
        y_train_vals = y_train.values
        y_true_vals = y_true.values

        if sp > n:
            raise ValueError(
                f"Seasonal periodicity sp={sp} exceeds the number of test "
                f"observations ({n}). TheilU2 requires len(y_true) >= sp."
            )

        # naive forecast: value from sp periods ago
        # first sp values come from end of y_train, rest from y_true shifted
        naive_source = np.concatenate(
            [y_train_vals[-sp:], y_true_vals[:-sp]], axis=0
        )
        return naive_source

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, possibly averaged by variable.
        """
        y_train = kwargs["y_train"]
        multioutput = self.multioutput

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        y_true_vals = y_true.values
        y_pred_vals = y_pred.values
        y_naive = self._build_naive_pred(y_true, y_train)

        # MSE of forecast
        sqe_forecast = (y_true_vals - y_pred_vals) ** 2
        sqe_forecast_df = pd.DataFrame(
            sqe_forecast, index=y_true.index, columns=y_true.columns
        )
        sqe_forecast_df = self._get_weighted_df(sqe_forecast_df, **kwargs)
        mse_forecast = sqe_forecast_df.mean()

        # MSE of naive forecast
        sqe_naive = (y_true_vals - y_naive) ** 2
        sqe_naive_df = pd.DataFrame(
            sqe_naive, index=y_true.index, columns=y_true.columns
        )
        sqe_naive_df = self._get_weighted_df(sqe_naive_df, **kwargs)
        mse_naive = sqe_naive_df.mean()

        result = (mse_forecast / np.maximum(mse_naive, eps)).pow(0.5)

        return self._handle_multioutput(result, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from
        evaluate_by_index

        Uses jackknife pseudo-values since U2 is sqrt(mean/mean),
        not a simple per-index mean.

        Parameters
        ----------
        y_true : pandas.DataFrame
            Ground truth (correct) target values.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (jackknife pseudo-values).
        """
        y_train = kwargs["y_train"]
        multioutput = self.multioutput

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        n = y_true.shape[0]
        y_true_vals = y_true.values
        y_pred_vals = y_pred.values
        y_naive = self._build_naive_pred(y_true, y_train)

        sqe_forecast = (y_true_vals - y_pred_vals) ** 2
        sqe_forecast_df = pd.DataFrame(
            sqe_forecast, index=y_true.index, columns=y_true.columns
        )
        sqe_forecast_df = self._get_weighted_df(sqe_forecast_df, **kwargs)

        sqe_naive = (y_true_vals - y_naive) ** 2
        sqe_naive_df = pd.DataFrame(
            sqe_naive, index=y_true.index, columns=y_true.columns
        )
        sqe_naive_df = self._get_weighted_df(sqe_naive_df, **kwargs)

        # full metric value
        mse_forecast = sqe_forecast_df.mean(axis=0)
        mse_naive = sqe_naive_df.mean(axis=0)
        full_val = (mse_forecast / np.maximum(mse_naive, eps)).pow(0.5)

        # jackknife leave-one-out
        total_forecast = sqe_forecast_df.sum(axis=0)
        total_naive = sqe_naive_df.sum(axis=0)

        jack_mse_forecast = (total_forecast - sqe_forecast_df) / (n - 1)
        jack_mse_naive = (total_naive - sqe_naive_df) / (n - 1)
        jack_val = (jack_mse_forecast / np.maximum(jack_mse_naive, eps)).pow(0.5)

        pseudo_values = n * full_val - (n - 1) * jack_val

        return self._handle_multioutput(pseudo_values, multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or ``MyClass(**params[i])``
            creates a valid test instance.
            ``create_test_instance`` uses the first (or only) dictionary in
            ``params``
        """
        params1 = {}
        params2 = {"sp": 2}
        return [params1, params2]
