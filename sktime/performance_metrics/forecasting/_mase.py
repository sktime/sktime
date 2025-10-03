#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np

from sktime.performance_metrics.forecasting._base import (
    BaseForecastingErrorMetric,
    _ScaledMetricTags,
)


class MeanAbsoluteScaledError(_ScaledMetricTags, BaseForecastingErrorMetric):
    r"""Mean absolute scaled error (MASE).

    MASE output is non-negative floating point, in fractional units
    relative to a specified denominator.
    Lower is better, and the lowest possible value is 0.0.

    For a univariate, non-hierarchical sample of
    true values :math:`y_1, \dots, y_n`,
    pred values :math:`\widehat{y}_1, \dots, \widehat{y}_n` (in :math:`\mathbb{R}`),
    and in-sample training values :math:`y_1^{\text{train}}, \dots, y_m^{\text{train}}`,
    ``evaluate`` or call returns the Mean Absolute Scaled Error (MASE), defined as:

    .. math::
    \text{MASE} = \frac{
        \frac{1}{n} \sum_{i=1}^n |y_i - \widehat{y}_i|
    }{
        \frac{1}{m - s} \sum_{j=s+1}^m |y^{\text{train}}_j - y^{\text{train}}_{j-s}|
    }

    where :math:`s` is the seasonal periodicity (`sp`), and
    the denominator is the in-sample mean absolute error of the seasonal naive forecast.

    To avoid division by zero, the denominator above is replaced by ``eps``
    if it is smaller than ``eps``; the value of ``eps`` defaults to
    ``np.finfo(np.float64).eps`` if not specified.

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

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable errors are returned.

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
        * If ``True``, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

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
    np.float64(0.18333333333333335)
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mase(y_true, y_pred, y_train=y_train)
    np.float64(0.18181818181818182)
    >>> mase = MeanAbsoluteScaledError(multioutput='raw_values')
    >>> mase(y_true, y_pred, y_train=y_train)
    array([0.10526316, 0.28571429])
    >>> mase = MeanAbsoluteScaledError(multioutput=[0.3, 0.7])
    >>> mase(y_true, y_pred, y_train=y_train)
    np.float64(0.21935483870967742)
    """

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

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        Parameters
        ----------
        y_true : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_pred : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Predicted values to evaluate.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_train : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Training data used to calculate the naive forecasting error.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).

            * pd.Series if self.multioutput="uniform_average" or array-like;
              index is equal to index of y_true;
              entry at index i is metric at time i, averaged over variables.
            * pd.DataFrame if self.multioutput="raw_values";
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j.
        """
        y_train = kwargs["y_train"]
        multioutput = self.multioutput
        sp = self.sp

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        raw_values = (y_true - y_pred).abs()
        raw_values = self._get_weighted_df(raw_values, **kwargs)

        # Calculating the naive forecasting error
        naive_forecast_true = y_train[sp:]
        naive_forecast_pred = y_train[:-sp]
        naive_diff = (naive_forecast_true - naive_forecast_pred.values).abs()
        naive_error = naive_diff.mean()

        raw_values = raw_values / np.maximum(naive_error, eps)

        raw_values = self._get_weighted_df(raw_values, **kwargs)

        return self._handle_multioutput(raw_values, multioutput)

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
        params1 = {}
        params2 = {"sp": 2}
        return [params1, params2]
