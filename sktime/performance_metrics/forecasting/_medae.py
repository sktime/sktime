#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.performance_metrics.forecasting._common import _weighted_percentile


class MedianAbsoluteError(BaseForecastingErrorMetric):
    r"""Median absolute error (MdAE).

    MdAE output is non-negative floating point. The best value is 0.0.

    Like MAE, MdAE is on the same scale as the data. Because MAE takes the
    absolute value of the forecast error rather than squaring it, MAE penalizes
    large errors to a lesser degree than MdSE or RdMSE.

    Taking the median instead of the mean of the absolute errors also makes
    this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    For a univariate, non-hierarchical sample of true values :math:`y_1, \dots, y_n`
    and predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns the Median Absolute Error,
    :math:`\text{median}\left(|y_i - \widehat{y}_i|\right)_{i=1}^{n}`.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    the absolute error at that time index, :math:`|y_i - \widehat{y}_i|`,
    for all time indices :math:`t_1, \dots, t_n` in the input.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.

        * If array-like, values used as weights to average the errors.
        * If ``'raw_values'``,
          returns a full set of errors in case of multioutput input.
        * If ``'uniform_average'``,
          errors of all outputs are averaged with uniform weight.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Determines averaging over time points in direct call to metric object.

        * If False, direct call to the metric object averages over time points,
          equivalent to a call of the``evaluate`` method.
        * If True, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

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
    np.float64(0.5)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdae(y_true, y_pred)
    np.float64(0.75)
    >>> mdae = MedianAbsoluteError(multioutput='raw_values')
    >>> mdae(y_true, y_pred)
    array([0.5, 1. ])
    >>> mdae = MedianAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdae(y_true, y_pred)
    np.float64(0.85)
    """

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
        multioutput = self.multioutput
        sample_weight = kwargs.get("sample_weight", None)

        if sample_weight is None:
            output_errors = np.median(np.abs(y_pred - y_true), axis=0)
        else:
            output_errors = _weighted_percentile(
                np.abs(y_pred - y_true), sample_weight=sample_weight
            )

        return self._handle_multioutput(output_errors, multioutput)

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
            Calculated metric, by time point.
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
                return raw_values.median(axis=1)

        # else, we expect multioutput to be array-like
        return raw_values.dot(multioutput)
