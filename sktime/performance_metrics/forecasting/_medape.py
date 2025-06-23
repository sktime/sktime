#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    median_absolute_percentage_error,
)


class MedianAbsolutePercentageError(BaseForecastingErrorMetricFunc):
    r"""Median absolute percentage error (MdAPE) or symmetric version.

    For a univariate, non-hierarchical sample of true values :math:`y_1, \dots, y_n`
    and predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns the Median Absolute Percentage Error,
    :math:`median(\left|\frac{y_i - \widehat{y}_i}{y_i} \right|)`.
    (the time indices are not used)

    if ``symmetric`` is True then calculates
    symmetric Median Absolute Percentage Error (sMdAPE), defined as
    :math:`median(\frac{2|y_i-\widehat{y}_i|}{|y_i|+|\widehat{y}_i|})`.

    Both MdAPE and sMdAPE output non-negative floating point which is in fractional
    units rather than percentage. The best value is 0.0.

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because it takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    MAPE has no limit on how large the error can be, particularly when ``y_true``
    values are close to zero. In such cases the function returns a large value
    instead of ``inf``. While sMAPE is bounded at 2.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    the absolute percentage error at that time index,
    :math:`\left| \frac{y_i - \widehat{y}_i}{y_i} \right|`,
    or :math:`\frac{2|y_i - \widehat{y}_i|}{|y_i| + |\widehat{y}_i|}`,
    the symmetric version, if ``symmetric`` is True, for all time indices
    :math:`t_1, \dots, t_n` in the input.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric

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
    MeanAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianAbsolutePercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdape = MedianAbsolutePercentageError(symmetric=False)
    >>> mdape(y_true, y_pred)
    np.float64(0.16666666666666666)
    >>> smdape = MedianAbsolutePercentageError(symmetric=True)
    >>> smdape(y_true, y_pred)
    np.float64(0.18181818181818182)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdape(y_true, y_pred)
    np.float64(0.5714285714285714)
    >>> smdape(y_true, y_pred)
    np.float64(0.39999999999999997)
    >>> mdape = MedianAbsolutePercentageError(multioutput='raw_values', symmetric=False)
    >>> mdape(y_true, y_pred)
    array([0.14285714, 1.        ])
    >>> smdape = MedianAbsolutePercentageError(multioutput='raw_values', symmetric=True)
    >>> smdape(y_true, y_pred)
    array([0.13333333, 0.66666667])
    >>> mdape = MedianAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mdape(y_true, y_pred)
    np.float64(0.7428571428571428)
    >>> smdape = MedianAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=True)
    >>> smdape(y_true, y_pred)
    np.float64(0.5066666666666666)
    """  # noqa: E501

    func = median_absolute_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        by_index=False,
    ):
        self.symmetric = symmetric
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

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
        y_pred : time series in sktime compatible data container format
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

        numer_values = (y_true - y_pred).abs()

        if self.symmetric:
            denom_values = (y_true.abs() + y_pred.abs()) / 2
        else:
            denom_values = y_true.abs()

        raw_values = numer_values / denom_values

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return raw_values

            if multioutput == "uniform_average":
                return raw_values.median(axis=1)

        return raw_values.dot(multioutput)

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
        params2 = {"symmetric": True}
        return [params1, params2]
