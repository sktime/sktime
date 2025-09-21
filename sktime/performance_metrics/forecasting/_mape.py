#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.performance_metrics.forecasting._common import _percentage_error


class MeanAbsolutePercentageError(BaseForecastingErrorMetric):
    r"""Mean absolute percentage error (MAPE) or symmetric MAPE.

    Both MAPE and sMAPE are non-negative floating point,
    is in fractional units relative to a specified denominator.
    Lower is better, and the lowest possible value is 0.0.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns the Mean Absolute Percentage Error,
    :math:`\frac{1}{n} \sum_{i=1}^n \left|\frac{y_i-\widehat{y}_i}{y_i} \right|`.
    (the time indices are not used)

    if ``symmetric`` is True then calculates
    symmetric mean absolute percentage error (sMAPE), defined as
    :math:`\frac{2}{n} \sum_{i=1}^n \frac{|y_i - \widehat{y}_i|}
    {|y_i| + |\widehat{y}_i|}`.

    To avoid division by zero, any denominator above is replaced by ``eps``
    if it is smaller than ``eps``; the value of ``eps`` defaults to
    ``np.finfo(np.float64).eps`` if not specified.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

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

    relative_to : {"y_true", "y_pred"}, default="y_true"
        Determines the denominator of the percentage error.

        * If ``"y_true"``, the denominator is the true values,
        * If ``"y_pred"``, the denominator is the predicted values.

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
    MedianAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mape = MeanAbsolutePercentageError(symmetric=False)
    >>> mape(y_true, y_pred)
    np.float64(0.33690476190476193)
    >>> smape = MeanAbsolutePercentageError(symmetric=True)
    >>> smape(y_true, y_pred)
    np.float64(0.5553379953379953)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mape(y_true, y_pred)
    np.float64(0.5515873015873016)
    >>> smape(y_true, y_pred)
    np.float64(0.6080808080808081)
    >>> mape = MeanAbsolutePercentageError(multioutput='raw_values', symmetric=False)
    >>> mape(y_true, y_pred)
    array([0.38095238, 0.72222222])
    >>> smape = MeanAbsolutePercentageError(multioutput='raw_values', symmetric=True)
    >>> smape(y_true, y_pred)
    array([0.71111111, 0.50505051])
    >>> mape = MeanAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mape(y_true, y_pred)
    np.float64(0.6198412698412699)
    >>> smape = MeanAbsolutePercentageError(multioutput=[0.3, 0.7], symmetric=True)
    >>> smape(y_true, y_pred)
    np.float64(0.5668686868686869)
    """

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        by_index=False,
        relative_to="y_true",
        eps=None,
    ):
        self.symmetric = symmetric
        self.relative_to = relative_to
        self.eps = eps
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
        y_true : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_pred : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Predicted values to evaluate.
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
        multioutput = self.multioutput
        symmetric = self.symmetric

        raw_values = _percentage_error(
            y_true=y_true,
            y_pred=y_pred,
            symmetric=symmetric,
            relative_to=self.relative_to,
            eps=self.eps,
        )
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
        params2 = {"symmetric": True}
        params3 = {"relative_to": "y_pred"}
        return [params1, params2, params3]
