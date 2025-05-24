#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import median_squared_error


class MedianSquaredError(BaseForecastingErrorMetricFunc):
    r"""Median squared error (MdSE) or root median squared error (RMdSE).

    If ``square_root`` is False then calculates MdSE and if ``square_root`` is True
    then RMdSE is calculated. Both MdSE and RMdSE return non-negative floating
    point. The best value is 0.0.

    The Median Squared Error (MdSE) is calculated as:

    .. math::
        \\text{MdSE} = \text{median}\left((y_{i} - \\widehat{y}_{i})^2\right)_{i=1}^{n}

    where:
    - \\( y_i \\) are the true values,
    - \\( \\hat{y}_i \\) are the predicted values,
    - \\( n \\) is the number of observations.

    If ``square_root`` is set to True,
    the Root Median Squared Error Percentage (RMdSE) is computed:

    .. math::
        \\text{RMdSE} = \\sqrt{ \\text{MdSE} }

    Like MSE, MdSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data, like RMSE.

    Taking the median instead of the mean of the squared errors makes
    this metric more robust to error outliers relative to a meean based metric
    since the median tends to be a more robust measure of central tendency in
    the presence of outliers.

    Because the median commutes with monotonic transformations, MdSE and RMdSE
    do not penalize large errors more than the MdAE.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`:

    * if ``square_root`` is False, the squared error at that time index,
        :math:`(y_i - \widehat{y}_i)^2`,
    * if ``square_root`` is True, the absolute error at that time index,
      the absolute error at that time index, :math:`|y_i - \widehat{y}_i|`,

    for all time indices :math:`t_1, \dots, t_n` in the input.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to take the square root of the metric

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
    MedianAbsoluteError
    MeanSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdse = MedianSquaredError()
    >>> mdse(y_true, y_pred)
    np.float64(0.25)
    >>> rmdse = MedianSquaredError(square_root=True)
    >>> rmdse(y_true, y_pred)
    np.float64(0.5)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdse(y_true, y_pred)
    np.float64(0.625)
    >>> rmdse(y_true, y_pred)
    np.float64(0.75)
    >>> mdse = MedianSquaredError(multioutput='raw_values')
    >>> mdse(y_true, y_pred)
    array([0.25, 1.  ])
    >>> rmdse = MedianSquaredError(multioutput='raw_values', square_root=True)
    >>> rmdse(y_true, y_pred)
    array([0.5, 1. ])
    >>> mdse = MedianSquaredError(multioutput=[0.3, 0.7])
    >>> mdse(y_true, y_pred)
    np.float64(0.7749999999999999)
    >>> rmdse = MedianSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rmdse(y_true, y_pred)
    np.float64(0.85)
    """

    func = median_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
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

        raw_values = (y_true - y_pred) ** 2

        if self.square_root:
            raw_values = np.sqrt(raw_values)

        raw_values = self._get_weighted_df(raw_values, **kwargs)

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return raw_values

            if multioutput == "uniform_average":
                return raw_values.median(axis=1)

        # else, we expect multioutput to be array-like
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
        params2 = {"square_root": True}
        return [params1, params2]
