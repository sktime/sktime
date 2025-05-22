#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    geometric_mean_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    geometric_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_asymmetric_error,
    mean_linex_error,
    mean_relative_absolute_error,
    mean_squared_error,
    mean_squared_percentage_error,
    mean_squared_scaled_error,
    median_absolute_error,
    median_absolute_percentage_error,
    median_absolute_scaled_error,
    median_relative_absolute_error,
    median_squared_error,
    median_squared_percentage_error,
    median_squared_scaled_error,
    relative_loss,
)


class GeometricMeanAbsoluteError(BaseForecastingErrorMetricFunc):
    """Geometric mean absolute error (GMAE).

    GMAE output is non-negative floating point. The best value is approximately
    zero, rather than zero.

    Like MAE and MdAE, GMAE is measured in the same units as the input data.
    Because GMAE takes the absolute value of the forecast error rather than
    squaring it, MAE penalizes large errors to a lesser degree than squared error
    variants like MSE, RMSE or GMSE or RGMSE.

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
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_squared_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of ``y_true`` and ``y_pred`` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when ``y_true`` equals ``y_pred``)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import GeometricMeanAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmae = GeometricMeanAbsoluteError()
    >>> gmae(y_true, y_pred)
    np.float64(0.000529527232030127)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmae(y_true, y_pred)
    np.float64(0.5000024031086919)
    >>> gmae = GeometricMeanAbsoluteError(multioutput='raw_values')
    >>> gmae(y_true, y_pred)
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmae = GeometricMeanAbsoluteError(multioutput=[0.3, 0.7])
    >>> gmae(y_true, y_pred)
    np.float64(0.7000014418652152)
    """

    func = geometric_mean_absolute_error


class GeometricMeanSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean squared error (GMSE) or Root geometric mean squared error (RGMSE).

    If ``square_root`` is False then calculates GMSE and if ``square_root`` is True
    then RGMSE is calculated. Both GMSE and RGMSE return non-negative floating
    point. The best value is approximately zero, rather than zero.

    Like MSE and MdSE, GMSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE and RdMSE. Because GMSE and RGMSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than GMAE.

    Parameters
    ----------
    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root geometric mean squared error (RGMSE)
        If False, returns geometric mean squared error (GMSE)

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
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_absolute_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of ``y_true`` and ``y_pred`` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when ``y_true`` equals ``y_pred``)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import GeometricMeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(2.80399089461488e-07)
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.000529527232030127)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5000000000115499)
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5000024031086919)
    >>> gmse = GeometricMeanSquaredError(multioutput='raw_values')
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    array([2.30997255e-11, 1.00000000e+00])
    >>> rgmse = GeometricMeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rgmse(y_true, y_pred)# doctest: +SKIP
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7])
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7000000000069299)
    >>> rgmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7000014418652152)
    """

    func = geometric_mean_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput, multilevel=multilevel, by_index=by_index
        )

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


class MeanAbsolutePercentageError(BaseForecastingErrorMetricFunc):
    r"""Mean absolute percentage error (MAPE) or symmetric MAPE.

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

    Both MAPE and sMAPE output non-negative floating point which is in fractional units
    rather than percentage. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    MAPE has no limit on how large the error can be, particulalrly when ``y_true``
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

    func = mean_absolute_percentage_error

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
        y_pred :time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
                index is equal to index of y_true
                entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
                index and columns equal to those of y_true
                i,j-th entry is metric at time i, at variable j
        """
        multioutput = self.multioutput
        symmetric = self.symmetric

        numer_values = (y_true - y_pred).abs()

        if symmetric:
            denom_values = (y_true.abs() + y_pred.abs()) / 2
        else:
            denom_values = y_true.abs()

        raw_values = numer_values / denom_values

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return raw_values

            if multioutput == "uniform_average":
                return raw_values.mean(axis=1)

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
        params2 = {"symmetric": True}
        return [params1, params2]


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

    MAPE has no limit on how large the error can be, particulalrly when ``y_true``
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


class MeanSquaredPercentageError(BaseForecastingErrorMetricFunc):
    """Mean squared percentage error (MSPE), or RMSPE, or symmetric MSPE, RMSPE.

    If ``square_root`` is False then calculates MSPE and if ``square_root`` is True
    then calculates root mean squared percentage error (RMSPE). If ``symmetric``
    is True then calculates sMSPE or sRMSPE. Output is non-negative floating
    point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    There is no limit on how large the error can be, particulalrly when ``y_true``
    values are close to zero. In such cases the function returns a large value
    instead of ``inf``.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
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
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MeanSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mspe = MeanSquaredPercentageError(symmetric=False)
    >>> mspe(y_true, y_pred)
    np.float64(0.23776218820861678)
    >>> smspe = MeanSquaredPercentageError(square_root=True, symmetric=False)
    >>> smspe(y_true, y_pred)
    np.float64(0.48760864246710883)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mspe(y_true, y_pred)
    np.float64(0.5080309901738473)
    >>> smspe(y_true, y_pred)
    np.float64(0.7026794936195895)
    >>> mspe = MeanSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mspe(y_true, y_pred)
    array([0.34013605, 0.67592593])
    >>> smspe = MeanSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    array([0.58321184, 0.82214714])
    >>> mspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mspe(y_true, y_pred)
    np.float64(0.5751889644746787)
    >>> smspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    np.float64(0.7504665536595034)
    """

    func = mean_squared_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        square_root=False,
        by_index=False,
    ):
        self.symmetric = symmetric
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

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
        params2 = {"symmetric": True, "square_root": True}
        return [params1, params2]


class MedianSquaredPercentageError(BaseForecastingErrorMetricFunc):
    """Median squared percentage error (MdSPE), or RMdSPE, or symmetric MdSPE, RMDsPE.

    If ``square_root`` is False then calculates MdSPE and if ``square_root`` is True
    then calculates root median squared percentage error (RMdSPE). If ``symmetric``
    is True then calculates sMdSPE or sRMdSPE. Output is non-negative floating
    point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    RMdSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when ``y_true``
    values are close to zero. In such cases the function returns a large value
    instead of ``inf``.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
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
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MeanSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MedianSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mdspe = MedianSquaredPercentageError(symmetric=False)
    >>> mdspe(y_true, y_pred)
    np.float64(0.027777777777777776)
    >>> smdspe = MedianSquaredPercentageError(square_root=True, symmetric=False)
    >>> smdspe(y_true, y_pred)
    np.float64(0.16666666666666666)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mdspe(y_true, y_pred)
    np.float64(0.5102040816326531)
    >>> smdspe(y_true, y_pred)
    np.float64(0.5714285714285714)
    >>> mdspe = MedianSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mdspe(y_true, y_pred)
    array([0.02040816, 1.        ])
    >>> smdspe = MedianSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smdspe(y_true, y_pred)
    array([0.14285714, 1.        ])
    >>> mdspe = MedianSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mdspe(y_true, y_pred)
    np.float64(0.7061224489795918)
    >>> smdspe = MedianSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smdspe(y_true, y_pred)
    np.float64(0.7428571428571428)
    """

    func = median_squared_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        square_root=False,
        by_index=False,
    ):
        self.symmetric = symmetric
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

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
        params2 = {"symmetric": True, "square_root": True}
        return [params1, params2]


class MeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Mean relative absolute error (MRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MRAE applies mean absolute error (MAE) to the resulting relative errors.

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
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae = MeanRelativeAbsoluteError()
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.9511111111111111)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8703703703703702)
    >>> mrae = MeanRelativeAbsoluteError(multioutput='raw_values')
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.51851852, 1.22222222])
    >>> mrae = MeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(1.0111111111111108)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = mean_relative_absolute_error


class MedianRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Median relative absolute error (MdRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MdRAE applies medan absolute error (MdAE) to the resulting relative errors.

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
    MeanRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae = MedianRelativeAbsoluteError()
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(1.0)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.6944444444444443)
    >>> mdrae = MedianRelativeAbsoluteError(multioutput='raw_values')
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.55555556, 0.83333333])
    >>> mdrae = MedianRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.7499999999999999)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = median_relative_absolute_error


class GeometricMeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Geometric mean relative absolute error (GMRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRAE applies geometric mean absolute error (GMAE) to the resulting relative
    errors.

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
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae = GeometricMeanRelativeAbsoluteError()
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.0007839273064064755)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.5578632807409556)
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput='raw_values')
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.97801163e-06, 1.11572158e+00])
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.7810066018326863)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = geometric_mean_relative_absolute_error


class GeometricMeanRelativeSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean relative squared error (GMRSE).

    If ``square_root`` is False then calculates GMRSE and if ``square_root`` is True
    then calculates root geometric mean relative squared error (RGMRSE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRSE applies geometric mean squared error (GMSE) to the resulting relative
    errors. RGMRSE applies root geometric mean squared error (RGMSE) to the
    resulting relative errors.

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
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse = GeometricMeanRelativeSquaredError()
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.0008303544925949156)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.622419372049448)
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput='raw_values')
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.09227746e-06, 1.24483465e+00])
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput=[0.3, 0.7])
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8713854839582426)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = geometric_mean_relative_squared_error

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


class MeanAsymmetricError(BaseForecastingErrorMetricFunc):
    """Calculate mean of asymmetric loss function.

    Output is non-negative floating point. The best value is 0.0.

    Error values that are less than the asymmetric threshold have
    ``left_error_function`` applied. Error values greater than or equal to
    asymmetric threshold  have ``right_error_function`` applied.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    Setting ``asymmetric_threshold`` to zero, ``left_error_function`` to 'squared'
    and ``right_error_function`` to 'absolute` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for ``left_error_function`` set to 'absolute' and ``right_error_function``
    set to 'squared`.

    The left_error_penalty and right_error_penalty can be used to add differing
    multiplicative penalties to over-predictions and under-predictions.

    Parameters
    ----------
    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have ``left_error_function``
        applied. Error values greater than or equal to asymmetric threshold
        have ``right_error_function`` applied.
    left_error_function : {'squared', 'absolute'}, default='squared'
        Loss penalty to apply to error values less than the asymmetric threshold.
    right_error_function : {'squared', 'absolute'}, default='absolute'
        Loss penalty to apply to error values greater than or equal to the
        asymmetric threshold.
    left_error_penalty : int or float, default=1.0
        An additional multiplicative penalty to apply to error values less than
        the asymmetric threshold.
    right_error_penalty : int or float, default=1.0
        An additional multiplicative penalty to apply to error values greater
        than the asymmetric threshold.

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
    mean_linex_error

    Notes
    -----
    Setting ``left_error_function`` and ``right_error_function`` to "absolute", but
    choosing different values for ``left_error_penalty`` and ``right_error_penalty``
    results in the "lin-lin" error function discussed in [2]_.

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [2] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
       Thomson, South-Western: Ohio, US.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAsymmetricError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> asymmetric_error = MeanAsymmetricError()
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5)
    >>> asymmetric_error = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.4625)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> asymmetric_error = MeanAsymmetricError()
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.75)
    >>> asymmetric_error = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7083333333333334)
    >>> asymmetric_error = MeanAsymmetricError(multioutput='raw_values')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    array([0.5, 1. ])
    >>> asymmetric_error = MeanAsymmetricError(multioutput=[0.3, 0.7])
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.85)
    """

    func = mean_asymmetric_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
        left_error_penalty=1.0,
        right_error_penalty=1.0,
        by_index=False,
    ):
        self.asymmetric_threshold = asymmetric_threshold
        self.left_error_function = left_error_function
        self.right_error_function = right_error_function
        self.left_error_penalty = left_error_penalty
        self.right_error_penalty = right_error_penalty

        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

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
        params2 = {
            "asymmetric_threshold": 0.1,
            "left_error_function": "absolute",
            "right_error_function": "squared",
            "left_error_penalty": 2.0,
            "right_error_penalty": 0.5,
        }
        return [params1, params2]


class MeanLinexError(BaseForecastingErrorMetricFunc):
    """Calculate mean linex error.

    Output is non-negative floating point. The best value is 0.0.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    The linex error function accounts for this by penalizing errors on one side
    of a threshold approximately linearly, while penalizing errors on the other
    side approximately exponentially. If ``a`` > 0 then negative errors
    (over-predictions) are penalized approximately linearly and positive errors
    (under-predictions) are penalized approximately exponentially. If ``a`` < 0
    the reverse is true.

    Parameters
    ----------
    a : int or float
        Controls whether over- or under- predictions receive an approximately
        linear or exponential penalty. If ``a`` > 0 then negative errors
        (over-predictions) are penalized approximately linearly and positive errors
        (under-predictions) are penalized approximately exponentially. If ``a`` < 0
        the reverse is true.
    b : int or float
        Multiplicative penalty to apply to calculated errors.

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
    mean_asymmetric_error

    Notes
    -----
    Calculated as b * (np.exp(a * error) - a * error - 1), where a != 0 and b > 0
    according to formula in [2]_.

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [1] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
       Thomson, South-Western: Ohio, US.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanLinexError
    >>> linex_error = MeanLinexError()
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.19802627763937575)
    >>> linex_error = MeanLinexError(b=2)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.3960525552787515)
    >>> linex_error = MeanLinexError(a=-1)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.2391800623225643)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> linex_error = MeanLinexError()
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.2700398392309829)
    >>> linex_error = MeanLinexError(a=-1)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.49660966225813563
    >>> linex_error = MeanLinexError(multioutput='raw_values')
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    array([0.17220024, 0.36787944])
    >>> linex_error = MeanLinexError(multioutput=[0.3, 0.7])
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.30917568000716666)
    """

    func = mean_linex_error

    def __init__(
        self,
        a=1.0,
        b=1.0,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        self.a = a
        self.b = b
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

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
        params2 = {"a": 0.5, "b": 2}
        return [params1, params2]


class RelativeLoss(BaseForecastingErrorMetricFunc):
    """Calculate relative loss of forecast versus benchmark forecast.

    Applies a forecasting performance metric to a set of forecasts and
    benchmark forecasts and reports ratio of the metric from the forecasts to
    the the metric from the benchmark forecasts. Relative loss output is
    non-negative floating point. The best value is 0.0.

    If the score of the benchmark predictions for a given loss function is zero
    then a large value is returned.

    This function allows the calculation of scale-free relative loss metrics.
    Unlike mean absolute scaled error (MASE) the function calculates the
    scale-free metric relative to a defined loss function on a benchmark
    method instead of the in-sample training data. Like MASE, metrics created
    using this function can be used to compare forecast methods on a single
    series and also to compare forecast accuracy between series.

    This is useful when a scale-free comparison is beneficial but the training
    data used to generate some (or all) predictions is unknown such as when
    comparing the loss of 3rd party forecasts or surveys of professional
    forecasters.

    Only metrics that do not require y_train are currently supported.

    Parameters
    ----------
    relative_loss_function : function
        Function to use in calculation relative loss.

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

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import RelativeLoss
    >>> from sktime.performance_metrics.forecasting import mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8148148148148147)
    >>> relative_mse = RelativeLoss(relative_loss_function=mean_squared_error)
    >>> relative_mse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.5178095088655261)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8490566037735847)
    >>> relative_mae = RelativeLoss(multioutput='raw_values')
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.625     , 1.03448276])
    >>> relative_mae = RelativeLoss(multioutput=[0.3, 0.7])
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.927272727272727)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = relative_loss

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        relative_loss_function=mean_absolute_error,
        by_index=False,
    ):
        self.relative_loss_function = relative_loss_function
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"relative_loss_function": mean_squared_error}
        return [params1, params2]
