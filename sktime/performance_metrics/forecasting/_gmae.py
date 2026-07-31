#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import pandas as pd
from scipy.stats import gmean

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc


class GeometricMeanAbsoluteError(BaseForecastingErrorMetricFunc):
    r"""Geometric mean absolute error (GMAE).

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n` (in :math:`mathbb{R}`),
    at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns the Geometric Mean Absolute Error,
    :math:`\left( \prod_{i=1}^n |y_i - \widehat{y}_i| \right)^{\frac{1}{n}}`.
    (the time indices are not used)

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    jackknife pseudo-samples of the GMAE at that time index,
    :math:`n * \bar{\varepsilon} - (n-1) * \varepsilon_i`,
    where :math:`\bar{\varepsilon}` is the GMAE over all time indices,
    and :math:`\varepsilon_i` is the GMAE with the i-th time index removed.

    GMAE output is non-negative floating point. The best value is approximately
    zero, rather than zero.

    Like MAE and MdAE, GMAE is measured in the same units as the input data.
    Because GMAE takes the absolute value of the forecast error rather than
    squaring it, MAE penalizes large errors to a lesser degree than squared error
    variants like MSE, RMSE or GMSE or RGMSE.

    Parameters
    ----------
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

    def _evaluate(self, y_true, y_pred, sample_weight=None, **kwargs):
        """Evaluate the Geometric Mean Absolute Error (GMAE) metric on given inputs.

        This private method contains core logic for computing the GMAE metric.
        By default, it uses `_evaluate_by_index` to compute the
        arithmetic mean over time points.

        Parameters
        ----------
        y_true : pd.Series or pd.DataFrame
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series scitype: pd.DataFrame
            Panel scitype: pd.DataFrame with 2-level row MultiIndex
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex

        y_pred : pd.Series or pd.DataFrame
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, possibly averaged by variable given ``multioutput``.

            * float if ``multioutput="uniform_average" or array-like,
              Value is metric averaged over variables and levels (see class docstring)
            * ``np.ndarray`` of shape ``(y_true.columns,)``
              if `multioutput="raw_values"``
              i-th entry is the, metric calculated for i-th variable
        """
        abs_err_np = (y_true - y_pred).abs().values.flatten()
        gmae = gmean(abs_err_np, axis=0, weights=sample_weight)
        gmae = pd.Series(gmae, index=y_true.columns)

        return self._handle_multioutput(gmae, self.multioutput)

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

        raw_values = (y_true - y_pred).abs()

        n = raw_values.shape[0]
        gmae = gmean(raw_values, axis=0)

        gmae_jackknife = (raw_values ** (-1 / n) * gmae) ** (1 + 1 / (n - 1))
        pseudo_values = n * gmae - (n - 1) * gmae_jackknife

        pseudo_values = self._get_weighted_df(pseudo_values, **kwargs)

        return self._handle_multioutput(pseudo_values, multioutput)
