#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mean absolute error percentage (MAE%) forecasting error metric."""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc


class MeanAbsoluteErrorPercentage(BaseForecastingErrorMetricFunc):
    r"""Mean Absolute Error Percentage (MAE%).

    Mean absolute error, expressed as a fraction of the absolute mean of the true
    values. Output is non-negative floating point, lower is better,
    and the lowest possible value is 0.0.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`
    (in :math:`\mathbb{R}`), at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns the Mean Absolute Error Percentage,

    .. math::
        \text{MAE\%} = \frac{\frac{1}{n}\sum_{i=1}^n |y_i - \widehat{y}_i|}
                            {\left|\frac{1}{n}\sum_{i=1}^n y_i\right|}

    (the time indices are not used).

    MAE% is a fraction of averages. It is not the same as the
    mean absolute percentage error, ``MeanAbsolutePercentageError`` (MAPE),
    which is an average of fractions,
    :math:`\frac{1}{n}\sum_{i=1}^n \frac{|y_i - \widehat{y}_i|}{|y_i|}`.
    Unlike MAPE, MAE% does not become undefined or explode if individual true values
    are zero or close to zero; it is undefined only if the mean of the true values
    is zero.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    the jackknife pseudo-value of MAE% at that time index,
    :math:`n \cdot \text{MAE\%} - (n - 1) \cdot \text{MAE\%}_{-i}`,
    where :math:`\text{MAE\%}_{-i}` is MAE% computed with the
    :math:`i`-th time index removed, for all time indices
    :math:`t_1, \dots, t_n` in the input.

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
    MeanAbsolutePercentageError
    MeanSquaredErrorPercentage

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import (
    ...     MeanAbsoluteErrorPercentage,
    ...     MeanAbsolutePercentageError,
    ... )
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> maep = MeanAbsoluteErrorPercentage()
    >>> round(float(maep(y_true, y_pred)), 4)
    0.2037

    MAE% is a fraction of averages, so it differs from MAPE,
    which is an average of fractions:

    >>> mape = MeanAbsolutePercentageError()
    >>> round(float(mape(y_true, y_pred)), 4)
    0.3369
    """

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs.

        private _evaluate containing core logic, called from evaluate

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
        loss : float or np.ndarray
            Calculated metric, possibly averaged by variable given ``multioutput``.

            * float if ``multioutput="uniform_average" or array-like,
              Value is metric averaged over variables and levels (see class docstring)
            * ``np.ndarray`` of shape ``(y_true.columns,)``
              if ``multioutput="raw_values"``,
              i-th entry is the metric calculated for i-th variable
        """
        abs_err = (y_true - y_pred).abs()
        abs_err = self._get_weighted_df(abs_err, **kwargs)

        maep = abs_err.mean() / y_true.mean().abs()

        return self._handle_multioutput(maep, self.multioutput)

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
            Calculated metric, by time point (jackknife pseudo-values).

            * pd.Series if self.multioutput="uniform_average" or array-like;
              index is equal to index of y_true;
              entry at index i is metric at time i, averaged over variables.
            * pd.DataFrame if self.multioutput="raw_values";
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j.
        """
        abs_err = (y_true - y_pred).abs()

        n = abs_err.shape[0]

        num_mean = abs_err.mean()
        denom_mean = y_true.mean()

        # leave-one-out means, i.e., means with the i-th time point removed
        num_jk = num_mean * (1 + 1 / (n - 1)) - abs_err / (n - 1)
        denom_jk = denom_mean * (1 + 1 / (n - 1)) - y_true / (n - 1)

        maep = num_mean / denom_mean.abs()
        maep_jk = num_jk / denom_jk.abs()

        pseudo_values = n * maep - (n - 1) * maep_jk
        pseudo_values = self._get_weighted_df(pseudo_values, **kwargs)

        return self._handle_multioutput(pseudo_values, self.multioutput)

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
        params2 = {"multioutput": "raw_values"}
        return [params1, params2]
