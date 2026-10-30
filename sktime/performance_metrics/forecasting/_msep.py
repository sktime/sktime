#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc


class MeanSquaredErrorPercentage(BaseForecastingErrorMetricFunc):
    r"""Mean Squared Error Percentage (MSE%) and root-MSE% forecasting error metrics.

    Calculates the mean squared error percentage between the true and predicted values.
    Optionally, the root mean squared error percentage (RMSE%) can be computed by
    setting ``square_root=True``.

    The Mean Squared Error Percentage (MSE%) is calculated as:

    .. math::
        \text{MSE\%} = \frac{\frac{1}{n}\sum_{i=1}^n (y_i - \widehat{y}_i)^2}
                            {\left|\frac{1}{n}\sum_{i=1}^n y_i\right|}

    where :math:`y_i` are the true values, :math:`\widehat{y}_i` are the
    predicted values, and :math:`n` is the number of observations.

    If ``square_root`` is set to True,
    the Root Mean Squared Error Percentage (RMSE%) is computed,
    the root mean squared error as a fraction of the absolute mean of the true values:

    .. math::
        \text{RMSE\%} = \frac{\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \widehat{y}_i)^2}}
                             {\left|\frac{1}{n}\sum_{i=1}^n y_i\right|}

    RMSE% is unit-free, i.e., rescaling the data does not change its value.

    Parameters
    ----------
    square_root : bool, default = False
        Whether to compute RMSE% instead of MSE%, i.e., take the square root of the
        mean squared error before dividing by the absolute mean of the true values

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

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import (
    ...     MeanSquaredErrorPercentage,
    ... )
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.0, 2.0, 4.0])
    >>> msep = MeanSquaredErrorPercentage()
    >>> round(float(msep(y_true, y_pred)), 3)
    0.167

    RMSE% is unit-free, so the same forecast in different units gives the same value:

    >>> rmsep = MeanSquaredErrorPercentage(square_root=True)
    >>> round(float(rmsep(y_true, y_pred)), 3)
    0.289
    >>> round(float(rmsep(100 * y_true, 100 * y_pred)), 3)
    0.289
    """

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
        self.multioutput = multioutput

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
              if `multioutput="raw_values"``
              i-th entry is the, metric calculated for i-th variable
        """
        multioutput = self.multioutput
        raw_values = (y_true - y_pred) ** 2
        raw_values = self._get_weighted_df(raw_values, **kwargs)
        num = raw_values.mean()
        denom = y_true.mean().abs()

        # RMSE%: the square root applies to the MSE only, keeping the metric unit-free
        if self.square_root:
            num = num.pow(0.5)

        msqe = num / denom

        return self._handle_multioutput(msqe, multioutput)

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

        raw_values_mse = (y_true - y_pred) ** 2
        raw_values_p = y_true

        # what we need to do is efficiently
        # compute msqe but using data with the i-th time point removed
        # msqe[i] = msqe(all data minus i-th time point)

        n = raw_values_mse.shape[0]

        num_mean = raw_values_mse.mean()
        denom_mean = raw_values_p.mean()

        num_jk = num_mean * (1 + 1 / (n - 1)) - raw_values_mse / (n - 1)
        denom_jk = denom_mean * (1 + 1 / (n - 1)) - raw_values_p / (n - 1)

        # RMSE%: the square root applies to the MSE only, consistent with _evaluate;
        # leave-one-out MSEs can be slightly negative due to floating point rounding
        if self.square_root:
            num_mean = num_mean.pow(0.5)
            num_jk = num_jk.clip(lower=0).pow(0.5)

        # absolute value of the mean, consistent with _evaluate
        msep_jk = num_jk / denom_jk.abs()
        msep = num_mean / denom_mean.abs()

        pseudo_values = n * msep - (n - 1) * msep_jk
        pseudo_values = self._get_weighted_df(pseudo_values, **kwargs)

        return self._handle_multioutput(pseudo_values, multioutput)

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
