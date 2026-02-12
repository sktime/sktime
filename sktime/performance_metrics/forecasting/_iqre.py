#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interquartile range error (IQR) metric."""

__author__ = ["michaelellis003"]

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class InterQuartileRangeError(BaseForecastingErrorMetric):
    r"""Interquartile range error (IQR).

    Normalizes the root mean squared error (RMSE) by the interquartile range
    (IQR) of the true values, making it location-scale invariant. Output is
    non-negative floating point, lower is better, with 0.0 indicating
    a perfect forecast.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns

    .. math::
        \text{IQR} = \frac{
        \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \widehat{y}_i)^2}
        }{Q_3(y) - Q_1(y)}

    where :math:`Q_1(y)` and :math:`Q_3(y)` are the 25th and 75th percentiles
    of the true values.

    If the interquartile range is zero (e.g. more than half the values are
    identical), it is clamped to ``eps`` to avoid division by zero.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns jackknife pseudo-values of the IQR error,
    at each time index :math:`t_i`, computed as
    :math:`n \cdot \text{IQR} - (n-1) \cdot \text{IQR}_{-i}`,
    where :math:`\text{IQR}_{-i}` is the metric with the i-th observation
    removed.

    Parameters
    ----------
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
    MeanSquaredError
    NormalizedMeanSquaredError

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import InterQuartileRangeError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> iqre = InterQuartileRangeError()
    >>> iqre(y_true, y_pred)
    np.float64(0.6422616289332564)
    """

    _tags = {
        "authors": ["michaelellis003"],
    }

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
        eps=None,
    ):
        self.eps = eps
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
        y_true : pandas.DataFrame
            Ground truth (correct) target values.

        y_pred : pandas.DataFrame
            Predicted values to evaluate.

        Returns
        -------
        loss : float or np.ndarray
            Calculated metric, possibly averaged by variable.
        """
        multioutput = self.multioutput

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        raw_values = (y_true - y_pred) ** 2
        raw_values = self._get_weighted_df(raw_values, **kwargs)
        rmse = raw_values.mean().pow(0.5)

        q75 = y_true.quantile(0.75)
        q25 = y_true.quantile(0.25)
        iqr = np.maximum(q75 - q25, eps)

        result = rmse / iqr

        return self._handle_multioutput(result, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from
        evaluate_by_index

        Uses jackknife pseudo-values since IQR error is not a simple mean
        of per-index terms.

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
        multioutput = self.multioutput

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        n = y_true.shape[0]
        sqe = (y_true - y_pred) ** 2
        sqe = self._get_weighted_df(sqe, **kwargs)

        mse = sqe.mean(axis=0)
        rmse = mse.pow(0.5)

        q75 = y_true.quantile(0.75)
        q25 = y_true.quantile(0.25)
        iqr = np.maximum(q75 - q25, eps)
        full_val = rmse / iqr

        # Jackknife leave-one-out: remove each squared error from the sum
        # but keep IQR fixed (recomputing quantiles per LOO subset would be
        # expensive and introduce discrete jumps that destabilize pseudo-values).
        sqe_sum = sqe.sum(axis=0)
        mse_jack = (sqe_sum - sqe) / (n - 1)
        rmse_jack = mse_jack.pow(0.5)
        val_jack = rmse_jack / iqr

        pseudo_values = n * full_val - (n - 1) * val_jack

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
        params2 = {"eps": 1e-6}
        return [params1, params2]
