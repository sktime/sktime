#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Normalized mean squared error (NMSE) metric."""

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class NormalizedMeanSquaredError(BaseForecastingErrorMetric):
    r"""Normalized mean squared error (NMSE).

    NMSE normalizes the root mean squared error by the standard deviation
    of the true values, making it location-scale invariant. Output is
    non-negative floating point, lower is better, with 0.0 indicating
    a perfect forecast.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns

    .. math::
        \text{NMSE} = \sqrt{
        \frac{\sum_{i=1}^{n}(y_i - \widehat{y}_i)^2}
        {\sum_{i=1}^{n}(y_i - \bar{y})^2}
        }

    where :math:`\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i`.

    Note that the squared NMSE equals one minus the coefficient of
    determination: :math:`\text{NMSE}^2 = 1 - R^2`. A model no better than
    predicting the mean yields :math:`\text{NMSE} = 1`.

    If the variance of the true values is zero (constant series),
    the denominator is clamped to ``eps`` to avoid division by zero.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns jackknife pseudo-values of the NMSE,
    at each time index :math:`t_i`, computed as
    :math:`n \cdot \text{NMSE} - (n-1) \cdot \text{NMSE}_{-i}`,
    where :math:`\text{NMSE}_{-i}` is the NMSE with the i-th observation removed.

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
    InterQuartileRangeError

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import NormalizedMeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> nmse = NormalizedMeanSquaredError()
    >>> nmse(y_true, y_pred)
    np.float64(0.2630806138733395)
    """

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
        mse = raw_values.mean()

        y_var = ((y_true - y_true.mean()) ** 2).mean()
        y_var = np.maximum(y_var, eps)

        nmse = (mse / y_var).pow(0.5)

        return self._handle_multioutput(nmse, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from
        evaluate_by_index

        Uses jackknife pseudo-values since NMSE is not a simple mean
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
        y_var = ((y_true - y_true.mean()) ** 2).mean()
        y_var = np.maximum(y_var, eps)
        nmse = (mse / y_var).pow(0.5)

        sqe_sum = sqe.sum(axis=0)
        y_dev = (y_true - y_true.mean()) ** 2
        y_dev_sum = y_dev.sum(axis=0)

        mse_jack = (sqe_sum - sqe) / (n - 1)
        y_var_jack = (y_dev_sum - y_dev) / (n - 1)
        y_var_jack = np.maximum(y_var_jack, eps)
        nmse_jack = (mse_jack / y_var_jack).pow(0.5)

        pseudo_values = n * nmse - (n - 1) * nmse_jack

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
