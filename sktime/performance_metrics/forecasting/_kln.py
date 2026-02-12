#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""KL-divergence based metric assuming normal errors (KL-N)."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class KLDivergenceNormal(BaseForecastingErrorMetric):
    r"""KL-divergence based forecast error assuming normal errors (KL-N).

    KL-N uses the Kullback-Leibler divergence between the actual and predicted
    distributions, assuming normally distributed forecast errors scaled by
    a rolling variance of the true values. Output is non-negative floating
    point, lower is better, with 0.0 indicating a perfect forecast.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns

    .. math::
        \text{KL-N} = \sqrt{
        \frac{1}{n}\sum_{i=1}^{n}
        \frac{(y_i - \widehat{y}_i)^2}{S_i^2}
        }

    where

    .. math::
        S_i^2 = \frac{1}{i-1}\sum_{k=1}^{i-1}(y_k - \bar{y}_{i-1})^2
        \quad (i \ge 2)

    is the rolling sample variance of the first :math:`i-1` true values,
    and :math:`\bar{y}_{i-1}` is their mean.

    :math:`S_1^2` is undefined (no prior observations) and :math:`S_2^2`
    can be zero (single prior observation); both are clamped to ``eps``.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns jackknife pseudo-values of KL-N,
    at each time index :math:`t_i`, computed as
    :math:`n \cdot \text{KL-N} - (n-1) \cdot \text{KL-N}_{-i}`,
    where :math:`\text{KL-N}_{-i}` removes the i-th error contribution
    from the aggregate while keeping the rolling variances :math:`S_i^2`
    fixed (since they depend causally on prior observations).

    Parameters
    ----------
    window : int or None, default=None
        Number of prior observations used to estimate the rolling variance.

        * If ``None`` (default), an expanding window is used: all prior
          observations :math:`y_1, \dots, y_{i-1}` contribute to
          :math:`S_i^2`.
        * If ``int``, a fixed-length rolling window of the given size is
          used.  For example, ``window=5`` reproduces the **KL-N1** variant
          and ``window=10`` reproduces the **KL-N2** variant of
          Chen & Yang (2004).

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
    KLDivergenceDoubleExponential
    NormalizedMeanSquaredError

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import KLDivergenceNormal
    >>> y_true = np.array([3.0, 5.0, 2.0, 7.0, 4.0, 6.0])
    >>> y_pred = np.array([3.0, 5.0, 3.0, 6.0, 5.0, 5.5])
    >>> kln = KLDivergenceNormal()
    >>> kln(y_true, y_pred)
    np.float64(0.5771341616115743)
    """

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
        eps=None,
        window=None,
    ):
        self.eps = eps
        self.window = window
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _compute_rolling_var(self, y_true_vals, eps):
        """Compute rolling variance S_i^2 for each time index.

        Parameters
        ----------
        y_true_vals : np.ndarray, shape (n,) or (n, p)
            True values as numpy array.
        eps : float
            Epsilon for clamping.

        Returns
        -------
        s_sq : np.ndarray, same shape as y_true_vals
            Rolling variance, clamped to at least eps.
        """
        n = y_true_vals.shape[0]
        window = self.window

        if window is None:
            # Vectorized expanding window: Var(X) = E[X^2] - E[X]^2
            cumsum = np.cumsum(y_true_vals, axis=0)
            cumsqsum = np.cumsum(y_true_vals**2, axis=0)
            counts = np.arange(1, n + 1, dtype=np.float64)
            if y_true_vals.ndim > 1:
                counts = counts.reshape(-1, 1)
            rolling_var = cumsqsum / counts - (cumsum / counts) ** 2

            s_sq = np.zeros_like(y_true_vals, dtype=np.float64)
            s_sq[1:] = np.maximum(rolling_var[: n - 1], 0.0)
        else:
            s_sq = np.zeros_like(y_true_vals, dtype=np.float64)
            for i in range(1, n):
                past = y_true_vals[max(0, i - window) : i]
                mean_past = past.mean(axis=0)
                s_sq[i] = ((past - mean_past) ** 2).mean(axis=0)

        return np.maximum(s_sq, eps)

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

        y_true_vals = y_true.values
        s_sq = self._compute_rolling_var(y_true_vals, eps)

        sqe = (y_true.values - y_pred.values) ** 2
        scaled_sqe = sqe / s_sq

        scaled_sqe_df = pd.DataFrame(
            scaled_sqe, index=y_true.index, columns=y_true.columns
        )
        scaled_sqe_df = self._get_weighted_df(scaled_sqe_df, **kwargs)

        result = scaled_sqe_df.mean().pow(0.5)

        return self._handle_multioutput(result, multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from
        evaluate_by_index

        Uses jackknife pseudo-values since KL-N is sqrt(mean(...)),
        not mean(sqrt(...)). The leave-one-out metric removes the error
        contribution of each observation from the aggregate but keeps the
        rolling variance S_i^2 fixed, since S_i depends causally on prior
        true values and recomputing it without an interior observation
        would break the temporal ordering semantics.

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
        y_true_vals = y_true.values
        s_sq = self._compute_rolling_var(y_true_vals, eps)

        sqe = (y_true.values - y_pred.values) ** 2
        scaled_sqe = sqe / s_sq

        scaled_sqe_df = pd.DataFrame(
            scaled_sqe, index=y_true.index, columns=y_true.columns
        )
        scaled_sqe_df = self._get_weighted_df(scaled_sqe_df, **kwargs)

        full_mean = scaled_sqe_df.mean(axis=0)
        full_val = full_mean.pow(0.5)

        total = scaled_sqe_df.sum(axis=0)
        jack_mean = (total - scaled_sqe_df) / (n - 1)
        jack_val = jack_mean.pow(0.5)

        pseudo_values = n * full_val - (n - 1) * jack_val

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
        params3 = {"window": 5}
        return [params1, params2, params3]
