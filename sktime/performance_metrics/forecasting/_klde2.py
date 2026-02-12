#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""KL-divergence based metric assuming double-exponential errors (KL-DE2)."""

import numpy as np

from sktime.performance_metrics.forecasting._klde_base import (
    _KLDivergenceLaplaceBase,
)


class KLDivergenceDoubleExponential(_KLDivergenceLaplaceBase):
    r"""KL-divergence based forecast error assuming double-exponential errors (KL-DE2).

    KL-DE2 uses the Kullback-Leibler divergence between the actual and
    predicted distributions, assuming double-exponential (Laplace) distributed
    forecast errors scaled by a rolling mean absolute deviation (MAD) of the
    true values. Output is non-negative floating point, lower is better,
    with 0.0 indicating a perfect forecast.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns

    .. math::
        \text{KL-DE2} = \frac{1}{n}\sum_{i=1}^{n}
        \left[
        \exp\!\left(-\frac{|e_i|}{\hat\sigma_i}\right)
        + \frac{|e_i|}{\hat\sigma_i} - 1
        \right]

    where :math:`e_i = y_i - \widehat{y}_i` and

    .. math::
        \hat\sigma_i = \frac{1}{i-1}\sum_{j=1}^{i-1}|y_j - \bar{y}_{i-1}|
        \quad (i \ge 2)

    is the rolling mean absolute deviation (MAD) of the first :math:`i-1`
    true values (same rolling MAD as in msMAPE).

    :math:`\hat\sigma_1` is undefined (no prior observations); it is clamped
    to ``eps``.

    Since KL-DE2 is a simple mean of per-index terms, ``evaluate_by_index``
    returns the per-index KL-DE2 terms directly.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    Parameters
    ----------
    window : int or None, default=None
        Number of prior observations used to estimate the rolling MAD.

        * If ``None`` (default), an expanding window is used: all prior
          observations :math:`y_1, \dots, y_{i-1}` contribute to
          :math:`\hat\sigma_i`.
        * If ``int``, a fixed-length rolling window of the given size is
          used.

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
    KLDivergenceSingleExponential
    KLDivergenceNormal
    MeanAbsolutePercentageErrorStabilized

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import (
    ...     KLDivergenceDoubleExponential,
    ... )
    >>> y_true = np.array([3.0, 5.0, 2.0, 7.0, 4.0, 6.0])
    >>> y_pred = np.array([3.0, 5.0, 3.0, 6.0, 5.0, 5.5])
    >>> klde2 = KLDivergenceDoubleExponential()
    >>> klde2(y_true, y_pred)
    np.float64(0.14407771573805175)
    """

    def _compute_rolling_scale(self, y_true_vals, eps):
        """Compute rolling mean absolute deviation (MAD) for each time index.

        Parameters
        ----------
        y_true_vals : np.ndarray, shape (n,) or (n, p)
            True values as numpy array.
        eps : float
            Epsilon for clamping.

        Returns
        -------
        sigma : np.ndarray, same shape as y_true_vals
            Rolling MAD, clamped to at least eps.
        """
        n = y_true_vals.shape[0]
        window = self.window

        sigma = np.zeros_like(y_true_vals, dtype=np.float64)
        if window is None:
            # Precompute cumulative means to avoid redundant slicing
            cumsum = np.cumsum(y_true_vals, axis=0)
            counts = np.arange(1, n + 1, dtype=np.float64)
            if y_true_vals.ndim > 1:
                counts = counts.reshape(-1, 1)
            rolling_mean = cumsum / counts

            for i in range(1, n):
                sigma[i] = np.abs(
                    y_true_vals[:i] - rolling_mean[i - 1]
                ).mean(axis=0)
        else:
            for i in range(1, n):
                past = y_true_vals[max(0, i - window) : i]
                mean_past = past.mean(axis=0)
                sigma[i] = np.abs(past - mean_past).mean(axis=0)

        return np.maximum(sigma, eps)
