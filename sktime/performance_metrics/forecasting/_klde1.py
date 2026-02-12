#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""KL-divergence based metric assuming single-exponential errors (KL-DE1)."""

__author__ = ["michaelellis003"]

import numpy as np

from sktime.performance_metrics.forecasting._klde_base import (
    _KLDivergenceLaplaceBase,
)


class KLDivergenceSingleExponential(_KLDivergenceLaplaceBase):
    r"""KL-divergence based forecast error assuming single-exponential errors (KL-DE1).

    KL-DE1 uses the Kullback-Leibler divergence between the actual and
    predicted distributions, assuming double-exponential (Laplace) distributed
    forecast errors scaled by the rolling standard deviation of the true
    values. Output is non-negative floating point, lower is better,
    with 0.0 indicating a perfect forecast.

    KL-DE1 differs from KL-DE2 only in the scale estimator:
    KL-DE1 uses the standard deviation (square root of variance), while
    KL-DE2 uses the mean absolute deviation (MAD).

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    ``evaluate`` or call returns

    .. math::
        \text{KL-DE1} = \frac{1}{n}\sum_{i=1}^{n}
        \left[
        \exp\!\left(-\frac{|e_i|}{\hat\sigma_i}\right)
        + \frac{|e_i|}{\hat\sigma_i} - 1
        \right]

    where :math:`e_i = y_i - \widehat{y}_i` and

    .. math::
        \hat\sigma_i = \sqrt{
        \frac{1}{i-1}\sum_{j=1}^{i-1}(y_j - \bar{y}_{i-1})^2
        }
        \quad (i \ge 2)

    is the rolling standard deviation of the first :math:`i-1` true values.

    :math:`\hat\sigma_1` is undefined (no prior observations); it is clamped
    to ``eps``.

    Since KL-DE1 is a simple mean of per-index terms, ``evaluate_by_index``
    returns the per-index KL-DE1 terms directly.

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    Parameters
    ----------
    window : int or None, default=None
        Number of prior observations used to estimate the rolling standard
        deviation.

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
    KLDivergenceDoubleExponential
    KLDivergenceNormal

    References
    ----------
    Chen, Z. and Yang, Y. (2004). "Assessing Forecast Accuracy Measures",
    Preprint 2004-10, Iowa State University.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import (
    ...     KLDivergenceSingleExponential,
    ... )
    >>> y_true = np.array([3.0, 5.0, 2.0, 7.0, 4.0, 6.0])
    >>> y_pred = np.array([3.0, 5.0, 3.0, 6.0, 5.0, 5.5])
    >>> klde1 = KLDivergenceSingleExponential()
    >>> klde1(y_true, y_pred)
    np.float64(0.1285730069501481)
    """

    _tags = {
        "authors": ["michaelellis003"],
    }

    def _compute_rolling_scale(self, y_true_vals, eps):
        """Compute rolling standard deviation for each time index.

        Parameters
        ----------
        y_true_vals : np.ndarray, shape (n,) or (n, p)
            True values as numpy array.
        eps : float
            Epsilon for clamping.

        Returns
        -------
        sigma : np.ndarray, same shape as y_true_vals
            Rolling standard deviation, clamped to at least eps.
        """
        n = y_true_vals.shape[0]
        window = self.window

        if window is None:
            # Vectorized expanding window using population variance:
            # std = sqrt(E[X^2] - E[X]^2)
            # The paper defines sigma_i via (1/(i-1)) sum_{k=1}^{i-1}(...)
            # which divides by the number of terms (population variance).
            cumsum = np.cumsum(y_true_vals, axis=0)
            cumsqsum = np.cumsum(y_true_vals**2, axis=0)
            counts = np.arange(1, n + 1, dtype=np.float64)
            if y_true_vals.ndim > 1:
                counts = counts.reshape(-1, 1)
            rolling_var = cumsqsum / counts - (cumsum / counts) ** 2

            sigma = np.zeros_like(y_true_vals, dtype=np.float64)
            sigma[1:] = np.sqrt(np.maximum(rolling_var[: n - 1], 0.0))
        else:
            sigma = np.zeros_like(y_true_vals, dtype=np.float64)
            for i in range(1, n):
                past = y_true_vals[max(0, i - window) : i]
                mean_past = past.mean(axis=0)
                sigma[i] = np.sqrt(((past - mean_past) ** 2).mean(axis=0))

        return np.maximum(sigma, eps)
