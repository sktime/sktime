"""Common utilities for forecasting metrics."""

import numpy as np


def _relative_error(y_true, y_pred, y_pred_benchmark, eps=None):
    r"""Relative error with respect to benchmark predictions.

    For arrays ``y_true``, ``y_pred``, and ``y_pred_benchmark``,
    writing :math:`y` for ``y_true``, writing :math:`\widehat{y}` for ``y_pred``,
    and writing :math:`\widehat{y}_b` for ``y_pred_benchmark``,
    this function calculates the element-wise relative error,
    of :math:`\widehat{y}` with respect to the benchmark :math:`\widehat{y}_b`,
    defined as

    .. math::
        \frac{y - \widehat{y}}{\widehat{y} - \widehat{y}_b}

    where all operations are element-wise.

    The denominator is replaced by ``eps`` if it is smaller than ``eps``, entry-wise.

    Parameters
    ----------
    y_true : array-like of ground truth values
        Ground truth (correct) target values.

    y_pred : array-like of predicted values, must be same shape as y_true
        Predicted values.

    y_pred_benchmark : array-like of benchmark predictions, must be same shape as y_true
        Benchmark predictions to compare against.

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    Returns
    -------
    relative_error : np.ndarray, same shape as y_true and y_pred
        The element-wise relative error.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of \
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """
    if eps is None:
        eps = np.finfo(np.float64).eps

    denominator = np.where(
        y_true - y_pred_benchmark >= 0,
        np.maximum((y_true - y_pred_benchmark), eps),
        np.minimum((y_true - y_pred_benchmark), -eps),
    )
    return (y_true - y_pred) / denominator


def _percentage_error(y_true, y_pred, symmetric=False, relative_to="y_true", eps=None):
    r"""Percentage error.

    For arrays ``y_true`` and ``y_pred``,
    writing :math:`y` for ``y_true`` and :math:`\widehat{y}` for ``y_pred``,
    this function calculates the element-wise percentage error, defined as

    * :math:`\frac{|y - \widehat{y}|}{|y|}` if ``symmetric`` is ``False``,
      and ``relative_to`` is ``'y_true'``,
    * :math:`\frac{|y - \widehat{y}|}{|\widehat{y}|}` if ``symmetric`` is ``False``,
      and ``relative_to`` is ``'y_pred'``,
    * :math:`2 \frac{|y - \widehat{y}|}{|y| + |\widehat{y}|}`
      if ``symmetric`` is ``True``,

    where all operations are element-wise.

    All denominators are replaced by ``eps`` if they are smaller than ``eps``,
    entry-wise.

    Parameters
    ----------
    y_true : array-like of ground truth values
        Ground truth (correct) target values.

    y_pred : array-like of predicted values, must be same shape as y_true
        Predicted values.

    symmetric : bool, default = False
        Whether to calculate symmetric percentage error.

    relative_to : {'y_true', 'y_pred'}, default='y_true'
        Determines the denominator of the percentage error.

        * If ``'y_true'``, the denominator is the true values,
        * If ``'y_pred'``, the denominator is the predicted values.

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    Returns
    -------
    percentage_error : np.ndarray, same shape as y_true and y_pred
        The element-wise percentage error.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of \
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """
    if eps is None:
        eps = np.finfo(np.float64).eps

    if symmetric:
        denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), eps) / 2
    elif relative_to == "y_pred":
        denominator = np.maximum(np.abs(y_pred), eps)
    else:
        denominator = np.maximum(np.abs(y_true), eps)
    percentage_error = np.abs(y_true - y_pred) / denominator
    return percentage_error


def _fraction(enumerator, denominator, eps=None):
    r"""Element-wise fraction of enumerator to denominator.

    This function calculates the element-wise ratio of two arrays, ``enumerator``
    and ``denominator``, while ensuring numerical stability by replacing small
    denominator values with a specified epsilon.

    .. math::
        \text{Fraction}(x, y) = \frac{x}{y}

    where :math:`x` is the enumerator and :math:`y` is the denominator.

    The denominator is replaced by ``eps`` if it is smaller than ``eps``, entry-wise

    Parameters
    ----------
    enumerator : array-like
        Numerator values for the fraction calculation.

    denominator : array-like, must be the same shape as enumerator
        Denominator values for the fraction calculation.

    eps : float, default=None
        Numerical epsilon used to avoid division by zero or instability.
        Absolute values of the denominator smaller than ``eps`` are replaced
        by ``eps``. If None, defaults to ``np.finfo(np.float64).eps``.

    Returns
    -------
    fraction : np.ndarray, same shape as enumerator and denominator
        The element-wise fraction of enumerator to denominator, with small
        denominator values replaced by ``eps`` for numerical stability.
    """
    if eps is None:
        eps = np.finfo(np.float64).eps

    safe_denominator = np.where(
        denominator >= 0,
        np.maximum(denominator, eps),
        np.minimum(denominator, -eps),
    )
    return enumerator / safe_denominator


def _pseudovalues_sqrt(raw: np.ndarray):
    r"""Jackknife pseudo-values for square-root based metrics (e.g., RMSE/RMSSE).

    This function computes jackknife pseudo-values for square-root-based metrics
    (e.g., RMSSE or RMSE) given an array ``raw`` of per-timepoint scaled squared errors.

    Let :math:`S` denote the input array ``raw`` with entries :math:`S_{i j}`,
    where :math:`i = 1,\dots,n` indexes timepoints (rows) and
    :math:`j = 1,\dots,d` indexes series/outputs (columns). Define

    .. math::
        \overline{S}_{\cdot j}
        = \frac{1}{n}\sum_{i=1}^{n} S_{i j}
        \quad\text{(full-sample mean for column $j$),}

    and for the leave-one-out (LOO) mean at row :math:`i`:

    .. math::
        \overline{S}_{( - i), j}
        = \frac{1}{n-1}\sum_{\substack{k=1 \\ k \ne i}}^{n} S_{k j}
        \;=\; \frac{\sum_{k=1}^n S_{k j} - S_{i j}}{\,n-1\,}.

    Then define the full and leave-one-out root-mean values as

    .. math::
        \text{full}_j &= \sqrt{\overline{S}_{\cdot j}}
                        \quad\text{(Series of shape (d,))},\\[4pt]
        \text{loo}_{i j} &= \sqrt{\overline{S}_{(-i), j}}
                         \quad\text{(Array of shape (n, d)).}

    The jackknife pseudo-values are computed element-wise as

    .. math::
        \text{pseudo}_{i j} = n\cdot\text{full}_{j} - (n-1)\cdot\text{loo}_{i j},

    where :math:`n` is the number of rows (timepoints).

    Parameters
    ----------
    raw : np.ndarray
        1-D or 2-D numeric array containing per-timepoint scaled squared errors.
        Accepted shapes:
        - ``(n,)`` for univariate series (will be treated as shape ``(n, 1)``),
        - ``(n, d)`` for multivariate series.

    Returns
    -------
    pseudo : np.ndarray
        Array of the same shape as ``raw`` (or squeezed to 1-D if input was 1-D)
        containing jackknife pseudo-values computed as above.

    Notes
    -----
    - Reductions are *nan-safe*: the implementation uses ``np.nanmean``/``np.nansum``
      so ``NaN`` entries in ``raw`` are ignored in the means/sums.
    - Numerical safety: small negative values that may arise from round-off are
      clamped to zero before taking square roots.
    - If ``n <= 1`` (fewer than 2 rows), jackknife is undefined; the function
      returns the ``full`` value repeated for every row (so averaging the
      pseudo-values reproduces ``full``).
    - The arithmetic mean of the returned pseudo-values equals the jackknife
      estimator for the square-root statistic; for nonlinear statistics (like
      square-root) this jackknife estimator may differ from the plain
      full-sample square-root.
    """
    raw = np.asarray(raw, dtype=float)

    # normalize to 2D for unified processing
    squeezed = False
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
        squeezed = True
    elif raw.ndim != 2:
        raise ValueError("raw must be a 1D or 2D array")

    n, d = raw.shape

    # Use nan-safe reductions so NaNs (if any) are ignored in mean/sum
    mse_full = np.nanmean(raw, axis=0)  # shape (d,)
    full_rms = np.sqrt(np.maximum(mse_full, 0.0))

    if n <= 1:
        # jackknife undefined for n <= 1: repeat full value for each row
        out = np.tile(full_rms.reshape(1, -1), (n, 1))
        return out.squeeze() if squeezed else out

    sum_raw = np.nansum(raw, axis=0)  # shape (d,)

    # Leave-one-out mean squared error: broadcasting yields (n, d)
    msse_loo = (sum_raw - raw) / (n - 1)

    # Guard against tiny negative values caused by numerical noise
    msse_loo = np.maximum(msse_loo, 0.0)

    # Leave-one-out RMS (n, d)
    rmsse_loo = np.sqrt(msse_loo)

    # Jackknife pseudo-values: (n, d)
    pseudo = (n * full_rms.reshape(1, -1)) - ((n - 1) * rmsse_loo)

    return pseudo.squeeze() if squeezed else pseudo
