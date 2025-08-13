"""Common utilities for forecasting metrics."""

import numpy as np
import pandas as pd


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


def _pseudovalues_sqrt(scaled: pd.DataFrame):
    r"""Jackknife pseudo-values for square-root-based metrics (e.g., RMSE-like).

    This function computes jackknife pseudo-values for square-root-based metrics
    (e.g., RMSSE or RMSE) given a DataFrame of scaled squared errors.

    For a DataFrame ``scaled`` of per-timepoint scaled squared errors, the pseudo-values
    are computed as:

    .. math::
        \text{pseudo} = n \cdot \text{full} - (n - 1) \cdot \text{loo}

    where:

    * :math:`\text{full} = \sqrt{\text{mean\_over\_time}(\text{scaled})}`
      (Series of shape (d,))
    * :math:`\text{loo} =
        \sqrt{\dfrac{\text{sum\_over\_time}(\text{scaled}) - \text{scaled}}{n - 1}}`
      (DataFrame of shape ``(h, d)``)

    Parameters
    ----------
    scaled : pd.DataFrame
        DataFrame of shape (h, d) containing per-timepoint scaled squared errors.
        Rows correspond to forecast timepoints, columns to series/outputs.

    Returns
    -------
    pseudo : pd.DataFrame
        DataFrame of same shape as ``scaled`` containing jackknife pseudo-values
        for the square-root metric.

    Notes
    -----
    - If ``scaled`` has fewer than 2 rows (n <= 1), jackknife is undefined.
      In this case, the function returns a DataFrame filled with the ``full``
      value repeated on every row (so averaging the pseudo-values reproduces ``full``).
    """
    if not isinstance(scaled, pd.DataFrame):
        scaled = pd.DataFrame(scaled)

    n = scaled.shape[0]
    mse_full = scaled.mean(axis=0)
    full_rms = mse_full.pow(0.5)

    if n <= 1:
        return pd.DataFrame(
            np.tile(full_rms.values, (n, 1)),
            index=scaled.index,
            columns=scaled.columns,
        )

    sum_scaled = scaled.sum(axis=0)

    msse_loo = (sum_scaled - scaled) / (n - 1)

    rmsse_loo = msse_loo.pow(0.5)

    pseudo = pd.DataFrame(
        n * full_rms - (n - 1) * rmsse_loo,
        index=scaled.index,
        columns=scaled.columns,
    )

    return pseudo
