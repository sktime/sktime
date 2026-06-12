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
