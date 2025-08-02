"""Common utilities for forecasting metrics."""

import numpy as np


def _relative_error(y_true, y_pred, y_pred_benchmark, eps=None):
    """Relative error for observations to benchmark method.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    Returns
    -------
    relative_error : float
        relative error

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
    """Percentage error.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    symmetric : bool, default = False
        Whether to calculate symmetric percentage error.

    relative_to : bool, default = "y_true"
        Whether to calculate percentage error by forecast.

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    Returns
    -------
    percentage_error : float

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
