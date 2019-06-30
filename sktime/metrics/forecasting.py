import numpy as np
from sklearn.utils.validation import check_consistent_length

__author__ = ['Markus Loning']

# for reference implementations, see https://github.com/M4Competition/M4-methods/blob/master/ML_benchmarks.py


def mase_loss(y_true, y_pred, y_train, sp=1):
    """Mean absolute scaled error

    Parameters
    ----------
    y_true : array-like of shape = (fh) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : array-like of shape = (fh)
        Estimated target values.
    y_train : array-like of shape = (n_samples)
        Observed training values.
    sp : int
        Seasonal periodicity of training data.

    Returns
    -------
    loss : float
        MASE loss

    References
    ----------
    ..[1]   Hyndman, R. J. (2006). "Another look at measures of forecast accuracy", Foresight, Issue 4.
    """

    check_consistent_length(y_true, y_pred)

    #  naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean absolute error of naive seasonal prediction
    mae_naive = np.mean(np.abs(y_train[sp:] - y_pred_naive))

    return np.mean(np.abs(y_true - y_pred)) / mae_naive


def smape_loss(y_true, y_pred):
    """Symmetric mean absolute percentage error

    Parameters
    ----------
    y_true : array-like of shape = (fh) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : array-like of shape = (fh)
        Estimated target values.

    Returns
    -------
    loss : float
        SMAPE loss
    """
    check_consistent_length(y_true, y_pred)

    nominator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 2 * np.mean(nominator / denominator)
