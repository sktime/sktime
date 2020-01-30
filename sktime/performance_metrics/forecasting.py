import numpy as np
from sktime.utils.validation.forecasting import check_consistent_time_index, validate_time_index, validate_y

__author__ = ['Markus Löning']
__all__ = ["mase_score", "smape_score"]

# for reference implementations, see https://github.com/M4Competition/M4-methods/blob/master/ML_benchmarks.py


def mase_score(y_test, y_pred, y_train, sp=1):
    """Negative mean absolute scaled error

    Parameters
    ----------
    y_test : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.
    y_train : pandas Series of shape = (n_obs,)
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

    # input checks
    y_test = validate_y(y_test)
    y_pred = validate_y(y_pred)
    y_train = validate_y(y_train)
    check_consistent_time_index(y_test, y_pred, y_train=y_train)

    #  naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean absolute error of naive seasonal prediction
    mae_naive = np.mean(np.abs(y_train[sp:] - y_pred_naive))

    return -np.mean(np.abs(y_test - y_pred)) / mae_naive


def smape_score(y_true, y_pred):
    """Negative symmetric mean absolute percentage error

    Parameters
    ----------
    y_true : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.

    Returns
    -------
    loss : float
        SMAPE loss
    """
    check_consistent_time_index(y_true, y_pred)

    nominator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return -2 * np.mean(nominator / denominator)
