__author__ = ["Markus Löning"]

from warnings import warn

import numpy as np
from statsmodels.tsa.stattools import acf
from sktime.utils.validation.forecasting import check_y, check_sp

def seasonality_test(y, sp):
    """Seasonality test used in M4 competition

    # original implementation
    # def seasonality_test(original_ts, ppy):
    #
    # Seasonality test
    # :param original_ts: time series
    # :param ppy: periods per year
    # :return: boolean value: whether the TS is seasonal
    #
    # s = acf(original_ts, 1)
    # for i in range(2, ppy):
    #     s = s + (acf(original_ts, i) ** 2)
    #
    # limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))
    #
    # return (abs(acf(original_ts, ppy))) > limit

    Parameters
    ----------
    y : pd.Series or np.array
        Time series
    sp : int
        Seasonal periodicity, periods per year (ppy)

    Returns
    -------
    is_seasonal : bool
        Whether or not seasonality is present in data for given frequency

    References
    ----------
    ..[1]  https://github.com/M4Competition/M4-methods/blob/master/ML_benchmarks.py
    """
    y = check_y(y)
    y = np.asarray(y)
    n_timepoints = len(y)

    sp = check_sp(sp)

    if n_timepoints < 3 * sp:
        warn("Did not perform seasonality test, as `y`` is too short for the given `sp`, returned: False")
        is_seasonal = False

    else:
        coefs = acf(y, nlags=sp)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
        limits = tcrit / np.sqrt(n_timepoints) * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
        limit = limits[sp - 1]  #  zero-based indexing
        is_seasonal = np.abs(coef) > limit

    return is_seasonal
