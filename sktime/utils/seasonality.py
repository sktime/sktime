from statsmodels.tsa.stattools import acf
import numpy as np


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
    x : ndarray
        Time series
    freq : int
        Frequency, periods per year (ppy)

    Returns
    -------
    test : bool
        Whether or not seasonality is present in data for given frequency

    References
    ----------
    https://github.com/M4Competition/M4-methods/blob/master/ML_benchmarks.py

    """
    y = np.asarray(y)
    crit_val = 1.645
    n_timepoints = len(y)
    r = acf(y, nlags=sp, fft=False)
    s = r[1] + np.sum(r[2:] ** 2)
    limit = crit_val * np.sqrt((1 + 2 * s) / n_timepoints)
    return np.abs(r[sp]) > limit
