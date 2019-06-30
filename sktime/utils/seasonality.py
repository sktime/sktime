from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np


def remove_seasonality(x, sp=1, model='additive'):
    """Remove seasonality from time series

    Parameters
    ----------
    x : ndarray
    sp : int, optional (default=1)
        Seasonal periodicity
    model : str {'additive', 'multiplicative'}, optional (default='additive')
        Model to use for estimating seasonal component

    Returns
    -------
    xt : ndarray
        Series with seasonality component removed
    si : ndarray
        Seasonal indicices, shape=[freq,]

    See Also
    --------
    add_seasonality
    """

    # TODO: extend to handle multiple series (rows) more efficiently, requires to change seasonality testing too,
    #  statsmodels decompose methods does handle multiple series (expected to be in columns of 2d array)

    x = np.asarray(x)

    if sp > 1 and seasonality_test(x, freq=sp):
        # adjust for seasonality
        si = seasonal_decompose(x, freq=sp, model=model, filt=None, two_sided=True,
                                extrapolate_trend=0).seasonal
        xt = x / si
        si = si[:sp]

    else:
        # no seasonality adjustment
        xt = x
        si = np.ones(sp)

    return xt, si


def add_seasonality(x, si):
    """Add seasonality given as seasonality indices to time series

    Parameters
    ----------
    x : ndarray
        Time series
    si : ndarray
        Seasonal indices

    Returns
    -------
    xt : ndarray
        Time series with added seasonality component

    See Also
    --------
    remove_seasonality
    """

    x = np.asarray(x)
    freq = len(si)

    if freq == 1:
        xt = x

    else:
        xt = x * np.resize(si, len(x))

    return xt


def seasonality_test(x, freq):
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
    x = np.asarray(x)
    crit_val = 1.645

    n = len(x)
    r = acf(x, nlags=freq)
    s = r[1] + np.sum(r[2:] ** 2)
    limit = crit_val * np.sqrt((1 + 2 * s) / n)

    return np.abs(r[freq]) > limit
