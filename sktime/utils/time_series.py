import numpy as np


def time_series_slope(y, axis=0):
    """Compute slope of time series (y) using ordinary least squares.

    Parameters
    ----------
    y : array_like
        Time-series.
    axis : int
        Axis along which the time-series slope is computed.

    Returns
    -------
    slope : float
        Slope of time-series.
    """

    n, m = np.atleast_2d(y).shape
    if m < 2:
        return np.zeros(n) if n > 1 else 0
    else:
        x = np.arange(m)
        x_mean = (m - 1) / 2  # x.mean()
        return (np.mean(x * y, axis=axis) - x_mean * np.mean(y, axis=axis)) / (np.mean(x ** 2) - x_mean ** 2)


