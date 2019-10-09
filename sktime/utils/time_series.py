import numpy as np
from sklearn.utils import check_array

from sktime.forecasters.model_selection import RollingWindowSplit
from sktime.utils.validation.forecasting import validate_fh, validate_time_index


def get_n_intervals(n_timepoints, n_intervals="sqrt"):
    """
    Compute number of intervals for given length of time series

    Parameters
    ----------
    n_timepoints : int
    n_intervals : {int, float, str, callable}

    Returns
    -------
    n_intervals : int
        Computed number of intervals
    """
    # check n_timepoints
    if not isinstance(n_timepoints, int):
        raise ValueError(f"n_timepoints must be an integer, but found: {type(n_timepoints)}")
    if not n_timepoints >= 1:
        raise ValueError(f"n_timepoints must be >= 1, but found: {n_timepoints}")

    # compute number of splits
    allowed_strings = ("sqrt", "log")

    # TODO add callables/functions as input args
    # integer
    if isinstance(n_intervals, int):

        if not n_intervals <= n_timepoints:
            raise ValueError(f"n_intervals must be smaller than n_timepoints, but found: "
                             f"n_intervals={n_intervals} and n_timepoints={n_timepoints}")

        if n_intervals < 1:
            raise ValueError(f"n_intervals must be >= 1, but found: {n_intervals}")

        n_intervals_ = n_intervals

    # function
    elif callable(n_intervals):
        n_intervals_ = n_intervals(n_timepoints)

    # string
    elif isinstance(n_intervals, str):
        if not n_intervals in allowed_strings:
            raise ValueError(f"n_intervals must be in {allowed_strings}, but found: {n_intervals}")

        str_func_map = {
            "sqrt": np.sqrt,
            "log": np.log
        }
        func = str_func_map[n_intervals]
        n_intervals_ = func(n_timepoints)

    # float
    elif isinstance(n_intervals, float):
        if not (0 < n_intervals <= 1):
            raise ValueError(f"n_intervals must be > 0 and <= 1, but found: {n_intervals}")

        n_intervals_ = n_intervals * n_timepoints

    else:
        raise ValueError(f'Number of intervals must be either one of the allowed string options in '
                         f'{{allowed_strings}}, an integer or a float number, but found: {n_intervals}')

    # make sure n_intervals is an integer and there is at least one interval
    n_intervals_ = np.maximum(1, np.int(n_intervals_))
    return n_intervals_


def time_series_slope(y):
    """
    Compute slope of time series (y) using ordinary least squares.

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
    y = np.asarray(y).ravel()
    len_series = len(y)

    if len_series < 2:
        return 0
    else:
        x = np.arange(len_series)  # time index
        x_mean = (len_series - 1) / 2  # faster than x.mean()
        return (np.mean(x * y) - x_mean * np.mean(y)) / (np.mean(x ** 2) - x_mean ** 2)


def fit_trend(x, order=0):
    """Fit linear regression with polynomial terms of given order

        x : array_like, shape=[n_samples, n_obs]
        Time series data, each sample is fitted separately
    order : int
        The polynomial order of the trend, zero is constant (mean), one is
        linear trend, two is quadratic trend, and so on.

    Returns
    -------
    coefs : ndarray, shape=[n_samples, order + 1]
        Fitted coefficients of polynomial order for each sample, one column means order zero, two columns mean order 1
        (linear), three columns mean order 2 (quadratic), etc

    See Also
    -------
    add_trend
    remove_trend
    """
    x = check_array(x)

    if order == 0:
        coefs = np.mean(x, axis=1).reshape(-1, 1)

    else:
        n_obs = x.shape[1]
        index = np.arange(n_obs)
        poly_terms = np.vander(index, N=order + 1)

        # linear least squares fitting using numpy's optimised routine, assuming samples in columns
        # coefs = np.linalg.pinv(poly_terms).dot(x.T).T
        coefs, _, _, _ = np.linalg.lstsq(poly_terms, x.T, rcond=None)

        # returning fitted coefficients in expected format with samples in rows
        coefs = coefs.T

    return coefs


def remove_trend(x, coefs, time_index=None):
    """Remove trend from an array with a trend of given order along axis 0 or 1

    Parameters
    ----------
    x : array_like, shape=[n_samples, n_obs]
        Time series data, each sample is de-trended separately
    coefs : ndarray, shape=[n_samples, order + 1]
        Fitted coefficients for each sample, single column means order zero, two columns mean order 1
        (linear), three columns mean order 2 (quadratic), etc
    time_index : array-like, shape=[n_obs], optional (default=None)
        Time series index for which to add the trend components

    Returns
    -------
    xt : ndarray
        The de-trended series is the residual of the linear regression of the
        data on the trend of given order.

    See Also
    --------
    fit_trend
    add_trend

    References
    ----------
    Adapted from statsmodels (0.9.0), see
    https://www.statsmodels.org/dev/_modules/statsmodels/tsa/tsatools.html#detrend
    """
    x = check_array(x)

    # infer order from shape of given coefficients
    order = coefs.shape[1] - 1

    # special case, remove mean
    if order == 0:
        xt = x - coefs
        return xt

    else:
        if time_index is None:
            # if no time index is given, create range index
            n_obs = x.shape[1]
            time_index = np.arange(n_obs)
        else:
            # validate given time index
            time_index = validate_time_index(time_index)
            if not len(time_index) == x.shape[1]:
                raise ValueError('Length of passed index does not match length of passed x')

        poly_terms = np.vander(time_index, N=order + 1)
        xt = x - np.dot(poly_terms, coefs.T).T

    return xt


def add_trend(x, coefs, time_index=None):
    """Add trend to array for given fitted coefficients along axis 0 or 1, inverse function to `remove_trend()`

    Parameters
    ----------
    x : array_like, shape=[n_samples, n_obs]
        Time series data, each sample is treated separately
    coefs : array-like, shape=[n_samples, order + 1]
        fitted coefficients of polynomial order for each sample, one column means order zero, two columns mean order 1
        (linear), three columns mean order 2 (quadratic), etc
    time_index : array-like, shape=[n_obs], optional (default=None)
        Time series index for which to add the trend components

    Returns
    -------
    xt : ndarray
        The series with added trend.

    See Also
    -------
    fit_trend
    remove_trend
    """
    x = check_array(x)

    #  infer order from shape of given coefficients
    order = coefs.shape[1] - 1

    # special case, add mean
    if order == 0:
        xt = x + coefs

    else:
        if time_index is None:
            n_obs = x.shape[1]
            time_index = np.arange(n_obs)

        else:
            # validate given time index
            time_index = validate_time_index(time_index)

            if not len(time_index) == x.shape[1]:
                raise ValueError('Length of passed index does not match length of passed x')

        poly_terms = np.vander(time_index, N=order + 1)
        xt = x + np.dot(poly_terms, coefs.T).T

    return xt


def split_into_tabular_train_test(x, window_length=None, fh=None, test_size=1):
    """Helper function to split single time series into tabular train and
    test sets using rolling window approach"""

    # validate forecasting horizon
    fh = validate_fh(fh)

    # get time series index
    index = np.arange(len(x))

    # set up rolling window iterator
    rw = RollingWindowSplit(window_length=window_length, fh=fh)

    # slice time series into windows
    xs = []
    ys = []
    for input, output in rw.split(index):
        xt = x[input]
        yt = x[output]
        xs.append(xt)
        ys.append(yt)

    # stack windows into tabular array
    x = np.array(xs)
    y = np.array(ys)

    # split into train and test set
    x_train = x[:-test_size, :]
    y_train = y[:-test_size, :]

    x_test = x[-test_size:, :]
    y_test = y[-test_size:, :]

    return x_train, y_train, x_test, y_test





