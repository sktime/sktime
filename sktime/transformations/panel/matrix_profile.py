# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.panel import check_X


def _sliding_dot_products(q, t, q_len, t_len):
    """
    Computes the sliding dot products between a query and a time series.

    Parameters
    ----------
    q: numpy.array
        Query.
    t: numpy.array
        Time series.
    q_len: int
        Length of the query.
    t_len: int
        Length of the time series.

    Returns
    -------
    dot_prod: numpy.array
                Sliding dot products between q and t.
    """

    # Reversing query and padding both query and time series
    t_padded = np.pad(t, (0, t_len))
    q_reversed = np.flipud(q)
    q_reversed_padded = np.pad(q_reversed, (0, 2 * t_len - q_len))

    # Applying FFT to both query and time series
    t_fft = np.fft.fft(t_padded)
    q_fft = np.fft.fft(q_reversed_padded)

    # Applying inverse FFT to obtain the convolution of the time series by
    # the query
    element_wise_mult = np.multiply(t_fft, q_fft)
    inverse_fft = np.fft.ifft(element_wise_mult)

    # Returns only the valid dot products from inverse_fft
    dot_prod = inverse_fft[q_len - 1 : t_len].real

    return dot_prod


def _calculate_distance_profile(
    dot_prod, q_mean, q_std, t_mean, t_std, q_len, n_t_subs
):
    """
    Calculates the distance profile for the given query.

    Parameters
    ----------
        dot_prod: numpy.array
            Sliding dot products between the time series and the query.
        q_mean: float
            Mean of the elements of the query.
        q_std: float
            Standard deviation of elements of the query.
        t_mean: numpy.array
            Array with the mean of the elements from each subsequence of
            length(query) from the time series.
        t_std: numpy.array
            Array with the standard deviation of the elements from each
            subsequence of length(query) from the time series.
        q_len: int
            Length of the query.
        n_t_subs: int
            Number of subsequences in the time series.

    Output
    ------
        d: numpy.array
            Distance profile of query q.
    """

    d = [
        2
        * q_len
        * (
            1
            - ((dot_prod[i] - q_len * q_mean * t_mean[i]) / (q_len * q_std * t_std[i]))
        )
        for i in range(0, n_t_subs)
    ]
    d = np.absolute(d)
    d = np.sqrt(d)

    return d


def _minimum_distance(mp, ip, dp, i, m, dp_len):
    """
    Finds the minimum distance in the distance profile, considering the
    exclusion zone.

    Parameters
    ----------
        mp: numpy.array
            Matrix profile.
        ip: numpy.array
            Index profile.
        dp: numpy.array
            Distance profile.
        i: int
            Index of the element to be compared from the matrix profile.
        m: int
            Length of the subsequences.
        dp_len: int
            Length of the distance profile.

    Output
    ------
        mp: numpy.array
            Array with the distance between every subsequence and its
            nearest neighbor from the same time series.
        ip: numpy.array
            Array with the indexes of the nearest neighbors of each
            subsequence.
    """

    # Initialization
    min_value = float("inf")
    min_index = -1

    for k in range(0, dp_len):
        if dp[k] < min_value and (k < i - m / 2 or k > i + m / 2):
            min_value = dp[k]
            min_index = k
    mp[i] = min_value
    ip[i] = min_index

    return mp, ip


def _stomp_self(ts, m):
    """
    STOMP implementation for self-similarity join.

    Parameters
    ----------
        ts: numpy.array
            Time series.
        m: int
            Length of the subsequences.

    Output
    ------
        mp: numpy.array
            Array with the distance between every subsequence from ts1
            to the nearest subsequence with same length from ts2.
        ip: numpy.array
            Array with the index of the nearest neighbor of ts1 in ts2.
    """

    ts = ts.flatten()

    ts_len = ts.shape[0]

    # Number of subsequences
    n_subs = ts_len - m + 1

    # Compute the mean and standard deviation
    ts_mean = [np.mean(ts[i : i + m]) for i in range(0, n_subs)]
    ts_std = [np.std(ts[i : i + m]) for i in range(0, n_subs)]

    # Compute the dot products between the first subsequence and every other
    # subsequence
    dot_prod = _sliding_dot_products(ts[0:m], ts, m, ts_len)
    first_dot_prod = np.copy(dot_prod)

    # Initialization
    mp = np.full(n_subs, float("inf"))  # matrix profile
    ip = np.zeros(n_subs)  # index profile

    # Compute the distance profile for the first subsequence
    dp = _calculate_distance_profile(
        dot_prod, ts_mean[0], ts_std[0], ts_mean, ts_std, m, n_subs
    )

    # Updates the matrix profile
    mp, ip = _minimum_distance(mp, ip, dp, 0, m, n_subs)

    for i in range(1, n_subs):
        for j in range(n_subs - 1, 0, -1):
            dot_prod[j] = (
                dot_prod[j - 1] - ts[j - 1] * ts[i - 1] + ts[j - 1 + m] * ts[i - 1 + m]
            )  # compute the next dot products
            # using the previous ones
        dot_prod[0] = first_dot_prod[i]
        dp = _calculate_distance_profile(
            dot_prod, ts_mean[i], ts_std[i], ts_mean, ts_std, m, n_subs
        )
        mp, ip = _minimum_distance(mp, ip, dp, i, m, n_subs)

    return mp


class MatrixProfile(_PanelToTabularTransformer):
    """
    Takes as input a time series dataset and returns the matrix profile and
    index profile for each time series of the dataset.

    Example of use:
    # Xt = MatrixProfile(m).transform(X)
    X, a pandas DataFrame, is the the dataset.
    m, an integer, is the desired subsequence length to be used.
    Xt is the transformed X, i.e., a pandas DataFrame with the same number
    of rows as X, but each row has the matrix profile for the
    corresponding time series.
    """

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(self, m=10):
        self.m = m  # subsequence length
        super(MatrixProfile, self).__init__()

    def transform(self, X, y=None):
        """
        Takes as input a time series dataset and returns the matrix profile
        for each single time series of the dataset.

        Parameters
        ----------
        X: pandas.DataFrame
           Time series dataset.

        Returns
        -------
        Xt: pandas.DataFrame
            Dataframe with the same number of rows as the input.
            The number of columns equals the number of subsequences
            of the desired length in each time series.
        """
        # Input checks
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        n_instances = X.shape[0]
        Xt = pd.DataFrame([_stomp_self(X[i], self.m) for i in range(n_instances)])
        return Xt
