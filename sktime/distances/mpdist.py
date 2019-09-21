import numpy as np
import pandas as pd

def sliding_dot_products(q, t, q_len, t_len):
    """

        Computes the sliding dot products between a query and a time series.

        Parameters
        ----------
            q: pandas.Series
                Query.
            t: pandas.Series
                Time series.
            q_len: int
                Length of the query.
            t_len: int
                Length of the time series.

        Output
        ------
            dot_prod: numpy.array
                        Sliding dot products between q and t.

    """

    # Reversing query and padding both query and time series
    t_padded = np.pad(t, (0, t_len))
    q_reversed = np.flipud(q)
    q_reversed_padded = np.pad(q_reversed, (0, 2*t_len-q_len))

    # Applying FFT to both query and time series
    t_fft = np.fft.fft(t_padded)
    q_fft = np.fft.fft(q_reversed_padded)

    # Applying inverse FFT to obtain the convolution of the time series by the query
    element_wise_mult = np.multiply(t_fft, q_fft)
    reverse_fft = np.fft.ifft(element_wise_mult)

    # Returns only the valid dot products from reverse_fft
    dot_prod = reverse_fft[q_len-1:t_len].real

    return dot_prod

def compute_mean_std(q, t, q_len, t_len, n_t_subs):
    """

        Computes the mean and standard deviation of the query and the time series.

        Parameters
        ----------
            q: pandas.Series
                Query.
            t: pandas.Series
                Time series.
            q_len: int
                Length of the query.
            t_len: int
                Length of the time series.
            n_t_subs: int
                Number of subsequences in the time series.

        Output
        ------
            q_mean: float
                    Mean of the elements from the query.
            q_std: float
                    Standard deviation of the elements from the query.
            t_mean: numpy.array
                    Array with the mean of the elements from each subsequence of length(query) from the time series.
            t_std: numpy.array
                    Array with the standard deviation of the elements from each subsequence of length(query) from the time series.

    """

    # Query
    q_mean = np.mean(q)
    q_std = np.std(q)

    # Time series
    t_mean = np.empty(n_t_subs)

    for i in range(0, n_t_subs):
        t_mean[i] = np.mean(t[i:i+q_len])

    t_std = np.empty(n_t_subs)

    for i in range(0, n_t_subs):
        t_std[i] = np.std(t[i:i+q_len])

    return q_mean, q_std, t_mean, t_std

def calculate_distance_profile(q, t, dot_prod, q_mean, q_std, t_mean, t_std, q_len, t_len, n_t_subs):
    """

        Calculates the distance profile for the given query.

        Parameters
        ----------
            q: pandas.Series
                Query.
            t: pandas.Series
                Time series.
            dot_prod: numpy.array
                Sliding dot products between the time series and the query.
            q_mean: float
                Mean of the query's elements.
            q_std: float
                Standard deviation of query's elements.
            t_mean: numpy.array
                Array with the mean of the elements from each subsequence of length(query) from the time series.
            t_std: numpy.array
                Array with the standard deviation of the elements from each subsequence of length(query) from the time series.
            q_len: int
                Length of the query.
            t_len: int
                Length of the time series.
            n_t_subs: int
                Number of subsequences in the time series.

        Output
        ------
            d: numpy.array
                Distance profile of query q.

    """

    d = np.zeros(n_t_subs)

    for i in range(0, n_t_subs):
        d[i] = np.sqrt(abs(2*q_len*(1-((dot_prod[i]-q_len*q_mean*t_mean[i])/(q_len*q_std*t_std[i])))))

    return d

def mass(q, t, q_len, t_len):
    """

        MASS algorithm (Mueen's ultra-fast Algorithm for Similarity Search)

        Parameters
        ----------
            q: pandas.Series
                Query.
            t: pandas.Series
                Time series.
            q_len: int
                Length of the query.
            t_len: int
                Length of the time series.

        Output
        ------
            d: numpy.array
                Distance profile of query q.

    """

    n_t_subs = t_len-q_len+1 # number of subsequences in the time series

    # Compute the dot products
    dot_prod = sliding_dot_products(q, t, q_len, t_len)

    # Compute the mean and standard deviation of the query and the time series
    q_mean, q_std, t_mean, t_std = compute_mean_std(q, t, q_len, t_len, n_t_subs)

    # Compute the distance profile
    d = calculate_distance_profile(q, t, dot_prod, q_mean, q_std, t_mean, t_std, q_len, t_len, n_t_subs)

    return d

def minimum_distance(mp, ip, dp, i):
    """

        Compares each element from the distance profile with the corresponding element from the matrix profile.

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

        Output
        ------
            mp: numpy.array
                Array with the distance between every subsequence from ts1 to the nearest subsequence with same length from ts2.
            ip: numpy.array
                Array with the index of the ts1's nearest neighbor in ts2.

    """

    for k in range(0, dp.size):
        if dp[k] < mp[i]:
            mp[i] = dp[k]
            ip[i] = k

    return mp, ip

def stamp(ts1, ts2, m, len1, len2):
    """

        STAMP implementation.

        Parameters
        ----------
            ts1: pandas.Series
                First time series.
            ts2: pandas.Series
                Second time series.
            m: int
                Length of the subsequences.
            len1: int
                First time series' length.
            len2: int
                Second time series' length.

        Output
        ------
            mp: numpy.array
                Array with the distance between every subsequence from ts1 to the nearest subsequence with same length from ts2.
            ip: numpy.array
                Array with the index of the ts1's nearest neighbor in ts2.

    """

    n_subs1 = len1-m+1 # number of subsequences in ts1

    mp = np.full(n_subs1, float('inf')) # matrix profile
    ip = np.zeros(n_subs1) # index profile

    for i in range(0, n_subs1):
        dp = mass(ts1[i:i+m], ts2, m, len2) # computes the distance profile
        mp, ip = minimum_distance(mp, ip, dp, i) # finds the minimum distance and updates the matrix and index profile

    return mp, ip

def mpdist(ts1, ts2, m):
    """
        MPDist implementation using STAMP to compute the Matrix Profile.

        Parameters
        ----------
            ts1: pandas.Series
                First time series.
            ts2: pandas.Series
                Second time series.
            m: int
                Length of the subsequences.

        Output
        ------
            mpdist: float
                Distance between the two time series.
    """

    len1 = ts1.size
    len2 = ts2.size

    threshold = 0.05
    mp, ip = stamp(ts1, ts2, m, len1, len2) # compute the matrix profile

    k = int(threshold * (len1 + len2))

    sorted_mp = np.sort(mp) # sort the matrix profile in ascending order

    if len(sorted_mp) > k:
        mpdist = sorted_mp[k]
    else:
        mpdist = sorted_mp[len(sorted_mp)-1]

    return mpdist
