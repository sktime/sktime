import numpy as np
from sklearn.utils.validation import check_random_state


def rand_intervals_rand_n(x, random_state=None):
    """
    Computes a random number of overlapping intervals from index (x) with
    random starting points and lengths.

    :param x : array_like, shape = [n_observations]
    :param random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :return : array-like, shape = [n, 2]
        2d array containing start and end points of intervals


    References
    ----------
    .. [1] Deng, Houtao, et al. "A time series forest for classification and feature extraction."
    Information Sciences 239 (2013): 142-153.


    See also
    --------
    rand_intervals_fixed_n
    """

    rng = check_random_state(random_state)

    starts = []
    ends = []
    m = x.size  # series length

    W = rng.randint(1, m, size=int(np.sqrt(m)))
    for w in W:
        size = m - w + 1
        start = rng.randint(size, size=int(np.sqrt(size)))
        starts.extend(start)
        for s in start:
            end = s + w
            ends.append(end)
    return np.column_stack([starts, ends])


def rand_intervals_fixed_n(x, n=None, random_state=None):
    """
    Computes a fixed number (n) of overlapping intervals from index (x) with
    random starting points and lengths.

    :param x : array_like, shape = [n_observations]
        Array containing the time-series index.
    :param n : None or int
        Number of random intervals to compute.

        - If n, n random intervals are generated.
        - If None, int(sqrt(m)) intervals are generated where m is the length of the time-series.
    :param random_state : int, RandomState instance or None, optional (default=None)

        - If int, random_state is the seed used by the random number generator;
        - If RandomState instance, random_state is the random number generator;
        - If None, the random number generator is the RandomState instance used by `np.random`.
    :return : array-like, shape = [n, 2]
        2d array containing start and end points of intervals


    See also
    --------
    rand_intervals_rand_n
    """

    rng = check_random_state(random_state)

    m = x.size + 1  # series length, plus one for half-open interval indexing in Python
    if n is None:
        n = int(np.sqrt(m))  # number of random intervals

    min_length = 1
    starts = rng.randint(m - min_length, size=n)
    ends = np.zeros(n, dtype=int)
    for i, start in enumerate(starts):
        length = rng.randint(min_length, m - start)
        end = start + length
        if end > m:
            end = m
        ends[i] = end

    return np.column_stack([starts, ends])


def time_series_slope(y):
    """
    Compute slope of time series (y) using ordinary least squares.

    :param y: array_like
        Time-series.
    :return: float
        Slope of time-series.
    """
    n = y.shape[0]
    if n < 2:
        return 0
    else:
        x = np.arange(n) + 1
        x_mu = x.mean()
        return (((x * y).mean() - x_mu * y.mean())
                / ((x ** 2).mean() - x_mu ** 2))
