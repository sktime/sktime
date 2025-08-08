"""Synthetic data generating functions."""

from typing import Union

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_random_state


def piecewise_normal_multivariate(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    variances: Union[npt.ArrayLike, float] = 1.0,
    covariances: npt.ArrayLike = None,
    random_state: Union[int, np.random.RandomState] = None,
) -> npt.ArrayLike:
    """Generate multivariate series from segments.

    Each segment has length specified in ``lengths`` and data sampled from a
    multivariate normal distribution with a mean from ``means`` and covariance
    from ``covariances`` (either specified or built from ``variances`` when
    unspecified)

    Parameters
    ----------
    lengths : array_like
        Lengths of the segments to be generated of shape (n_segments,)
    means : array_like
        Means of the segments to be generated, as an array of shape
        (n_segments, n_series)
    variances : float or array_like (default=1.0)
        Variance of the segments to be generated
    covariances : array_like (default=None)
        Covariances of segments to be generated of shape
        (n_segments, n_series, n_series)
        If None, this will be constructed from variances by assuming independence
        of random variables, i.e. variance as diagonal elements of covariance matrix
    random_state : int or np.random.RandomState
        Either a random seed or ``RandomState`` instance

    Returns
    -------
    data : array_like
        Multivariate time series as ``np.array`` of shape (sum(lengths), n_series)

    Examples
    --------
    >>> from sktime.detection.datagen import piecewise_normal_multivariate
    >>> piecewise_normal_multivariate(means=[[1, 1], [2, 2], [3, 3]],\
        lengths=[2, 3, 1], random_state=2)
    array([[ 0.58324215,  0.94373317],
           [-1.1361961 ,  2.64027081],
           [ 0.20656441,  1.15825263],
           [ 2.50288142,  0.75471191],
           [ 0.94204778,  1.09099239],
           [ 3.55145404,  5.29220801]])

    >>> from sktime.detection.datagen import piecewise_normal_multivariate
    >>> piecewise_normal_multivariate(means=[[1, 1], [2, 2], [3, 3]],\
        lengths=[2, 3, 1], variances=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],\
        random_state=2)
    array([[ 0.58324215,  0.94373317],
           [-1.1361961 ,  2.64027081],
           [ 0.20656441,  1.15825263],
           [ 2.50288142,  0.75471191],
           [ 0.94204778,  1.09099239],
           [ 3.55145404,  5.29220801]])

    >>> from sktime.detection.datagen import piecewise_normal_multivariate
    >>> piecewise_normal_multivariate(means=[[1, 1], [2, 2], [3, 3]],\
        lengths=[2, 3, 1], covariances=[[[1.0, 0], [0, 1.0]], [[1.0, 0],\
            [0, 1.0]], [[1.0, 0], [0, 1.0]]], random_state=2)
    array([[ 0.58324215,  0.94373317],
           [-1.1361961 ,  2.64027081],
           [ 0.20656441,  1.15825263],
           [ 2.50288142,  0.75471191],
           [ 0.94204778,  1.09099239],
           [ 3.55145404,  5.29220801]])

    >>> from sktime.detection.datagen import piecewise_normal_multivariate
    >>> piecewise_normal_multivariate(means=[[1, 3], [4, 5]], lengths=[3, 3],\
        covariances=[[[0.5, 0.3], [0.3, 1.0]], [[1.0, 0.3], [0.3, 0.7]]],\
        random_state=2)
    array([[ 0.78066776,  2.61125356],
           [ 0.92296736,  0.51689669],
           [-0.2694238 ,  1.47959507],
           [ 4.00389069,  3.95225998],
           [ 5.32264874,  5.05088075],
           [ 2.62479901,  6.08308546]])
    """

    def get_covariances(var):
        """Fill 1D variance array of length N to 2D covariance array of size (N,N)."""
        cov = np.zeros((N, N), float)
        np.fill_diagonal(cov, var)
        return cov

    L, N = np.array(means).shape

    rng = check_random_state(random_state)
    assert len(lengths) == L

    # if no covariance is specified, build it from variance
    # assuming independent random variables
    if covariances is None:
        assert variances is not None

        # variances van be specified as a float, make 1D array, repeat L times
        if isinstance(variances, (float, int)):
            variances = np.repeat(variances, N)
            variances = np.tile(variances, (L, 1))

        assert np.array(variances).shape == (L, N)

        # get covariance matrices from variance arrays
        covariances = [get_covariances(var) for var in variances]

    else:
        assert all(np.allclose(np.array(cov), np.array(cov).T) for cov in covariances)
        assert all(np.all(np.linalg.eigvals(cov) >= 0) for cov in covariances)

    assert np.array(covariances).shape[0] == L
    assert np.array(covariances).shape[1] == N

    return np.concatenate(
        [
            rng.multivariate_normal(mean=mean, cov=cov, size=length)
            for mean, cov, length in zip(means, covariances, lengths)
        ]
    )


def piecewise_normal(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    std_dev: Union[npt.ArrayLike, float] = 1.0,
    random_state: Union[int, np.random.RandomState] = None,
) -> npt.ArrayLike:
    """Generate series from segments.

    Each segment has length specified in ``lengths`` and data sampled from a normal
    distribution with a mean from ``means`` and standard deviation from ``std_dev``.

    Parameters
    ----------
    means : array_like
        Means of the segments to be generated
    lengths : array_like
        Lengths of the segments to be generated
    std_dev : float or array_like
        Standard deviations of the segments to be generated
    random_state : int or np.random.RandomState
        Either a random seed or RandomState instance

    Returns
    -------
    data : np.array
        univariate time series as np.array

    Examples
    --------
    >>> from sktime.detection.datagen import piecewise_normal
    >>> piecewise_normal([1, 2, 3], lengths=[2, 4, 8], random_state=42) # doctest: +SKIP
    array([1.49671415, 0.8617357 , 2.64768854, 3.52302986, 1.76584663,
        1.76586304, 4.57921282, 3.76743473, 2.53052561, 3.54256004,
        2.53658231, 2.53427025, 3.24196227, 1.08671976])

    >>> from sktime.detection.datagen import piecewise_normal
    >>> piecewise_normal([1, 2, 3], lengths=[2, 4, 8], std_dev=0) # doctest: +SKIP
    array([1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.])

    >>> from sktime.detection.datagen import piecewise_normal
    >>> piecewise_normal([1, 2, 3], lengths=[2, 4, 8], std_dev=[0, 0.5, 1.0])\
        # doctest: +SKIP
    array([1.        , 1.        , 2.32384427, 2.76151493, 1.88292331,
        1.88293152, 4.57921282, 3.76743473, 2.53052561, 3.54256004,
        2.53658231, 2.53427025, 3.24196227, 1.08671976])
    """
    rng = check_random_state(random_state)
    assert len(means) == len(lengths)

    if isinstance(std_dev, (float, int)):
        std_dev = np.repeat(std_dev, len(means))

    assert len(std_dev) == len(means)

    segments_data = [
        rng.normal(loc=mean, scale=sd, size=[length])
        for mean, length, sd in zip(means, lengths, std_dev)
    ]
    return np.concatenate(tuple(segments_data))


def piecewise_multinomial(
    n_trials: int,
    lengths: npt.ArrayLike,
    p_vals: npt.ArrayLike,
    random_state: Union[int, np.random.RandomState] = None,
) -> npt.ArrayLike:
    """Generate series from segments.

    Each segment has length specified in ``lengths`` and data sampled from a multinomial
    distribution with a total number of experiments for each trial set from ``n_trials``
    and the probability for each outcome stored inside a list contained in  ``p_vals``.

    Parameters
    ----------
    n_trials : int
        Number of experiments to run during each trial
    lengths : array_like
        Lengths of the segments to be generated
    p_vals : array_like
        Set of probabilities for each outcome for each distribution
        Each set of probabilities must be equal length
    random_state : int or np.random.RandomState
        Either a random seed or RandomState instance

    Returns
    -------
    data : np.array
        univariate or multivariate time series as np.array
        that has dimensions sum(lengths) x n_outcomes
        where n_outcomes = # of outcomes for each item in ``p_vals``

    Examples
    --------
    >>> from sktime.detection.datagen import piecewise_multinomial
    >>> piecewise_multinomial(20, lengths=[3, 2], p_vals=[[1/4, 3/4], \
        [3/4, 1/4]], random_state=42) # doctest: +SKIP
    array([[ 4, 16],
       [ 8, 12],
       [ 6, 14],
       [15,  5],
       [17,  3]])

    >>> from sktime.detection.datagen import piecewise_multinomial
    >>> piecewise_multinomial(10, lengths=[2, 4, 8], \
        p_vals=[[1, 0], [0, 1], [1, 0]]) # doctest: +SKIP
    array([[10,  0],
       [10,  0],
       [ 0, 10],
       [ 0, 10],
       [ 0, 10],
       [ 0, 10],
       [10,  0],
       [10,  0],
       [10,  0],
       [10,  0],
       [10,  0],
       [10,  0],
       [10,  0],
       [10,  0]])
    """
    rng = check_random_state(random_state)

    # error handling for inputs
    if len(lengths) != len(p_vals):
        raise ValueError("lengths and p_vals arguments must be same length")
    elif not all(sum(p_val) == 1 for p_val in p_vals):
        raise ValueError("each set of probabilities in p_val must sum to 1")
    elif not (np.array([len(p_val) for p_val in p_vals]) == len(p_vals[0])).all():
        raise ValueError("each set of probabilities in p_val must be equal length")

    segments_data = [
        rng.multinomial(n=n_trials, pvals=p_val, size=[length])
        for p_val, length in zip(p_vals, lengths)
    ]
    return np.concatenate(tuple(segments_data))


def piecewise_poisson(
    lambdas: npt.ArrayLike,
    lengths: npt.ArrayLike,
    random_state: Union[int, np.random.RandomState] = None,
) -> npt.ArrayLike:
    """Generate series using Possion distribution.

    Each segment has length specified in ``lengths`` and data sampled from a Poisson
    distribution with expected lambda from ``lambdas``.

    Parameters
    ----------
    lambdas : array_like
        Expected number and variance of events within a specified time interval
    lengths : array_like
        Lengths of the segments to be generated
    random_state : int or np.random.RandomState
        Either a random seed or RandomState instance

    Returns
    -------
    data : np.array
        univariate time series as np.array

    Examples
    --------
    >>> from sktime.detection.datagen import piecewise_poisson
    >>> piecewise_poisson(lambdas=[1,2,3],lengths=[2,4,8],random_state=42)#doctest:+SKIP
    array([1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 2, 4, 2, 1])

    >>> from sktime.detection.datagen import piecewise_poisson
    >>> piecewise_poisson(lambdas=[1,3,6],lengths=[2,4,8],random_state=42)#doctest:+SKIP
    array([1, 2, 1, 3, 3, 2, 5, 5, 6, 4, 4, 9, 3, 5])
    """
    rng = check_random_state(random_state)

    assert len(lambdas) == len(lengths)

    try:
        segments_data = [
            rng.poisson(lam=lams, size=[length])
            for lams, length in zip(lambdas, lengths)
        ]
    except ValueError:
        raise Exception("Size mismatch")

    return np.concatenate(tuple(segments_data))


def labels_with_repeats(means: npt.ArrayLike, std_dev: npt.ArrayLike) -> npt.ArrayLike:
    """Generate labels for unique combinations of means and std_dev."""
    data = [means, std_dev]
    unique, indices = np.unique(data, axis=1, return_inverse=True)
    labels = np.arange(unique.shape[1])
    return labels[indices]


def label_piecewise_normal(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    std_dev: Union[npt.ArrayLike, float] = 1.0,
    repeated_labels: bool = True,
) -> npt.ArrayLike:
    """Generate labels for a series composed of segments.

    Parameters
    ----------
    means : array_like
        Means of the segments to be generated
    lengths : array_like
        Lengths of the segments to be generated
    std_dev : float or array_like
        Standard deviations of the segments to be generated
    repeated_labels : bool
        Flag to indicate whether segment labels should be repeated for similar segments.
        If ``True`` same label will be assigned for segments with same mean and std_dev,
        independently of length. If ``False`` each consecutive segment will have
        a unique label.

    Returns
    -------
    labels : np.array
        integer encoded array of labels, same length as data
    """
    if isinstance(std_dev, (float, int)):
        std_dev = np.repeat(std_dev, len(means))
    if repeated_labels:
        unique_labels = labels_with_repeats(means, std_dev)
    else:
        unique_labels = range(len(lengths))
    return np.repeat(unique_labels, lengths)


class GenBasicGauss:
    """Data generator base class in order to allow composition."""

    def __init__(self, means, lengths, std_dev=1.0, random_state=None):
        self.means = means
        self.lengths = lengths
        self.std_dev = std_dev
        self.random_state = random_state

    def sample(self):
        """Generate univariate mean shift random data sample."""
        return piecewise_normal(
            means=self.means,
            lengths=self.lengths,
            std_dev=self.std_dev,
            random_state=self.random_state,
        )
