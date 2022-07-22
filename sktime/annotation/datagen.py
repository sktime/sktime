# -*- coding: utf-8 -*-
"""Synthetic data generating functions."""

from typing import Union

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_random_state


def mean_shift(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    noise: Union[npt.ArrayLike, float] = 1.0,
    random_state: Union[int, np.random.RandomState] = None,
) -> npt.ArrayLike:
    """
    Generate series from segments.

    Each segment has length specified in ``lengths`` and data sampled from a normal
    distribution with a mean from ``means`` and standard deviation from ``noise``.

    Parameters
    ----------
    means : array_like
        Means of the segments to be generated
    lengths : array_like
        Lengths of the segments to be generated
    noise : float or array_like
        Standard deviations of the segments to be generated
    random_state : int or np.random.RandomState
        Either a random seed or RandomState instance

    Returns
    -------
    data : np.array
        univariate time series as np.array

    Examples
    --------
    >>> from sktime.annotation.datagen import mean_shift
    >>> mean_shift([1, 2, 3], lengths=[2, 4, 8], random_state=42)  # doctest: +SKIP
    array([1.49671415, 0.8617357 , 2.64768854, 3.52302986, 1.76584663,
        1.76586304, 4.57921282, 3.76743473, 2.53052561, 3.54256004,
        2.53658231, 2.53427025, 3.24196227, 1.08671976])

    >>> from sktime.annotation.datagen import mean_shift 
    >>> mean_shift([1, 2, 3], lengths=[2, 4, 8], noise=0)  # doctest: +SKIP
    array([1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.])

    >>> from sktime.annotation.datagen import mean_shift
    >>> mean_shift([1, 2, 3], lengths=[2, 4, 8], noise=[0, 0.5, 1.0])  # doctest: +SKIP
    array([1.        , 1.        , 2.32384427, 2.76151493, 1.88292331,
        1.88293152, 4.57921282, 3.76743473, 2.53052561, 3.54256004,
        2.53658231, 2.53427025, 3.24196227, 1.08671976])

    """
    rng = check_random_state(random_state)
    assert len(means) == len(lengths)

    if isinstance(noise, (float, int)):
        noise = np.repeat(noise, len(means))

    assert len(noise) == len(means)

    segments_data = [
        rng.normal(loc=mean, scale=sd, size=[length])
        for mean, length, sd in zip(means, lengths, noise)
    ]
    return np.concatenate(tuple(segments_data))


def labels_with_repeats(means: npt.ArrayLike, noise: npt.ArrayLike) -> npt.ArrayLike:
    """Generate labels for unique combinations of meas and noise."""
    data = [means, noise]
    unique, indices = np.unique(data, axis=1, return_inverse=True)
    labels = np.arange(unique.shape[1])
    return labels[indices]


def label_mean_shift(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    noise: Union[npt.ArrayLike, float] = 1.0,
    repeated_labels: bool = True,
) -> npt.ArrayLike:
    """
    Generate labels for a series composed of segments.

    Parameters
    ----------
    means : array_like
        Means of the segments to be generated
    lengths : array_like
        Lengths of the segments to be generated
    noise : float or array_like
        Standard deviations of the segments to be generated
    repeated_labels : bool
        Flag to indicate whether segment labels should be repeated for similar segments.
        If ``True`` same label will be assigned for segments with same mean and noise,
        independently of length. If ``False`` each consecutive segment will have
        a unique label.

    Returns
    -------
    labels : np.array
        integer encoded array of labels, same length as data
    """
    if isinstance(noise, (float, int)):
        noise = np.repeat(noise, len(means))
    if repeated_labels:
        unique_labels = labels_with_repeats(means, noise)
    else:
        unique_labels = range(len(lengths))
    return np.repeat(unique_labels, lengths)


class GenBasicGauss:
    """Data generator base class in order to allow composition."""

    def __init__(self, means, lengths, noise, random_state=None):
        self.means = means
        self.lengths = lengths
        self.noise = noise
        self.random_state = random_state

    def sample(self):
        """Generate data sample."""
        return mean_shift(
            means=self.means,
            lengths=self.lengths,
            noise=self.noise,
            random_state=self.random_state,
        )
