# -*- coding: utf-8 -*-
"""
Synthetic data generating functions.
"""

from typing import Union

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_random_state


# what if we could pass data_gen functions - rather than assuming means?
#   - could make it easier to swap out different mean assumptions


def mean_shift(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    noise: Union[npt.ArrayLike, float] = 1.0,
    random_state: Union[int, np.random.RandomState] = None,
) -> npt.ArrayLike:
    """
    Generates a series composed of segments, where each segment has length specified
    in ``lengths`` and data sampled from a normal distribution with a mean from ``means``
    and standard deviation from ``noise``.

    Parameters
    ----------
    means : array_like
        Means of the segments to be generated
    lengths : array_like
        Lengths of the segments to be generated
    noise : float ir array_like
        Standard deviations fo the segments to be generated
    repeated_labels : bool
        A flag to indicate whether segment labels should be repeated for similar segments.
        If ``True`` same label will be assigned for segments with same mean and noise,
        independently of length. If ``False`` each consecutive segment will have a unique
        label.
    random_state : int or np.random.RandomState
        Either a random seed or RandomState instance

    Returns
    -------
    data : np.array
        univariate time series as np.array
    """

    rng = check_random_state(random_state)
    assert len(means) == len(lengths)

    if isinstance(noise, float):
        noise = np.repeat(noise, len(means))

    assert len(noise) == len(means)

    segments_data = [
        rng.normal(loc=mean, scale=sd, size=[length])
        for mean, length, sd in zip(means, lengths, noise)
    ]
    return np.concatenate(tuple(segments_data))


def labels_with_repeats(means: npt.ArrayLike, noise: npt.ArrayLike) -> npt.ArrayLike:
    """
    Generate labels for unique combinations of meas and noise.
    """
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
    noise : float ir array_like
        Standard deviations fo the segments to be generated
    repeated_labels : bool
        A flag to indicate whether segment labels should be repeated for similar segments.
        If ``True`` same label will be assigned for segments with same mean and noise,
        independently of length. If ``False`` each consecutive segment will have a unique
        label.

    Returns
    -------
    labels : np.array
        integer encoded array of labels, same length as data
    """
    if isinstance(noise, float):
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


# this data has no autocorrelation concerns
# it just creates an (N x 6) data array,
# where the first half of the data exists in 1D,
# and the second half of the data exists in 3D
# need to add noise (and parametarise)

# generate dataset
# N=10
# np.random.seed(10002)
# X=np.zeros((N,6))

# # half the points from one generating regime
# for j in range(1):
# 	X[:N,j]= np.random.normal(0,3,N/2)

# # the other half from another
# for j in range(3):
# 	X[N:,j]= np.random.normal(2,1,N/2)
