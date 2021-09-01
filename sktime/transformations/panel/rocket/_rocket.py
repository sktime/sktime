# -*- coding: utf-8 -*-
__author__ = "Angus Dempster"
__all__ = ["Rocket"]

import multiprocessing

import numpy as np
import pandas as pd

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.panel import check_X

from numba import njit, get_num_threads, set_num_threads
from numba import prange


class Rocket(_PanelToTabularTransformer):
    """ROCKET

    RandOm Convolutional KErnel Transform

    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Francois and Webb,
      Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Parameters
    ----------
    num_kernels  : int, number of random convolutional kernels (default 10,000)
    normalise    : boolean, whether or not to normalise the input time
    series per instance (default True)
    n_jobs             : int, optional (default=1) The number of jobs to run in
    parallel for `transform`. ``-1`` means using all processors.
    random_state : int (ignored unless int due to compatability with Numba),
    random seed (optional, default None)
    """

    def __init__(self, num_kernels=10_000, normalise=True, n_jobs=1, random_state=None):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        super(Rocket, self).__init__()

    def fit(self, X, y=None):
        """Infers time series length and number of channels / dimensions (
        for multivariate time series) from input pandas DataFrame,
        and generates random kernels.

        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)

        Returns
        -------
        self
        """
        X = check_X(X, coerce_to_numpy=True)
        _, self.n_columns, n_timepoints = X.shape
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, self.n_columns, self.random_state
        )
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transforms input time series using random convolutional kernels.

        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)

        Returns
        -------
        pandas DataFrame, transformed features
        """
        self.check_is_fitted()
        _X = check_X(X, coerce_to_numpy=True)
        if self.normalise:
            _X = (_X - _X.mean(axis=-1, keepdims=True)) / (
                _X.std(axis=-1, keepdims=True) + 1e-8
            )
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        t = pd.DataFrame(_apply_kernels(_X, self.kernels))
        set_num_threads(prev_threads)
        return t


@njit(
    "Tuple((float64[:],int32[:],float64[:],int32[:],int32[:],int32[:],"
    "int32[:]))(int64,int64,int64,optional(int64))",
    cache=True,
)
def _generate_kernels(n_timepoints, num_kernels, n_columns, seed):
    if seed is not None:
        np.random.seed(seed)

    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    num_channel_indices = np.zeros(num_kernels, dtype=np.int32)
    for i in range(num_kernels):
        limit = min(n_columns, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float64), num_channel_indices.astype(np.float64))
        ),
        dtype=np.float64,
    )
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(num_kernels):

        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _weights = np.random.normal(0, 1, _num_channel_indices * _length)

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0  # for weights (per channel)
        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_columns), _num_channel_indices, replace=False
        )

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((n_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )


@njit(fastmath=True, cache=True)
def _apply_kernel_univariate(X, weights, length, bias, dilation, padding):
    n_timepoints = len(X)

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(fastmath=True, cache=True)
def _apply_kernel_multivariate(
    X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices
):
    n_columns, n_timepoints = X.shape

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < n_timepoints:

                for k in range(num_channel_indices):
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(
    "float64[:,:](float64[:,:,:],Tuple((float64[::1],int32[:],float64[:],"
    "int32[:],int32[:],int32[:],int32[:])))",
    parallel=True,
    fastmath=True,
    cache=True,
)
def _apply_kernels(X, kernels):
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    ) = kernels

    n_instances, n_columns, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros(
        (n_instances, num_kernels * 2), dtype=np.float64
    )  # 2 features per kernel

    for i in prange(n_instances):

        a1 = 0  # for weights
        a2 = 0  # for channel_indices
        a3 = 0  # for features

        for j in range(num_kernels):

            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 2

            if num_channel_indices[j] == 1:

                _X[i, a3:b3] = _apply_kernel_univariate(
                    X[i, channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )

            else:

                _weights = weights[a1:b1].reshape((num_channel_indices[j], lengths[j]))

                _X[i, a3:b3] = _apply_kernel_multivariate(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2],
                )

            a1 = b1
            a2 = b2
            a3 = b3

    return _X
