from numba import njit, prange
import numpy as np
import pandas as pd

from .base import BaseTransformer

__author__ = "Angus Dempster"
__all__ = ["ROCKET"]

class Rocket(BaseTransformer):
    """ROCKET

    RandOm Convolutional KErnel Transform

    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Parameters
    ----------
    input_length : int, length of input time series
    num_kernels  : int, number of random convolutional kernels
    num_channels : int, number of channels for multivariate time series (default 1)
    """

    def __init__(self,  input_length, num_kernels, num_channels = 1):
        self.input_length = input_length
        self.num_kernels = num_kernels
        self.num_channels = num_channels
        self.kernels = _generate_kernels(input_length, num_kernels, num_channels)

    def fit(self, X, y = None):
        return self

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def transform(self, X):
        """Transforms input time series using random convolutional kernels.

        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)

        Returns
        -------
        pandas DataFrame, transformed features
        """
        return pd.DataFrame(_apply_kernels(self._to_numpy(X), self.kernels))

    @staticmethod
    def _to_numpy(X, a = None, b = None):
        return np.stack(X.iloc[a:b].applymap(lambda cell : cell.to_numpy()).apply(lambda row : np.stack(row), axis = 1).to_numpy())

@njit("Tuple((float32[:],int32[:],float32[:],int32[:],int32[:],int32[:],int32[:]))(int64,int64,int64)")
def _generate_kernels(input_length, num_kernels, num_channels):

    candidate_lengths = np.array((7, 9, 11), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    num_channel_indices = np.zeros(num_kernels, dtype = np.int32)
    for i in range(num_kernels):
        limit = min(num_channels, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype = np.int32)

    weights = np.zeros(np.int32(np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))), dtype = np.float32)
    biases = np.zeros(num_kernels, dtype = np.float32)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0 # for weights
    a2 = 0 # for channel_indices

    for i in range(num_kernels):

        l = lengths[i]
        n = num_channel_indices[i]

        _weights = np.random.normal(0, 1, n * l)

        b1 = a1 + (n * l)
        b2 = a2 + n

        a3 = 0
        for j in range(n):
            b3 = a3 + l
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(np.arange(0, num_channels), n, replace = False)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (l - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((l - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices

@njit(fastmath = True)
def _apply_kernel(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices):

    # zero padding
    if padding > 0:
        _num_channels, _input_length = X.shape
        _X = np.zeros((_num_channels, _input_length + (2 * padding)))
        _X[:, padding:(padding + _input_length)] = X
        X = _X

    num_channels, input_length = X.shape

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    for i in range(output_length):

        _sum = bias

        for j in range(length):

            dilation_j = dilation * j

            for k in range(num_channel_indices):

                _sum += weights[k, j] * X[channel_indices[k], i + dilation_j]

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max

@njit("float32[:,:](float64[:,:,:],Tuple((float32[::1],int32[:],float32[:],int32[:],int32[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def _apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_examples = len(X)
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float32) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for channel_indices

        for j in range(num_kernels):

            b1 = a1 + (num_channel_indices[j] * lengths[j])
            b2 = a2 + num_channel_indices[j]

            _weights = weights[a1:b1].reshape((num_channel_indices[j], lengths[j]))

            _X[i, (j * 2):((j * 2) + 2)] = \
            _apply_kernel(X[i], _weights, lengths[j], biases[j], dilations[j], paddings[j], num_channel_indices[j], channel_indices[a2:b2])

            a1 = b1
            a2 = b2

    return _X
