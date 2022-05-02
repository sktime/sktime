# -*- coding: utf-8 -*-
import multiprocessing

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange, set_num_threads

from sktime.transformations.base import BaseTransformer


class MultiRocketMultivariate(BaseTransformer):
    """
    MultiRocket multivariate version.

    Multi RandOm Convolutional KErnel Transform

    **Multivariate**

    A provisional and naive extension of MultiRocket to multivariate input.  Use
    class MultiRocket for univariate input.

    @article{Tan2021MultiRocket,
    title={{MultiRocket}: Multiple pooling operators and transformations
    for fast and effective time series classification},
    author={Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and
    Webb, Geoffrey I},
    year={2021},
    journal={arxiv:2102.00457v3}
    }

    Parameters
    ----------
    num_kernels              : int, number of random convolutional kernels
    (default 6,250)
    calculated number of features is the nearest multiple of
    n_features_per_kernel(default 4)*84=336 < 50,000
    (2*n_features_per_kernel(default 4)*num_kernels(default 6,250))
    max_dilations_per_kernel : int, maximum number of dilations per kernel (default 32)
    n_features_per_kernel    : int, number of features per kernel (default 4)
    normalise                : int, normalise the data (default False)
    n_jobs                   : int, optional (default=1) The number of jobs to run in
    parallel for `transform`. ``-1`` means using all processors.
    random_state             : int, random seed (optional, default None)

    Attributes
    ----------
    parameter : tuple
        parameter (dilations, num_features_per_dilation, biases) for
        transformation of input X
    parameter1 : tuple
        parameter (dilations, num_features_per_dilation, biases) for
        transformation of input X1 = np.diff(X, 1)

    See Also
    --------
    MultiRocket, MiniRocket, MiniRocketMultivariate, Rocket

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph
        and Webb, Geoffrey I,
        "MultiRocket: Multiple pooling operators and transformations
        for fast and effective time series classification",
        2021, https://arxiv.org/abs/2102.00457v3

    Examples
    --------
    >>> from sktime.transformations.panel.rocket._multirocket import MultiRocket
    >>> from sktime.datasets import load_italy_power_demand
    >>> X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    >>> X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    >>> trf = MultiRocket()
    >>> trf.fit(X_train)
    MultiRocket(...)
    >>> X_train = trf.transform(X_train)
    >>> X_test = trf.transform(X_test)
    """

    _tags = {
        "univariate-only": False,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
    }

    def __init__(
        self,
        num_kernels=6_250,
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        normalise=False,
        n_jobs=1,
        random_state=None,
    ):

        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None

        self.parameter = None
        self.parameter1 = None

        super(MultiRocketMultivariate, self).__init__()

    def _fit(self, X, y=None):
        """Fit dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

        if X.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _X1 = np.zeros((X.shape[0], X.shape[1], 10), dtype=X.dtype)
            _X1[:, :, : X.shape[2]] = X
            X = _X1
            del _X1

        self.parameter = self._get_parameter(X)
        _X1 = np.diff(X, 1)

        self.parameter1 = self._get_parameter(_X1)

        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

        _X1 = np.diff(X, 1)

        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)

        X = _transform(
            X,
            _X1,
            self.parameter,
            self.parameter1,
            self.n_features_per_kernel,
        )
        X = np.nan_to_num(X)

        set_num_threads(prev_threads)

        return pd.DataFrame(X)

    def _get_parameter(self, X):
        _, num_channels, input_length = X.shape

        num_kernels = 84

        dilations, num_features_per_dilation = _fit_dilations(
            input_length, self.num_kernels, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = _quantiles(num_kernels * num_features_per_kernel)

        num_dilations = len(dilations)
        num_combinations = num_kernels * num_dilations

        max_num_channels = min(num_channels, 9)
        max_exponent = np.log2(max_num_channels + 1)

        num_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, num_combinations)
        ).astype(np.int32)

        channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

        num_channels_start = 0
        for combination_index in range(num_combinations):
            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]
            num_channels_end = num_channels_start + num_channels_this_combination
            channel_indices[num_channels_start:num_channels_end] = np.random.choice(
                num_channels, num_channels_this_combination, replace=False
            )

            num_channels_start = num_channels_end

        biases = _fit_biases(
            X,
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            quantiles,
            self.random_state,
        )

        return (
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            biases,
        )


@njit(
    "float32[:](float64[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],"
    "optional(int64))",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(
    X,
    num_channels_per_combination,
    channel_indices,
    dilations,
    num_features_per_dilation,
    quantiles,
    seed,
):

    if seed is not None:
        np.random.seed(seed)
    num_examples, num_channels, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array(
    #   [_ for _ in combinations(np.arange(9), 3)], dtype = np.int32
    # )
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[
                num_channels_start:num_channels_end
            ]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(
                (num_channels_this_combination, input_length), dtype=np.float32
            )
            C_alpha[:] = A

            C_gamma = np.zeros(
                (9, num_channels_this_combination, input_length), dtype=np.float32
            )
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


@njit(
    "float32[:,:](float64[:,:,:],float64[:,:,:],"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),int32)",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform(X, X1, parameters, parameters1, n_features_per_kernel=4):
    num_examples, num_channels, input_length = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters
    _, _, dilations1, num_features_per_dilation1, biases1 = parameters1

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array(
    #   [_ for _ in combinations(np.arange(9), 3)], dtype = np.int32
    # )
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)

    features = np.zeros(
        (num_examples, (num_features + num_features1) * n_features_per_kernel),
        dtype=np.float32,
    )
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations1):

            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros((num_channels, input_length - 1), dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, num_channels, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A1[:, :end]
                C_gamma[gamma_index, :, -end:] = G1[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A1[:, start:]
                C_gamma[gamma_index, :, :-start] = G1[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )

                feature_index_start = feature_index_end

    return features
