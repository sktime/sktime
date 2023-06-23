"""Isolated numba imports for _minirocket."""

__author__ = ["angus924"]

import numpy as np

from sktime.utils.numba.njit import njit
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("numba", severity="none"):
    from numba import prange, vectorize

    @vectorize("float32(float32,float32)", nopython=True, cache=True)
    def _PPV(a, b):
        if a > b:
            return 1
        else:
            return 0


@njit(
    "float32[:](float32[:,:],int32[:],int32[:],float32[:],optional(int32))",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles, seed):
    if seed is not None:
        np.random.seed(seed)

    n_instances, n_timepoints = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
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

    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(n_instances)]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end

    return biases


def _fit_dilations(n_timepoints, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
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


def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _fit(X, num_features=10_000, max_dilations_per_kernel=32, seed=None):
    _, n_timepoints = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(
        n_timepoints, num_features, max_dilations_per_kernel
    )

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles, seed)

    return dilations, num_features_per_dilation, biases


@njit(
    "float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])))",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform(X, parameters):
    n_instances, n_timepoints = X.shape

    dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
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

    features = np.zeros((n_instances, num_features), dtype=np.float32)

    for example_index in prange(n_instances):
        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[
                            example_index, feature_index_start + feature_count
                        ] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[
                            example_index, feature_index_start + feature_count
                        ] = _PPV(
                            C[padding:-padding],
                            biases[feature_index_start + feature_count],
                        ).mean()

                feature_index_start = feature_index_end

    return features
