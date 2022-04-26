# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]


import warnings
from typing import Any, List, Tuple

import numpy as np
from numba import njit
from numba.core.errors import NumbaWarning

from sktime.distances._distance_alignment_paths import compute_min_return_path
from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.base._types import DistanceAlignmentPathCallable
from sktime.distances.lower_bounding import resolve_bounding_matrix

# Warning occurs when using large time series (i.e. 1000x1000)
warnings.simplefilter("ignore", category=NumbaWarning)


class _TweDistance(NumbaDistance):

    def _distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            window: float = None,
            itakura_max_slope: float = None,
            bounding_matrix: np.ndarray = None,
            lmbda: float = 1.0,
            nu: float = 0.001,
            p: int = 2,
            **kwargs: Any
    ) -> DistanceCallable:
        x = pad_ts(x)
        y = pad_ts(y)
        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        def numba_twe_distance(
                _x: np.ndarray,
                _y: np.ndarray,
        ) -> float:
            cost_matrix = _twe_cost_matrix(_x, _y, _bounding_matrix, lmbda, nu, p)
            return cost_matrix[-1, -1], cost_matrix

        return numba_twe_distance


def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1.0 / p)

def pad_ts(x):
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1))
    zero_arr = np.array([0.0])
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x

def _twe_cost_matrix(
        x: np.ndarray,
        y: np.ndarray,
        bounding_matrix: np.ndarray,
        lmbda: float,
        nu: float,
        p: int,
) -> np.ndarray:
    """
        lmbda: float >= 0, default: 1.0
        A constant penalty that punishes the editing efforts
    nu: float > 0, default: 0.001
        A non-negative constant which characterizes the stiffness of the elastic TWED measure.
    p: int
        Order of the p-norm for local cost.
    """
    # padded_x = np.zeros((x.shape[0] + 1, x.shape[1] + 1))
    # padded_x[1, :] = x
    x = pad_ts(x)
    y = pad_ts(y)
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size, y_size))

    # Initialize DP Matrix and set first row and column to infinity
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    for i in range(1, x_size):
        for j in range(1, y_size):
            if np.isfinite(bounding_matrix[i, j]):
                # Deletion in A
                del_a = (
                        cost_matrix[i - 1, j]
                        + Dlp(x[:, i - 1], x[:, i], p=p)
                        + nu
                        + lmbda
                )

                # Deletion in B
                del_b = (
                        cost_matrix[i, j - 1]
                        + Dlp(y[:, j - 1], y[:, j], p=p)
                        + nu
                        + lmbda
                )

                # Keep data points in both time series
                match = (
                        cost_matrix[i - 1, j - 1]
                        + Dlp(x[:, i], y[:, j], p=p)
                        + Dlp(x[:, i - 1], y[:, j - 1], p=p)
                        + nu
                )

                # Choose the operation with the minimal cost and update DP Matrix
                cost_matrix[i, j] = min(del_a, del_b, match)
    return cost_matrix
