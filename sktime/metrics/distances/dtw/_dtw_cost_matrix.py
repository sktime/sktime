# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable, Tuple
from numba import njit, prange
from sktime.metrics.distances.dtw._dtw import Dtw, _cost_matrix
from sktime.metrics.distances.base.base import NumbaSupportedDistance, _numba_pairwise
from sktime.metrics.distances.base._types import SktimeMatrix


class DtwCostMatrix(Dtw, NumbaSupportedDistance):
    def _distance(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Method used to compute the distance between two ts series

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        float
            Distance between time series x and time series y
        np.ndarray
            Path generated to calculate dtw
        """

        bounding_matrix, pre_computed_distances = self._dtw_setup(x, y)

        cost_matrix = _cost_matrix(x, y, bounding_matrix, pre_computed_distances)

        return np.sqrt(cost_matrix[-1, -1]), cost_matrix

    def pairwise(
        self, x: SktimeMatrix, y: SktimeMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method that takes a distance function and computes the pairwise distance and
        returns the cost matrix

        Parameters
        ----------
        x: np.ndarray
            First 3d numpy array containing multiple timeseries
        y: np.ndarray
            Second 3d numpy array containing multiple timeseries

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between the two matrices
        np.ndarray
            Matrix containing the cost for each of the pairwise
        """
        if x.ndim <= 2:
            x = np.reshape(x, x.shape + (1,))
            y = np.reshape(y, y.shape + (1,))
        x, y, symmetric = self._format_pairwise_matrix(x, y)

        distance_matrix, path_matrix = _numba_dtw_path_pairwise(
            x, y, symmetric, self.numba_distance(x, y)
        )

        return distance_matrix, path_matrix

    def numba_distance(
        self, x, y
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]]:
        """
        Method used to return a numba callable distance, this assume that all checks
        have been done so the function returned doesn't need to check but checks
        should be done before the return

        Parameters
        ----------
        x: np.ndarray
            First time series
        y: np.ndarray
            Second time series

        Returns
        -------
        Callable
            Numba compiled function (i.e. has @njit decorator)
        """
        bounding_matrix, dist_func = self._numba_parameter_check(x, y)

        @njit()
        def numba_dtw(
            x: np.ndarray,
            y: np.ndarray,
            bounding_matrix: np.ndarray = bounding_matrix,
            dist_func: Callable = dist_func,
        ) -> Tuple[float, np.ndarray]:
            symm = np.array_equal(x, y)

            computed_distances = _numba_pairwise(x, y, symm, dist_func)

            cost_matrix = _cost_matrix(x, y, bounding_matrix, computed_distances)

            return np.sqrt(cost_matrix[-1, -1]), cost_matrix

        return numba_dtw


@njit(parallel=True)
def _numba_dtw_path_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool,
    dist_func: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method that takes a distance function and computes the pairwise distance and
    returns the cost matrix

    Parameters
    ----------
    x: np.ndarray
        First 3d numpy array containing multiple timeseries
    y: np.ndarray
        Second 3d numpy array containing multiple timeseries
    symmetric: bool
        Boolean when true means the two matrices are the same
    dist_func: Callable[[np.ndarray, np.ndarray], float]
        Callable that is a distance function to measure the distance between two
        time series. NOTE: This must be a numba compatible function (i.e. @njit)

    Returns
    -------
    np.ndarray
        Matrix containing the pairwise distance between the two matrices
    np.ndarray
        Matrix containing the cost for each of the pairwise
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    x_dims = x.shape[1]
    y_dims = y.shape[1]

    pairwise_matrix_dist = np.zeros((x_size, y_size))
    pairwise_cost_matrix = np.zeros((x_size, y_size, x_dims, y_dims))

    for i in range(x_size):
        curr_x = x[i]

        for j in prange(y_size):
            if symmetric and j < i:
                pairwise_matrix_dist[i, j] = pairwise_matrix_dist[j, i]
                pairwise_cost_matrix[i, j] = pairwise_cost_matrix[j, i, :]
            else:
                pairwise_matrix_dist[i, j], pairwise_cost_matrix[i, j] = dist_func(
                    curr_x, y[j]
                )
    return pairwise_matrix_dist, pairwise_cost_matrix
