# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable, Tuple
from numba import njit, prange
from sktime.metrics.distances.dtw._dtw import Dtw, _cost_matrix
from sktime.metrics.distances.base.base import NumbaSupportedDistance, _numba_pairwise


@njit(cache=True)
def _return_path(cost_matrix: np.ndarray) -> np.ndarray:
    x_size, x_dim_size = cost_matrix.shape
    start_x = x_size - 1
    start_y = x_dim_size - 1
    path = []

    curr_x = start_x
    curr_y = start_y
    while curr_x != 0 or curr_y != 0:
        path.append([curr_x, curr_y])
        min_coord = np.argmin(
            np.array(
                [
                    cost_matrix[curr_x - 1, curr_y - 1],
                    cost_matrix[curr_x - 1, curr_y],
                    cost_matrix[curr_x, curr_y - 1],
                ]
            )
        )
        if min_coord == 0:
            curr_x -= 1
            curr_y -= 1
        elif min_coord == 1:
            curr_x -= 1
        else:
            curr_y -= 1
    path.append([0, 0])
    return np.array(path[::-1])


class DtwPath(Dtw, NumbaSupportedDistance):
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

        dtw_path = _return_path(cost_matrix)

        return np.sqrt(cost_matrix[-1, -1]), dtw_path

    def pairwise(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to compute a pairwise distance on a matrix (i.e. distance between each
        ts in the matrix) and return the dtw path

        Parameters
        ----------
        x: np.ndarray
            First matrix of multiple time series
        y: np.ndarray
            Second matrix of multiple time series.

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between the two matrices
        np.ndarray
            Matrix containing the path for each of the pairwise. NOTE: This is padded
            with 0s

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

            dtw_path = _return_path(cost_matrix)

            return np.sqrt(cost_matrix[-1, -1]), dtw_path

        return numba_dtw


@njit(parallel=True)
def _numba_dtw_path_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool,
    dist_func: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method that takes a distance function and computes the pairwise distance

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
        Matrix containing the path for each of the pairwise
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    path_matrix_size = x_size + y_size - 1

    pairwise_matrix_dist = np.zeros((x_size, y_size))
    pairwise_matrix_path = np.zeros((x_size, y_size, path_matrix_size, 2))

    for i in range(x_size):
        curr_x = x[i]

        for j in prange(y_size):
            if symmetric and j < i:
                pairwise_matrix_dist[i, j] = pairwise_matrix_dist[j, i]
                pairwise_matrix_path[i, j] = pairwise_matrix_path[j, i, :]
            else:
                pairwise_matrix_dist[i, j], path = dist_func(curr_x, y[j])

                path_index = 0
                for k in range(path_matrix_size - path.shape[0], path_matrix_size):
                    pairwise_matrix_path[i, j, k] = path[path_index]
                    path_index += 1

    return pairwise_matrix_dist, pairwise_matrix_path
