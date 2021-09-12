# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

from typing import Union, Tuple, Callable
import numpy as np
from numba import njit, prange

from sktime.metrics.distances.base._types import SktimeSeries, SktimeMatrix
from sktime.utils.validation.panel import to_numpy_time_series_matrix
from sktime.utils.validation.series import to_numpy_time_series


class BaseDistance:
    def distance(
        self,
        x: SktimeSeries,
        y: SktimeSeries,
    ) -> float:
        """
        Method used to compute the distance between two ts series

        Parameters
        ----------
        x: np.ndarray or pd.DataFrame or pd.Series or List
            First time series. This
        y: np.ndarray or pd.DataFrame or pd.Series or List
            Second time series

        Returns
        -------
        float
            Distance between time series x and time series y
        """
        x = to_numpy_time_series(x)
        y = to_numpy_time_series(y)
        if x.ndim != y.ndim:
            raise ValueError(
                "The number of dims of x must match the number of" "dims of y"
            )
        return self._distance(x, y)

    def _distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
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
        """
        raise NotImplementedError("Missing distance implementation")

    def pairwise(self, x: SktimeMatrix, y: SktimeMatrix = None) -> np.ndarray:
        """
        Method to compute a pairwise distance on a matrix (i.e. distance between each
        ts in the matrix)

        Parameters
        ----------
        x: np.ndarray or pd.Dataframe or List
            First matrix of multiple time series
        y: np.ndarray or pd.Dataframe or List
            Second matrix of multiple time series.

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between each point
        """
        x, y, symmetric = BaseDistance.format_pairwise_matrix(x, y)
        if isinstance(self, NumbaSupportedDistance):
            return _numba_pairwise(x, y, symmetric, self.numba_distance(x, y))
        else:
            return self._pairwise(x, y, symmetric)

    def _pairwise(
        self,
        x: np.ndarray,
        y: np.ndarray,
        symmetric: bool,
    ) -> np.ndarray:
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

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between the two matrices
        """
        x_size = x.shape[0]
        y_size = y.shape[0]

        pairwise_matrix = np.zeros((x_size, y_size))

        for i in range(x_size):
            curr_x = x[i]
            for j in range(y_size):
                if symmetric and j < i:
                    pairwise_matrix[i, j] = pairwise_matrix[j, i]
                else:
                    curr_y = y[j]
                    test = self.distance(curr_x, curr_y)
                    pairwise_matrix[i, j] = self.distance(curr_x, y[j])

        return pairwise_matrix

    @staticmethod
    def format_pairwise_matrix(
        x: SktimeMatrix, y: Union[SktimeMatrix, None] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Method used to check and format the pairwise matrices passed to a pairwise
        distance operation

        Parameters
        ----------
        x: np.ndarray or pd.Dataframe or List
            First matrix
        y: np.ndarray or pd.Dataframe or List, defaults = None
            Second matrix if None the set to the value of x (i.e. comparing to self)

        Returns
        -------
        x: np.ndarray
            First matrix checked and formatted
        y: np.ndarray
            Second matrix checked and formatted
        symmetric: bool
            Boolean marking if the pairwise matrix will be symmetric (i.e. x = y)
        """
        x = to_numpy_time_series_matrix(x)

        if y is None:
            y = np.copy(x)
            symmetric = True
        else:
            y = to_numpy_time_series_matrix(y)
            if np.array_equal(x, y):
                symmetric = True
            else:
                symmetric = False

        return x, y, symmetric


class NumbaSupportedDistance:
    def numba_distance(self, x, y) -> Callable[[np.ndarray, np.ndarray], float]:
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

        raise NotImplementedError("Implement method")


@njit(parallel=True)
def _numba_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    symmetric: bool,
    dist_func: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
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
    """
    x_size = x.shape[0]
    y_size = y.shape[0]

    pairwise_matrix = np.zeros((x_size, y_size))

    for i in range(x_size):
        curr_x = x[i]

        for j in prange(y_size):
            if symmetric and j < i:
                pairwise_matrix[i, j] = pairwise_matrix[j, i]
            else:
                pairwise_matrix[i, j] = dist_func(curr_x, y[j])

    return pairwise_matrix
