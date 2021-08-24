# -*- coding: utf-8 -*-
from typing import Union, Any, List, Tuple, Callable, Set

import numpy as np
from numba import njit, prange

from sktime.metrics.distances.base._types import SktimeSeries, SktimeMatrix
from sktime.utils.data_processing import (
    to_numpy_time_series,
    to_numpy_time_series_matrix,
)


class BaseDistance:
    """
    Class used as a base for time series distances

    Parameters
    ----------
    metric_name: str
        Str name for the metric (normally abbreviation for full name of distance)
    metric_aka: Set[str], defaults = None
        Other names for the metric (this is where full names go)
    """

    def __init__(self, metric_name: str, metric_aka: Set[str] = None):
        self.metric_name: str = metric_name
        self.metric_aka: Set[str] = metric_aka

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
            First time series
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
    ):
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


class BasePairwise:
    def pairwise(self, x: SktimeMatrix, y: SktimeMatrix = None) -> np.ndarray:
        """
        Method to compute a pairwise distance on a matrix (i.e. distance between each
        ts in the matrix)

        Parameters
        ----------
        x: np.ndarray
            First matrix of multiple time series
        y: np.ndarray
            Second matrix of multiple time series.

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between each point

        """
        if x.ndim <= 2:
            x = np.reshape(x, x.shape + (1,))
            y = np.reshape(y, y.shape + (1,))
        x, y, symmetric = BasePairwise._format_pairwise_matrix(x, y)
        return self._pairwise(x, y, symmetric)

    def _pairwise(self, x: np.ndarray, y: np.ndarray, symmetric: bool) -> np.ndarray:
        """
        Method to compute a pairwise distance on a matrix (i.e. distance between each
        ts in the matrix)

        Parameters
        ----------
        x: np.ndarray
            First matrix of multiple time series
        y: np.ndarray
            Second matrix of multiple time series.
        symmetric: bool
            boolean that is true when the two time series are equal to each other

        Returns
        -------
        np.ndarray
            Matrix containing the pairwise distance between each point
        """
        raise NotImplementedError("Missing distance implementation")

    @staticmethod
    def _format_pairwise_matrix(
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

    @staticmethod
    @njit(parallel=True)
    def compute_pairwise_matrix(
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

        for i in prange(x_size):
            curr_x = x[i]
            for j in range(y_size):
                if symmetric and j < i:
                    pairwise_matrix[i, j] = pairwise_matrix[j, i]
                else:
                    pairwise_matrix[i, j] = dist_func(curr_x, y[j])

        return pairwise_matrix
