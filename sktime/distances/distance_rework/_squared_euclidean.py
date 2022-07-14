# -*- coding: utf-8 -*-
import warnings
from typing import Callable, Tuple, Union

import numpy as np
from numba import njit

from sktime.distances.distance_rework.base import BaseDistance, DistanceCostCallable


class _SquaredEuclidean(BaseDistance):
    def independent_distance_factory(
        self, x: np.ndarray, y: np.ndarray, return_cost_matrix: bool, **kwargs: dict
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """Create a no_python distance for independent squared euclidean distance.

        This method will take only a univariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (1d array of shape (m1,))
            First time series
        y: np.ndarray (1d array of shape (m2,))
            Second time series
        return_cost_matrix: bool
            Boolean that when true will also return the cost matrix.
        kwargs: dict
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a cost matrix used to compute the distance is returned and a float
            that represents the dependent distance between x and y.
        """
        independent = self._independent_distance_factory(x, y, **kwargs)

        if return_cost_matrix is True:
            # Return callable that sums the cost matrix
            warnings.warn(
                "Manhattan does not produce a cost matrix so non will be " "returned."
            )

        @njit()
        def _distance_callable(_x: np.ndarray, _y: np.ndarray) -> float:
            total = 0
            for i in range(_x.shape[0]):
                curr_dist = independent(_x[i], _y[i])
                total += curr_dist
            return total

        return _distance_callable

    def _independent_distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCostCallable:
        """Create a no_python distance for independent squared euclidean distance.

        This method will take only a univariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (1d array of shape (m1,))
            First time series
        y: np.ndarray (1d array of shape (m2,))
            Second time series
        kwargs: dict
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a cost matrix used to compute the distance is returned and a float
            that represents the dependent distance between x and y.
        """

        @njit(cache=True)
        def _numba_squared_euclidean(_x, _y):
            x_size = _x.shape[0]
            distance = 0
            for i in range(x_size):
                distance += (_x[i] - _y[i]) ** 2

            # No cost matrix generated for squared euclidean
            return distance

        return _numba_squared_euclidean

    def dependent_distance_factory(
        self, x: np.ndarray, y: np.ndarray, return_cost_matrix: bool, **kwargs: dict
    ) -> Callable[[np.ndarray, np.ndarray], Union[float, Tuple[np.ndarray, float]]]:
        """Create a no_python distance for dependent distance.

        This method will take a multivariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d, m1))
            First time series
        y: np.ndarray (2d array of shape (d, m2))
            Second time series
        return_cost_matrix: bool
            Boolean that when true will also return the cost matrix.
        kwargs: dict
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], Union[float, Tuple[np.ndarray, float]]]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a cost matrix used to compute the distance is returned and a float
            that represents the dependent distance between x and y.
        """
        return self._dependent_distance_factory(x, y, **kwargs)

    def _dependent_distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCostCallable:
        """Create a no_python distance for dependent squared euclidean distance.

        This method will take a multivariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d, m1))
            First time series
        y: np.ndarray (2d array of shape (d, m2))
            Second time series
        kwargs: dict
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a cost matrix used to compute the distance is returned and a float
            that represents the dependent distance between x and y.
        """
        # In the case of Lp distances (i.e. euclidean is L2) independent and dependent
        # return the same result.
        return self.distance_factory(x, y, strategy="independent", **kwargs)
