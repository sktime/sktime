# -*- coding: utf-8 -*-
"""Base distance for numba distance."""
# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Set, Tuple, Union

import numpy as np
from numba import njit

from sktime.distances._distance_alignment_paths import compute_min_return_path
from sktime.distances.distance_rework.base._types import (
    AlignmentPathReturn,
    DistanceCostCallable,
)
from sktime.distances.lower_bounding import resolve_bounding_matrix


class BaseDistance(ABC):
    """Abstract class to define a numba compatible distance metric."""

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        strategy: str = "independent",
        **kwargs: dict,
    ) -> float:
        """Compute the distance between two time series.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series.
        y: np.ndarray (2d array)
            Second time series.
        strategy: str, defaults = 'independent'
            Strategy to use for computing the distance. Must either be 'independent'
            or 'dependent'.
        kwargs: dict
            kwargs for the distance computation.

        Returns
        -------
        float
            Distance between x and y.
        """
        dist_callable = self.distance_factory(x, y, strategy=strategy, **kwargs)
        return dist_callable(x, y)

    def distance_alignment_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
        strategy: str = "independent",
        **kwargs: dict,
    ) -> AlignmentPathReturn:
        """Compute the distance alignment path between two time series.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series.
        y: np.ndarray (2d array)
            Second time series.
        strategy: str, defaults = 'independent'
            Strategy to use for computing the distance. Must either be 'independent'
            or 'dependent'.
        kwargs: dict
            kwargs for the distance computation.

        Returns
        -------
        list[tuple]
            List of tuples that is the path through the matrix
        float
            Distance between x and y.
        np.ndarray (of shape (len(x), len(y)).
            Optional return only given if return_cost_matrix = True.
            Cost matrix used to compute the distance.
        """
        dist_callable = self.distance_alignment_path_factory(
            x, y, strategy=strategy, **kwargs
        )
        return dist_callable(x, y)

    def distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        strategy: str = "independent",
        return_cost_matrix: bool = False,
        **kwargs: dict,
    ) -> DistanceCostCallable:
        """Create a no_python distance.

        This method will validate the kwargs and ensure x and y are in the correct
        format and then return a no_python compiled distance that uses the kwargs.

        The no_python compiled distance will be in the form:
        Callable[[np.ndarray, np.ndarray], float]. #

        This can then be used to calculate distances efficiently or can be used
        inside other no_python functions.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series
        y: np.ndarray (2d array)
            Second time series
        strategy: str, defaults = 'independent'
            Strategy to use for computing the distance. Must either be 'independent'
            or 'dependent'.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        kwargs: dict
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a float is then returned that represents the distance between x and y.
            This callable will be no_python compiled.

        Raises
        ------
        ValueError
            If x or y is not a numpy array.
            If x or y has less than or greater than 2 dimensions.
        RuntimeError
            If the distance metric could not be compiled to no_python.
        """
        if strategy not in ["independent", "dependent"]:
            raise ValueError("strategy must be either 'independent' or 'dependent'")

        BaseDistance._validate_factory_timeseries(x)
        BaseDistance._validate_factory_timeseries(y)

        if strategy == "independent":
            return self.independent_distance_factory(x, y, return_cost_matrix, **kwargs)

        return self.dependent_distance_factory(x, y, return_cost_matrix, **kwargs)

    def distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        strategy: str = "independent",
        return_cost_matrix: bool = False,
        **kwargs: dict,
    ) -> DistanceCostCallable:
        """Create a no_python distance alignment path.

        It should validate kwargs and then compile a no_python callable
        that takes (x, y) as parameters and returns a float that represents the distance
        between the two time series.

        ----------
        x: np.ndarray (2d array)
            First time series
        y: np.ndarray (2d array)
            Second time series
        strategy: str, defaults = 'independent'
            Strategy to use for computing the distance. Must either be 'independent'
            or 'dependent'.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        kwargs: kwargs
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a float is then returned that represents the distance between x and y.
            This callable will be no_python compiled.

        Raises
        ------
        ValueError
            If x or y is not a numpy array.
            If x or y has less than or greater than 2 dimensions.
        RuntimeError
            If the distance metric could not be compiled to no_python.
        """
        dist_callable = self.distance_factory(
            x, y, return_cost_matrix=True, strategy=strategy, **kwargs
        )
        min_path_callable = self._compute_return_path_factory()

        window = None
        itakura_max_slope = None
        bounding_matrix = None
        if "window" in kwargs:
            window = kwargs["window"]
        if "itakura_max_slope" in kwargs:
            itakura_max_slope = kwargs["itakura_max_slope"]
        if "bounding_matrix" in kwargs:
            bounding_matrix = kwargs["bounding_matrix"]

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if return_cost_matrix is True:

            @njit()
            def _distance_callable(
                _x: np.ndarray, _y: np.ndarray
            ) -> AlignmentPathReturn:
                cost_matrix, distance = dist_callable(_x, _y)
                path = min_path_callable(cost_matrix, _bounding_matrix)
                return path, distance, cost_matrix

        else:

            @njit()
            def _distance_callable(
                _x: np.ndarray, _y: np.ndarray
            ) -> AlignmentPathReturn:
                cost_matrix, distance = dist_callable(_x, _y)
                path = min_path_callable(cost_matrix, _bounding_matrix)
                return path, distance

        return _distance_callable(x, y)

    def independent_distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        **kwargs: dict
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """Create a no_python distance for independent distance.

        This method will take only a univariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (1d array of shape (m1,))
            First time series
        y: np.ndarray (1d array of shape (m2,))
            Second time series
        return_cost_matrix: bool, default = False
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
            @njit()
            def _distance_callable(
                _x: np.ndarray, _y: np.ndarray
            ) -> Tuple[np.ndarray, float]:
                total = 0
                cost_matrix = np.zeros((_x.shape[1], _y.shape[1]))
                for i in range(x.shape[0]):
                    curr_cost_matrix, curr_dist = independent(_x[i], _y[i])
                    cost_matrix = np.add(cost_matrix, curr_cost_matrix)
                    total += curr_dist
                return cost_matrix, total

        else:

            @njit()
            def _distance_callable(_x: np.ndarray, _y: np.ndarray) -> float:
                total = 0
                for i in range(_x.shape[0]):
                    curr_cost_matrix, curr_dist = independent(_x[i], _y[i])
                    total += curr_dist
                return total

        return _distance_callable

    def dependent_distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        **kwargs: dict
    ) -> Callable[[np.ndarray, np.ndarray], Union[float, Tuple[np.ndarray, float]]]:
        """Create a no_python distance for dependent distance.

        This method will take a multivariate time series and compute the distance.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d, m1))
            First time series
        y: np.ndarray (2d array of shape (d, m2))
            Second time series
        return_cost_matrix: bool, defaults = False
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
        if return_cost_matrix is True:
            # Return callable that also has cost matrix
            return self._dependent_distance_factory(x, y, **kwargs)

        dependent = self._dependent_distance_factory(x, y, **kwargs)

        @njit()
        def _distance_callable(_x: np.ndarray, _y: np.ndarray) -> float:
            return dependent(_x, _y)[1]

        return _distance_callable

    @abstractmethod
    def _independent_distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCostCallable:
        """Create a no_python distance for independent distance.

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
        ...

    @abstractmethod
    def _dependent_distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCostCallable:
        """Create a no_python distance for dependent distance.

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
        ...

    @staticmethod
    def _validate_factory_timeseries(x: np.ndarray) -> None:
        """Ensure the time series are correct format.

        Parameters
        ----------
        x: np.ndarray (2d array)
            A time series to check.

        Raises
        ------
        ValueError
            If x is not a numpy array.
            If x has less than or greater than 2 dimensions.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(
                f"The array {x} is not a numpy array. Please ensure it"
                f"is a 2d numpy and try again."
            )

        if x.ndim != 2:
            raise ValueError(
                f"The array {x} has the incorrect number of dimensions."
                f"Ensure the array has exactly 2 dimensions and try "
                f"again."
            )

    @staticmethod
    def _compute_return_path_factory() -> Callable[
        [np.ndarray, np.ndarray], List[Tuple]
    ]:
        """Create a no_python callable that returns the path.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], List[Tuple]]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a list of tuples is then returned that represents the path between x and y.
        """
        return compute_min_return_path


# Metric
class MetricInfo(NamedTuple):
    """Define a registry entry for a metric."""

    # Name of the distance
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # Python distance function (can use numba inside but callable must be in python)
    dist_func: Callable[[np.ndarray, np.ndarray], float]
    # NumbaDistance class
    dist_instance: BaseDistance
    # Distance path callable
    dist_alignment_path_func: AlignmentPathReturn = None
