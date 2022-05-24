# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Set

import numpy as np

from sktime.distances.base._types import (
    AlignmentPathReturn,
    DistanceAlignmentPathCallable,
    DistanceCallable,
)


class NumbaDistance(ABC):
    """Abstract class to define a numba compatible distance metric."""

    def distance(self, x: np.ndarray, y: np.ndarray, **kwargs: dict) -> float:
        """Compute the distance between two time series.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series.
        y: np.ndarray (2d array)
            Second time series.
        kwargs: dict
            kwargs for the distance computation.

        Returns
        -------
        float
            Distance between x and y.
        """
        dist_callable = self.distance_factory(x, y, **kwargs)
        return dist_callable(x, y)

    def distance_alignment_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        **kwargs: dict,
    ) -> float:
        """Compute the distance alignment path between two time series.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series.
        y: np.ndarray (2d array)
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix
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
            x, y, return_cost_matrix=return_cost_matrix, **kwargs
        )
        return dist_callable(x, y)

    def distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCallable:
        """Create a no_python distance.

        This method will validate the kwargs and ensure x and y are in the correct
        format and then return a no_python compiled distance that uses the kwargs.

        The no_python compiled distance will be in the form:
        Callable[[np.ndarray, np.ndarray], float]. #

        This can then be used to to calculate distances efficiently or can be used
        inside other no_python functions.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series
        y: np.ndarray (2d array)
            Second time series
        kwargs: kwargs
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
        NumbaDistance._validate_factory_timeseries(x)
        NumbaDistance._validate_factory_timeseries(y)

        no_python_callable = self._distance_factory(x, y, **kwargs)

        return no_python_callable

    def distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        **kwargs: dict,
    ) -> DistanceCallable:
        """Create a no_python distance alignment path.

        It should validate kwargs and then compile a no_python callable
        that takes (x, y) as parameters and returns a float that represents the distance
        between the two time series.

        ----------
        x: np.ndarray (2d array)
            First time series
        y: np.ndarray (2d array)
            Second time series
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
        NumbaDistance._validate_factory_timeseries(x)
        NumbaDistance._validate_factory_timeseries(y)

        no_python_callable = self._distance_alignment_path_factory(
            x, y, return_cost_matrix=return_cost_matrix, **kwargs
        )

        return no_python_callable

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

    @abstractmethod
    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict
    ) -> DistanceCallable:
        """Abstract method to create a no_python compiled distance.

        _distance_factory should validate kwargs and then compile a no_python callable
        that takes (x, y) as parameters and returns a float that represents the distance
        between the two time series.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series
        y: np.ndarray (2d array)
            Second time series
        kwargs: kwargs
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a float is then returned that represents the distance between x and y.
            This callable will be no_python compiled.
        """
        ...

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        **kwargs: dict,
    ) -> DistanceAlignmentPathCallable:
        """Abstract method to create a no_python compiled distance path computation.

        _distance_factory should validate kwargs and then compile a no_python callable
        that takes (x, y) as parameters and returns a float that represents the distance
        between the two time series.

        Parameters
        ----------
        x: np.ndarray (2d array)
            First time series
        y: np.ndarray (2d array)
            Second time series
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        kwargs: kwargs
            kwargs for the given distance metric

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]
            Callable where two, numpy 2d arrays are taken as parameters (x and y),
            a np.ndarray of tuples containing the optimal path and a float is also
             returned that represents the distance between x and y.
            This callable will be no_python compiled.
        """
        raise NotImplementedError("This distance does not support a path.")


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
    dist_instance: NumbaDistance
    # Distance path callable
    dist_alignment_path_func: AlignmentPathReturn = None
