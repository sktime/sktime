# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numba import njit

from sktime.distances.distance_rework_two._base import BaseDistance

LocalDistanceParam = Union[np.ndarray, float]


class BaseLocalDistance(BaseDistance, ABC):
    """Base class for distance with local distance."""

    @staticmethod
    @abstractmethod
    def _local_distance(x: np.ndarray, y: np.ndarray, *args) -> float:
        """Compute the local squared distance between two points.

        Parameters
        ----------
        x: float
            First value
        y: float
            Second value

        Returns
        -------
        float
            Distance between x and y.
        """
        ...

    def distance(self, x: LocalDistanceParam, y: LocalDistanceParam, *args, **kwargs):
        """Compute the squared distance between two time series or points.

        Parameters
        ----------
        x: np.ndarray or float
            First time series or point.
        y: np.ndarray or float
            Second time series or point.
        args: list
            Additional arguments.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            Squared distance between x and y
        """
        if isinstance(x, float) and isinstance(y, float):
            kwargs["strategy"] = "local"
        return super().distance(x, y, *args, **kwargs)

    def distance_factory(
        self,
        strategy: str = "dependent",
    ):
        """Create a distance callable.

        Parameters
        ----------
        strategy : str
            Strategy to use for distance calculation. Either "dependent",
            "independent" or "local".
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Callable
            Distance callable.
        """
        if strategy == "local":

            _local_distance = self._local_distance

            if self._numba_distance:
                _local_distance = njit(cache=self._cache, fastmath=self._fastmath)(
                    self._local_distance
                )

            return _local_distance

        return super().distance_factory(strategy=strategy)

    def local_distance(self, x: float, y: float, *args, **kwargs) -> float:
        """Compute the local squared distance between two points.

        Parameters
        ----------
        x: float
            First value
        y: float
            Second value

        Returns
        -------
        float
            Squared distance between x and y
        """
        distance_callable = self.distance_factory(strategy="local")
        return distance_callable(x, y, *args)
