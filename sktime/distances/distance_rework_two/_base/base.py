# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import numpy as np
from numba import njit

# Types
DistanceCallableReturn = Union[float, Tuple[float, np.ndarray]]
DistanceCallable = Callable[[np.ndarray, np.ndarray], DistanceCallableReturn]
LocalDistanceCallable = Callable[[float, float], float]


class BaseDistance(ABC):
    """Base class for distance functions."""

    _numba_distance = False
    _cache = False
    _fastmath = False

    def distance_factory(self, strategy: str = "dependent"):
        """Create a distance function.

        Parameters
        ----------
        strategy : str, default="dependent"
            Strategy to use for distance calculation. Either "dependent" or
            "independent".

        Returns
        -------
        Callable
            Distance function.
        """
        distance_callable = self._convert_to_numba(self._distance)

        if strategy == "independent":
            if type(self)._independent_distance != BaseDistance._independent_distance:
                distance_callable = self._convert_to_numba(self._independent_distance)
            independent = self._independent_factory(distance_callable)
            _distance_callable = self._convert_to_numba(independent)
        else:
            _distance_callable = distance_callable

        preprocess_ts = self._preprocess_ts_factory(strategy=strategy)

        def _preprocess_distance_callable(x: np.ndarray, y: np.ndarray, *args):
            _x = preprocess_ts(x, *args)
            _y = preprocess_ts(y, *args)
            return _distance_callable(_x, _y, *args)

        preprocess_distance_callable = self._convert_to_numba(
            _preprocess_distance_callable
        )

        return self._result_process_factory(preprocess_distance_callable)

    def _result_process_factory(self, distance_callable: Callable):
        """Create a function to process the distance result.

        Parameters
        ----------
        distance_callable: Callable
            Distance function to process.

        Returns
        -------
        Callable
            Processed distance function.
        """
        if type(self)._result_process != BaseDistance._result_process:
            result_callback = self._convert_to_numba(self._result_process)

            def _result_process(x, y, *args):
                return result_callback(distance_callable(x, y, *args), *args)

            return self._convert_to_numba(_result_process)

        return distance_callable

    def _preprocess_ts_factory(self, strategy: str = "dependent"):
        """Create a function to preprocess time series.

        Parameters
        ----------
        strategy : str, default="dependent"
            Strategy to use for preprocessing time series. Either "dependent"
            or "independent".

        Returns
        -------
        Callable
            Preprocessing function.
        """

        def _convert_2d(x: np.ndarray, *args):
            if x.ndim == 1:
                # Use this instead of numpy because weird numba errors sometimes with
                # np.reshape
                x_size = x.shape[0]
                _process_x = np.zeros((1, x_size))
                _process_x[0] = x
                return_val = _process_x
                return _process_x
            return x

        convert_2d = self._convert_to_numba(_convert_2d)

        # If preprocessing overwritten then add it to preprocessing function
        if type(self)._preprocess_timeseries != BaseDistance._preprocess_timeseries:
            _preprocess_timeseries = self._convert_to_numba(self._preprocess_timeseries)

            def _convert_ts(x: np.ndarray, *args):
                _x = convert_2d(x, *args)
                return _preprocess_timeseries(_x, *args)

            convert_ts = self._convert_to_numba(_convert_ts)

        else:
            convert_ts = convert_2d

        if strategy == "independent":

            def _independent_preprocess_ts(x: np.ndarray, *args):
                _x = convert_ts(x)
                return _x.reshape((_x.shape[0], 1, _x.shape[1]))

            return self._convert_to_numba(_independent_preprocess_ts)

        return convert_ts

    def distance(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        """Compute the distance between two time series.

        Parameters
        ----------
        x : np.ndarray
            First time series.
        y : np.ndarray
            Second time series.
        **kwargs : dict
            Additional keyword arguments. The main kwarg is "strategy" which
            determines whether the distance is dependent or independent.

        Returns
        -------
        float
            Distance between x and y.
        """
        strategy = kwargs.get("strategy", "dependent")
        distance_callable = self.distance_factory(strategy=strategy)
        return distance_callable(x, y, *args)

    @staticmethod
    def _independent_distance(x: np.ndarray, y: np.ndarray, *args) -> float:
        """Independent distance between two time series.

        This is an optional method you can overload if you want to implement a specific
        way of computing the independent distance.

        Parameters
        ----------
        x : np.ndarray
            First time series.
        y : np.ndarray
            Second time series.
        **kwargs : dict
            Additional keyword arguments.
        """
        raise ValueError("Not implemented")

    def _convert_to_numba(self, func: Callable):
        """Check if needed to convert to numba function.

        Parameters
        ----------
        func : Callable
            Function to convert to numba.

        Returns
        -------
        Callable
            Numba function.
        """
        if self._numba_distance:
            return njit(cache=self._cache, fastmath=self._fastmath)(func)
        return func

    @staticmethod
    def _preprocess_timeseries(x, *args):
        """Change time series processing behaviour.

        Parameters
        ----------
        x : np.ndarray
            Time series.

        Returns
        -------
        np.ndarray
            Preprocessed time series.
        """
        return x

    @staticmethod
    def _result_process(result: float, *args):
        """Change the result of the distance calculation.

        Parameters
        ----------
        result : float
            Distance.

        Returns
        -------
        float
            Distance.
        """
        return result

    @staticmethod
    def _independent_factory(distance_callable: Callable):
        """Create independent distance callable.

        Parameters
        ----------
        distance_callable: Callable
            Independent distance callable.

        Returns
        -------
        Callable
            Independent distance callable.
        """

        def _distance_callable(_x: np.ndarray, _y: np.ndarray, *args):
            distance = 0
            for i in range(len(_x)):
                distance += distance_callable(_x[i], _y[i], *args)
            return distance

        return _distance_callable

    @staticmethod
    @abstractmethod
    def _distance(x: np.ndarray, y: np.ndarray, *args) -> float:
        """Distance between two time series.

        Parameters
        ----------
        x : np.ndarray
            First time series.
        y : np.ndarray
            Second time series.
        *args : list
            Additional arguments.
        """
        ...
