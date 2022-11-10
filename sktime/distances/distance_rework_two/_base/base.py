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

    def distance_factory(
        self,
        strategy: str = "dependent",
    ) -> DistanceCallable:
        """Create a distance callable.

        Parameters
        ----------
        strategy : str
            Strategy to use for distance calculation. Either "dependent" or
            "independent".

        Returns
        -------
        Callable
            Distance callable.
        """
        distance_callable = self._distance
        if strategy == "independent":
            independent_distance_callable = distance_callable
            if (
                not type(self)._independent_distance
                == BaseDistance._independent_distance
            ):
                independent_distance_callable = self._independent_distance
            distance_callable = self._create_independent_callable(
                independent_distance_callable
            )
        else:
            preprocess_ts = self._preprocess_timeseries
            if self._numba_distance:
                distance_callable = njit(cache=self._cache, fastmath=self._fastmath)\
                    (distance_callable)

            def _preprocess_callable(x, y, *args):
                _preprocess_x = preprocess_ts(x)
                _preprocess_y = preprocess_ts(y)
                return distance_callable(_preprocess_x, _preprocess_y, *args)

            if self._numba_distance:
                distance_callable = njit(cache=self._cache, fastmath=self._fastmath)\
                    (_preprocess_callable)

        return self._create_preprocess_distance_callable(distance_callable)

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
        ...

    def _create_preprocess_distance_callable(
        self, distance_callable: Callable
    ) -> DistanceCallable:
        """Create a distance callable that preprocesses the data.

        For the distance functions to work it is assumed a 2d array is given. This
        function inserts in checks and formatting to ensure the ts is in the
        correct format.

        Parameters
        ----------
        distance_callable: Callable
            Distance callable.

        Returns
        -------
        Callable
            Preprocessed distance callable.
        """

        def _preprocess_ts(x: np.ndarray):
            if x.ndim == 1:
                x_size = x.shape[0]
                _process_x = np.zeros((1, x_size))
                _process_x[0] = x
                return _process_x
            return x

        if self._numba_distance:
            _preprocess_ts = njit(cache=self._cache, fastmath=self._fastmath)(
                _preprocess_ts
            )

        def _formatted_ts_distance(_x: np.ndarray, _y: np.ndarray, *args):
            _x = _preprocess_ts(_x)
            _y = _preprocess_ts(_y)
            return distance_callable(_x, _y, *args)

        if self._numba_distance:
            _formatted_ts_distance = njit(cache=self._cache, fastmath=self._fastmath)(
                _formatted_ts_distance
            )

        return _formatted_ts_distance

    def _create_independent_callable(
        self, distance_callable: Callable
    ) -> DistanceCallable:
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
        if self._numba_distance:
            distance_callable = njit(cache=self._cache, fastmath=self._fastmath)(
                distance_callable
            )

        _distance_callable = self._independent_factory(distance_callable)

        if self._numba_distance:
            _distance_callable = njit(cache=self._cache, fastmath=self._fastmath)(
                _distance_callable
            )

        if type(self)._preprocess_timeseries != BaseDistance._preprocess_timeseries:
            _preprocess_timeseries = self._preprocess_timeseries

            def _distance_callable_with_user_preprocess_hook(
                _x: np.ndarray, _y: np.ndarray, *args
            ):
                _preprocess_x = _preprocess_timeseries(_x)
                _preprocess_y = _preprocess_timeseries(_y)
                return _distance_callable(_preprocess_x, _preprocess_y, *args)

            if self._numba_distance:
                _distance_callable_with_user_preprocess_hook = \
                    njit(cache=self._cache, fastmath=self._fastmath)(
                        _distance_callable_with_user_preprocess_hook
                    )
        else:
            _distance_callable_with_user_preprocess_hook = _distance_callable

        # If base distance not overwritten then need to add additional dim so that
        # dependent distance can be used.
        def _preprocessing_callable(_x: np.ndarray, _y: np.ndarray, *args):
            _x_preprocess = _x.reshape((_x.shape[0], 1, _x.shape[1]))
            _y_preprocess = _y.reshape((_y.shape[0], 1, _y.shape[1]))
            return _distance_callable_with_user_preprocess_hook(_x_preprocess, _y_preprocess, *args)

        if self._numba_distance:
            _preprocessing_callable = njit(cache=self._cache, fastmath=self._fastmath)(
                _preprocessing_callable
            )

        return _preprocessing_callable

    @staticmethod
    def _preprocess_timeseries(x):
        """Hook to change time series processing behaviour.

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
