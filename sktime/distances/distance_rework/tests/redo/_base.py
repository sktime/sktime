from typing import Callable, Tuple, Optional, Union
from abc import ABC, abstractmethod
from numba import njit

import numpy as np

__all__ = ['BaseDistance', 'numbadistance', 'DistanceCallable']

DistanceCallableReturn = Union[float, Tuple[float, np.ndarray]]

DistanceCallable = Callable[
    [np.ndarray, np.ndarray], DistanceCallableReturn
]


def numbadistance(*args, **kwargs):
    def wrapper(func):
        distance_type = args[0]
        cache = False
        return_cost_matrix = False
        fastmath = False
        if 'cache' in kwargs:
            cache = kwargs['cache']
        if 'fastmath' in kwargs:
            fastmath = kwargs['fastmath']
        if 'return_cost_matrix' in kwargs:
            return_cost_matrix = kwargs['return_cost_matrix']
        if return_cost_matrix is True:
            signature = 'Tuple((float64, float64[:, :]))(float64[:], float64[:])'
            if distance_type == "dependent":
                signature = 'Tuple((float64, float64[:, :]))' \
                            '(float64[:, :], float64[:, :])'
        else:
            signature = '(float64)(float64[:], float64[:])'
            if distance_type == "dependent":
                signature = '(float64)(float64[:, :], float64[:, :])'

        return njit(signature, cache=cache, fastmath=fastmath)(func)

    return wrapper


def format_time_series(*args, **kwargs):
    def wrapper(func):
        example_x = args[0]
        numba_distance = args[1]
        cache = False
        if 'cache' in kwargs:
            cache = kwargs['cache']
        fastmath = False
        if 'fastmath' in kwargs:
            fastmath = kwargs['fastmath']

        if example_x.ndim < 2:
            def _format(_x: np.ndarray):
                x_size = _x.shape[0]
                _process_x = np.zeros((x_size, 1))
                for i in range(0, x_size):
                    _process_x[i, :] = _x[i]
                return func(_process_x)

            if numba_distance is True:
                return njit(
                    '(float64[:])(float64[:, :])',
                    cache=cache,
                    fastmath=fastmath
                )(_format)
            return _format
        else:
            return func

    return wrapper


class BaseDistance(ABC):
    """Base class for distances.

    The base class that is used to create distance functions that are used in sktime.


    _has_cost_matrix : bool, default = False
        If the distance produces a cost matrix.
    _numba_distance : bool, default = False
        If the distance is compiled to numba.
    _cache : bool, default = False
        If the numba distance function should be cached.
    _fastmath : bool, default = False
        If the numba distance function should be compiled with fastmath.
    """
    _has_cost_matrix = False
    _numba_distance = False
    _cache = True
    _fastmath = False

    def distance_factory(
            self,
            x: np.ndarray,
            y: np.ndarray,
            strategy: str,
            return_cost_matrix: bool = False,
            **kwargs
    ) -> DistanceCallable:
        """Factory method for distance functions.

        Parameters
        ----------
        x : np.ndarray
            First time series.
        y : np.ndarray
            Second time series.
        strategy: str
            The strategy to use for the distance function. Either 'independent' or
            'dependent'.
        return_cost_matrix : bool, default = False
            If the distance function should return a cost matrix.
        kwargs : dict
            Additional keyword arguments.
        """
        if strategy == 'independent' or \
                type(self)._dependent_distance == BaseDistance._dependent_distance:
            strategy = 'independent'  # Do this in case dependent is not implemented.
            initial_distance_callable = self._independent_distance(
                x, y, **kwargs
            )
        else:
            initial_distance_callable = self._dependent_distance(
                x, y, **kwargs
            )

        if self._numba_distance is True:
            # This uses custom decorator defined above to compile to numba.
            initial_distance_callable = numbadistance(
                strategy,
                cache=self._cache,
                fastmath=self._fastmath,
                return_cost_matrix=self._has_cost_matrix
            )(initial_distance_callable)

        if return_cost_matrix is False:
            cost_matrix_callable = initial_distance_callable
            if self._has_cost_matrix is True:
                def _cost_matrix_callable(_x: np.ndarray, _y: np.ndarray):
                    return initial_distance_callable(_x, _y)[0]

                if self._numba_distance is True:
                    cost_matrix_callable = numbadistance(
                        strategy,
                        cache=self._cache,
                        fastmath=self._fastmath,
                        return_cost_matrix=False
                    )(_cost_matrix_callable)
                else:
                    cost_matrix_callable = _cost_matrix_callable

            callable_distance = cost_matrix_callable
        else:
            callable_distance = initial_distance_callable

        final_distance_callable = callable_distance

        if strategy == 'independent':

            if return_cost_matrix is True:
                def _independent_distance_wrapper(_x, _y):
                    total = 0
                    cost_matrix = np.zeros((_x.shape[1], _y.shape[1]))
                    for i in range(x.shape[0]):
                        curr_dist, curr_cost_matrix = callable_distance(_x[i], _y[i])
                        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
                        total += curr_dist
                    return total, cost_matrix
            else:
                def _independent_distance_wrapper(_x, _y):
                    total = 0
                    for i in range(x.shape[0]):
                        curr_dist = callable_distance(_x[i], _y[i])
                        total += curr_dist
                    return total

            if self._numba_distance is True:
                final_distance_callable = numbadistance(
                    'dependent',
                    # Marked as dependent because it takes 2d array as argument
                    cache=self._cache,
                    fastmath=self._fastmath,
                    return_cost_matrix=return_cost_matrix
                )(_independent_distance_wrapper)
            else:
                final_distance_callable = _independent_distance_wrapper

        result_callback = self._result_distance_callback()

        if self._numba_distance is True:
            result_callback = \
                njit(cache=self._cache, fastmath=self._fastmath)(result_callback)

        if return_cost_matrix is True:
            # This cant infer the type properly so probs need two seperate callbacks, one for the cost matrix one for distance
            def result_callback_callable(_x, _y):
                distance, cost_matrix = final_distance_callable(_x, _y)
                distance = result_callback(distance)
                return distance, cost_matrix
        else:
            def result_callback_callable(_x: np.ndarray, _y: np.ndarray):
                distance = final_distance_callable(_x, _y)
                return result_callback(distance)

        if self._numba_distance is True:
            result_callback_callable = numbadistance(
                'dependent',
                cache=self._cache,
                fastmath=self._fastmath,
                return_cost_matrix=return_cost_matrix
            )(result_callback_callable)

        _preprocess_time_series = self._preprocessing_time_series_callback(**kwargs)

        if self._numba_distance is True:
            _preprocess_time_series = njit(
                '(float64[:, :])(float64[:, :])', cache=self._cache,
                fastmath=self._fastmath
            )(_preprocess_time_series)

        _preprocess_time_series = format_time_series(
            x, self._numba_distance, cache=self._cache, fastmath=self._fastmath
        )(_preprocess_time_series)

        def _preprocessed_distance_callable(_x: np.ndarray, _y: np.ndarray):
            _preprocess_x = _preprocess_time_series(_x)
            _preprocess_y = _preprocess_time_series(_y)
            return result_callback_callable(_preprocess_x, _preprocess_y)

        if self._numba_distance is True:
            _preprocessed_distance_callable = numbadistance(
                'dependent',
                cache=self._cache,
                fastmath=self._fastmath,
                return_cost_matrix=return_cost_matrix
            )(_preprocessed_distance_callable)

        return _preprocessed_distance_callable

    def distance(
            self,
            x: np.ndarray,
            y: np.ndarray,
            strategy: str,
            return_cost_matrix: bool = False,
            **kwargs: dict
    ) -> DistanceCallableReturn:
        distance_callable = self.distance_factory(
            x, y, strategy, return_cost_matrix, **kwargs
        )

        return distance_callable(x, y)

    def independent_distance(
            self,
            x: np.ndarray,
            y: np.ndarray,
            return_cost_matrix: bool = False,
            **kwargs: dict
    ) -> DistanceCallableReturn:
        return self.distance(x, y, "independent", return_cost_matrix, **kwargs)

    def dependent_distance(
            self,
            x: np.ndarray,
            y: np.ndarray,
            return_cost_matrix: bool = False,
            **kwargs: dict
    ) -> DistanceCallableReturn:
        return self.distance(x, y, "dependent", return_cost_matrix, **kwargs)

    def _result_distance_callback(self) -> Callable[[float], float]:
        def _result_callback(distance: float) -> float:
            return distance

        return _result_callback

    def _preprocessing_time_series_callback(
            self, **kwargs
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Preprocess the time series before passed to the distance.

        All of the kwargs are given so they can be used as constants inside the
        return function.

        Parameters
        ---------
        **kwargs: dict
            Keyword arguments for the given distance.
        """

        def _preprocessing_callback(_x: np.ndarray) -> np.ndarray:
            return _x

        return _preprocessing_callback

    def _dependent_distance(self, x: np.ndarray, y: np.ndarray,
                            **kwargs) -> DistanceCallable:
        raise NotImplementedError("This method is an optional implementation. It will"
                                  "default to using the independent distance.")
        # return self.distance_factory(
        #     x, y, 'independent', self._has_cost_matrix, **kwargs
        # )

    @abstractmethod
    def _independent_distance(self, x: np.ndarray, y: np.ndarray,
                              **kwargs) -> DistanceCallable:
        ...


class Example(BaseDistance):
    _numba_distance = True
    _has_cost_matrix = True

    def _independent_distance(
            self, x: np.ndarray, y: np.ndarray, **kwargs
    ) -> DistanceCallable:
        def independent_example(_x, _y):
            return 1.2345, np.zeros((_x.shape[0], _y.shape[0])),

        return independent_example

    def _dependent_distance(
            self, x: np.ndarray, y: np.ndarray, **kwargs
    ) -> DistanceCallable:
        def dependent_example(_x, _y):
            return 1.2345, np.zeros((_x.shape[0], _y.shape[0])),

        return dependent_example


if __name__ == '__main__':
    test = Example()

    x = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 3.0]])
    y = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 3.0]])

    # ind = test.independent_distance(x, x)
    dep_dist, cm_dep = test.dependent_distance(x, y, return_cost_matrix=True)
    dep_only_dis = test.dependent_distance(x, x)

    indep_dist, indep_cm = test.independent_distance(x, y, return_cost_matrix=True)
    indep_only_dis = test.independent_distance(x, x)

    # ind2, ind_cm = test.independent_distance(x, x, return_cost_matrix=True)
    # dep2, dep_cm = test.dependent_distance(y, y, return_cost_matrix=True)
    joe = ''
