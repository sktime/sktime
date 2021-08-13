# -*- coding: utf-8 -*-

__author__ = ["Christopher Holder", "fkiraly"]

import numpy as np
from typing import Any, Union, Callable
from sktime.utils.data_processing import (
    to_numpy_time_series,
    to_numpy_time_series_matrix,
)


class BaseDistance:
    def __call__(self, x, y, **kwargs):
        return self.distance(x, y, **kwargs)

    def distance(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        """
        Method used to get the distance between two time series

        Parameters
        ----------
        x: np.ndarray, pd.Dataframe, list
            First time series to compare
        y: np.ndarray, pd.Dataframe, list
            Second time series to compare
        **kwargs: Any
            Kwargs to be used for specific distance measure

        Returns
        -------
            float that is the distance between the two time series
        """
        x = to_numpy_time_series(x)
        y = to_numpy_time_series(y)

        return self._distance(x, y, **kwargs)

    def _distance(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        raise NotImplementedError("Implement distance")


class BaseAverage:
    def __call__(self, x, y, **kwargs):
        return self.average(x, y, **kwargs)

    def average(self, series_matrix: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Method used to get the average between two time series

        Parameters
        ----------
        series_matrix: np.ndarray
            matrix to derive the average value from
        **kwargs: Any
            Kwargs to be used for specific distance measure

        Returns
        -------
        np.ndarray
            Average of the series
        """
        series_matrix = to_numpy_time_series_matrix(series_matrix)

        return self._average(series_matrix, **kwargs)

    def _average(self, series_matrix: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError("Implement average method")


class BasePairwise:
    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray = None,
        metric: Union[str, BaseDistance, Callable] = "euclidean",
        **kwargs: Any
    ):
        return self.pairwise(x, y, metric, **kwargs)

    def pairwise(
        self,
        x: np.ndarray,
        y: np.ndarray = None,
        metric: Union[str, BaseDistance, Callable] = "euclidean",
        **kwargs: Any
    ):
        x = to_numpy_time_series_matrix(x)

        if y is None:
            y = np.copy(x, copy=True)
        else:
            y = to_numpy_time_series_matrix(y)

        return self._pairwise(x, y, metric, **kwargs)

    def _pairwise(self, x, y, metric, **kwargs):
        raise NotImplementedError("Implement pairwise method")
