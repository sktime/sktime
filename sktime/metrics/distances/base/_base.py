# -*- coding: utf-8 -*-

__author__ = ["Christopher Holder", "fkiraly"]

import numpy as np
from typing import Any
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

    def pairwise(self, x: np.ndarray, y: np.ndarray = None, **kwargs: Any):
        x = to_numpy_time_series_matrix(x)

        if y is None:
            y = np.copy(x)
            kwargs["symmetric"] = True
        else:
            y = to_numpy_time_series_matrix(y)
            kwargs["symmetric"] = False

        return self._pairwise(x, y, **kwargs)

    def _pairwise(self, x, y, **kwargs):
        return self._pairwise_threading(x, y, **kwargs)

    def _pairwise_threading(self, x, y, **kwargs):
        pairwise_matrix = np.zeros((x.shape[0], y.shape[0]))
        is_symmetric = kwargs.get("symmetric")
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                if is_symmetric and j < i:
                    pairwise_matrix[i, j] = pairwise_matrix[j, i]
                pairwise_matrix[i, j] = self.distance(x[i], y[j], **kwargs)

        return pairwise_matrix
