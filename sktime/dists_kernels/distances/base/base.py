# -*- coding: utf-8 -*-
import numpy as np
from sktime.utils.validation.series import check_series
from sktime.utils.data_processing import to_numpy_time_series


class BaseDistance:
    def __call__(self, x, y, **kwargs):
        return self.distance(x, y, **kwargs)

    def distance(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Method used to get the distance between two time series

        Parameters
        ----------
        x: np.ndarray, pd.Dataframe, list
            First time series to compare
        y: np.ndarray, pd.Dataframe, list
            Second time series to compare

        Returns
        -------
            float that is the distance between the two time series
        """
        x = BaseDistance._check_distance_parameters(x)
        y = BaseDistance._check_distance_parameters(y)

        return self._distance(x, y, **kwargs)

    def _distance(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Implement distance")

    @staticmethod
    def _check_distance_parameters(x) -> np.ndarray:
        validated_series = check_series(x)
        return to_numpy_time_series(validated_series)
