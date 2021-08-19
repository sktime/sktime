# -*- coding: utf-8 -*-
__all__ = ["format_distance_series", "format_pairwise_matrix"]

from typing import Union, List, Tuple
import numpy as np
import pandas as pd

from sktime.utils.data_processing import (
    to_numpy_time_series,
    to_numpy_time_series_matrix,
)

# Types
SktimeSeries = Union[np.ndarray, pd.DataFrame, pd.Series, List]
SktimeMatrix = Union[np.ndarray, pd.DataFrame, List]


def format_distance_series(
    x: SktimeSeries, y: SktimeSeries
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method used to check and format series inputted to perform a distance between the
    two
    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or pd.Series or List
        First series to check and format
    y: np.ndarray or pd.Dataframe or pd.Series or List
        Second series to check and format
    Returns
    -------
    y: np.ndarray
        First series checked and formatted
    x: np.ndarray
        Second series checked and formatted
    """
    x = to_numpy_time_series(x)
    y = to_numpy_time_series(y)

    return x, y


def format_pairwise_matrix(x: SktimeMatrix) -> np.ndarray:
    """
    Method used to check and format the pairwise matrices passed to a pairwise distance
    operation

    Parameters
    ----------
    x: np.ndarray or pd.Dataframe or List
        Matrix of time series

    Returns
    -------
    x: np.ndarray
        Matrix of time series checked and formatted
    """
    x = to_numpy_time_series_matrix(x)

    return x
