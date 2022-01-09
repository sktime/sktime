# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np


def mean_average(X: np.ndarray) -> np.ndarray:
    """Take mean average of time series.

    TODO: Finish docstring
    """
    return X.mean(axis=0)


def dba_average(X: np.ndarray) -> np.ndarray:
    """Take dba average of time series.

    TODO: Finish docstring
    """
    pass


_AVERAGE_DICT = {"mean": mean_average, "dba": dba_average}


def resolve_average_callable(averaging_method: [str, Callable]) -> Callable:
    """Resolve a string or callable to a averaging callable.

    TODO: Finish docstring
    """
    if isinstance(averaging_method, str):
        if averaging_method not in _AVERAGE_DICT:
            raise ValueError(
                "averaging_method string is invalid. Please use one of the" "following",
                _AVERAGE_DICT.keys(),
            )
        return _AVERAGE_DICT[averaging_method]

    return averaging_method
