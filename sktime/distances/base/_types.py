# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]
__all__ = [
    "DistanceCallable",
    "DistancePathCallable",
    "DistanceFactoryCallable",
    "DistancePairwiseCallable",
    "ValidCallableTypes",
]

from typing import Callable, List, Tuple, Union

import numpy as np

# Callable types
DistanceCallable = (Callable[[np.ndarray, np.ndarray], float],)
DistancePathCallable = Callable[
    [np.ndarray, np.ndarray], Union[Tuple[List, float], Tuple[List, float, np.ndarray]]
]
DistanceFactoryCallable = Callable[
    [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
]
DistancePairwiseCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]

ValidCallableTypes = Union[
    Callable[[np.ndarray, np.ndarray], float],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]],
]
