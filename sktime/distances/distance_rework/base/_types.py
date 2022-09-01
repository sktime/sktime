# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]
__all__ = [
    "DistanceCostCallable",
    "DistanceAlignmentPathCallable",
    "DistanceFactoryCallable",
    "DistancePairwiseCallable",
    "ValidCallableTypes",
    "AlignmentPathReturn",
    "DerivativeCallable",
    "IndependentDistanceParameters",
    "DependentDistanceParameters",
    "DistanceReturn",
    "DistanceCostMatrixReturn",
    "DistancePathReturn",
    "DistancePathCostMatrixReturn",
]

from typing import Callable, List, Tuple, Union

import numpy as np
from numba import typeof

# Callable types
DerivativeCallable = Callable[[np.ndarray], np.ndarray]
DistanceCostCallable = (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],)
AlignmentPathReturn = Union[
    Tuple[List[Tuple], float], Tuple[List[Tuple], float, np.ndarray]
]
DistanceAlignmentPathCallable = Callable[[np.ndarray, np.ndarray], AlignmentPathReturn]
DistanceFactoryCallable = Callable[
    [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
]
DistancePairwiseCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]

ValidCallableTypes = Union[
    Callable[[np.ndarray, np.ndarray], float],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]],
]

# Numba types
_example_univariate = np.array([1.0, 2.0, 3.0])
_example_multivariate = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
_example_distance = 9.231342
_example_cost_matrix = np.array([[1.0, 2.0], [4.5, 3.2]])
_example_path = [(2, 1), (2, 4), (1, 4)]

IndependentDistanceParameters = typeof(_example_univariate, _example_univariate)
DependentDistanceParameters = typeof(_example_multivariate, _example_multivariate)

DistanceReturn = typeof(_example_distance)
DistanceCostMatrixReturn = typeof((_example_cost_matrix, _example_distance))
DistancePathReturn = typeof(_example_path, _example_distance)
DistancePathCostMatrixReturn = typeof(
    (_example_path, _example_distance, _example_cost_matrix)
)

