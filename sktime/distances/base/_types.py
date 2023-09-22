__author__ = ["chrisholder", "TonyBagnall"]
__all__ = [
    "DistanceCallable",
    "DistanceAlignmentPathCallable",
    "DistanceFactoryCallable",
    "DistancePairwiseCallable",
    "ValidCallableTypes",
    "AlignmentPathReturn",
]

from typing import Callable, List, Tuple, Union

import numpy as np

# Callable types
DistanceCallable = (Callable[[np.ndarray, np.ndarray], float],)
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
