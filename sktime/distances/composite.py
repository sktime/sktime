# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Union

import numpy as np

from sktime.distances._distance import _METRIC_INFOS
from sktime.distances._numba_utils import to_numba_timeseries
from sktime.distances._resolve_metric import _resolve_metric
from sktime.distances.base import NumbaDistance


class CompositeDistance:
    def __init__(
        self,
        metrics: List[
            Union[
                str,
                Callable[
                    [np.ndarray, np.ndarray, dict],
                    Callable[[np.ndarray, np.ndarray], float],
                ],
                Callable[[np.ndarray, np.ndarray], float],
                NumbaDistance,
            ]
        ],
        weights: List[List],
        variables: List[str],
    ):
        self.metrics = metrics
        self.weights = weights
        self.variables = variables

    def __call__(self, x: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> float:
        composite_distance_val = 0
        for metric, weight, variable in zip(self.metrics, self.weights, self.variables):
            for var in variable:
                _x = to_numba_timeseries(x[var])
                _y = to_numba_timeseries(y[var])
                _metric_callable = _resolve_metric(metric, _x, _y, _METRIC_INFOS)
                composite_distance_val += weight * _metric_callable(_x, _y)
        return composite_distance_val
