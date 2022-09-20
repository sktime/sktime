from typing import Any, Callable, Union


import numpy as np
from numba import njit

from sktime.distances.distance_rework import (
    BaseDistance,
    _DdtwDistance,
    _DtwDistance,
    _EdrDistance,
    _ErpDistance,
    _EuclideanDistance,
    _LcssDistance,
    _MsmDistance,
    _SquaredDistance,
    _TweDistance,
    _WddtwDistance,
    _WdtwDistance,
)
# from sktime.distances.distance_rework._base import MetricInfo

# def distance(
#         x: np.ndarray,
#         y: np.ndarray,
#         metric: str,
#         **kwargs: Any,
# ) -> float:
#     _metric_callable = _resolve_metric_to_factory(
#         metric, x, y, _METRIC_INFOS, **kwargs
#     )
#
#     return _metric_callable(x, y)
#
#
# def distance_factory(
#         x: np.ndarray = None,
#         y: np.ndarray = None,
#         metric: str = "euclidean",
#         **kwargs: Any,
# ) -> DistanceCallable:
#     if x is None:
#         x = np.zeros((1, 10))
#     if y is None:
#         y = np.zeros((1, 10))
#
#     dist_callable = _resolve_metric_to_factory(metric, x, y, _METRIC_INFOS, **kwargs)
#
#     return dist_callable
#
#
# def pairwise_distance(
#         x: np.ndarray,
#         y: np.ndarray = None,
#         metric: str = "euclidean",
#         **kwargs: Any,
# ) -> np.ndarray:
#     _x = _make_3d_series(x)
#     if y is None:
#         y = x
#     _y = _make_3d_series(y)
#     symmetric = np.array_equal(_x, _y)
#     _metric_callable = _resolve_metric_to_factory(
#         metric, _x[0], _y[0], _METRIC_INFOS, **kwargs
#     )
#     return _compute_pairwise_distance(_x, _y, symmetric, _metric_callable)
#
#
# def distance_alignment_path(
#         x: np.ndarray,
#         y: np.ndarray,
#         metric: str = 'euclidean',
#         return_cost_matrix: bool = False,
#         **kwargs: Any,
# ) -> AlignmentPathReturn:
#     _dist_instance = _resolve_dist_instance(metric, x, y, _METRIC_INFOS, **kwargs)
#
#     return _dist_instance.distance_alignment_path(
#         _x, _y, return_cost_matrix=return_cost_matrix, **kwargs
#     )
#
#
# def distance_alignment_path_factory(
#         x: np.ndarray,
#         y: np.ndarray,
#         metric: Union[
#             str,
#             Callable[
#                 [np.ndarray, np.ndarray, dict], Callable[[np.ndarray, np.ndarray], float]
#             ],
#             Callable[[np.ndarray, np.ndarray], float],
#             NumbaDistance,
#         ],
#         return_cost_matrix: bool = False,
#         **kwargs: Any,
# ) -> DistanceAlignmentPathCallable:
#
#     dist_instance = _resolve_dist_instance(metric, x, y, _METRIC_INFOS, **kwargs)
#     dist_callable = dist_instance.distance_alignment_path_factory(
#         x, y, return_cost_matrix, **kwargs
#     )
#
#     return dist_callable
#
#
# _METRIC_INFOS = [
#     MetricInfo(
#         canonical_name="euclidean",
#         aka={"euclidean", "ed", "euclid", "pythagorean"},
#         dist_func=_EuclideanDistance.distance,
#         dist_instance=_EuclideanDistance(),
#     ),
#     MetricInfo(
#         canonical_name="erp",
#         aka={"erp", "edit distance with real penalty"},
#         dist_func=_ErpDistance.distance,
#         dist_instance=_ErpDistance(),
#     ),
#     MetricInfo(
#         canonical_name="edr",
#         aka={"edr", "edit distance for real sequences"},
#         dist_func=_ErpDistance.distance,
#         dist_instance=_EdrDistance(),
#     ),
#     MetricInfo(
#         canonical_name="lcss",
#         aka={"lcss", "longest common subsequence"},
#         dist_func=_LcssDistance.distance,
#         dist_instance=_LcssDistance(),
#     ),
#     MetricInfo(
#         canonical_name="squared",
#         aka={"squared"},
#         dist_func=_SquaredDistance.distance,
#         dist_instance=_SquaredDistance(),
#     ),
#     MetricInfo(
#         canonical_name="dtw",
#         aka={"dtw", "dynamic time warping"},
#         dist_func=_DtwDistance.distance,
#         dist_instance=_DtwDistance(),
#     ),
#     MetricInfo(
#         canonical_name="ddtw",
#         aka={"ddtw", "derivative dynamic time warping"},
#         dist_func=_DdtwDistance.distance,
#         dist_instance=_DdtwDistance(),
#     ),
#     MetricInfo(
#         canonical_name="wdtw",
#         aka={"wdtw", "weighted dynamic time warping"},
#         dist_func=_WdtwDistance.distance,
#         dist_instance=_WdtwDistance(),
#     ),
#     MetricInfo(
#         canonical_name="wddtw",
#         aka={"wddtw", "weighted derivative dynamic time warping"},
#         dist_func=_WddtwDistance.distance,
#         dist_instance=_WddtwDistance(),
#     ),
#     MetricInfo(
#         canonical_name="msm",
#         aka={"msm", "move-split-merge"},
#         dist_func=_MsmDistance.distance,
#         dist_instance=_MsmDistance(),
#     ),
#     MetricInfo(
#         canonical_name="twe",
#         aka={"twe", "time warped edit"},
#         dist_func=_TweDistance.distance,
#         dist_instance=_TweDistance(),
#     ),
# ]
#
# _METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
# _METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)
# _METRIC_CALLABLES = dict(
#     (info.canonical_name, info.dist_func) for info in _METRIC_INFOS
# )
# _METRICS_NAMES = list(_METRICS.keys())
