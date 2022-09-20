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
from sktime.distances.distance_rework._base._base import MetricInfo
from sktime.distances.distance_rework._base import DistanceCallable
from sktime.distances.distance_rework._base._base_elastic import (
    AlignmentPathCallable, AlignmentPathCallableReturn, ElasticDistance
)
from sktime.distances._numba_utils import (
    _compute_pairwise_distance,
    _make_3d_series,
    _numba_to_timeseries,
    to_numba_timeseries,
)


_METRIC_INFOS = [
    MetricInfo(
        canonical_name="euclidean",
        aka={"euclidean", "ed", "euclid", "pythagorean"},
        dist_instance=_EuclideanDistance(),
    ),
    MetricInfo(
        canonical_name="erp",
        aka={"erp", "edit distance with real penalty"},
        dist_instance=_ErpDistance(),
    ),
    MetricInfo(
        canonical_name="edr",
        aka={"edr", "edit distance for real sequences"},
        dist_instance=_EdrDistance(),
    ),
    MetricInfo(
        canonical_name="lcss",
        aka={"lcss", "longest common subsequence"},
        dist_instance=_LcssDistance(),
    ),
    MetricInfo(
        canonical_name="squared",
        aka={"squared"},
        dist_instance=_SquaredDistance(),
    ),
    MetricInfo(
        canonical_name="dtw",
        aka={"dtw", "dynamic time warping"},
        dist_instance=_DtwDistance(),
    ),
    MetricInfo(
        canonical_name="ddtw",
        aka={"ddtw", "derivative dynamic time warping"},
        dist_instance=_DdtwDistance(),
    ),
    MetricInfo(
        canonical_name="wdtw",
        aka={"wdtw", "weighted dynamic time warping"},
        dist_instance=_WdtwDistance(),
    ),
    MetricInfo(
        canonical_name="wddtw",
        aka={"wddtw", "weighted derivative dynamic time warping"},
        dist_instance=_WddtwDistance(),
    ),
    MetricInfo(
        canonical_name="msm",
        aka={"msm", "move-split-merge"},
        dist_instance=_MsmDistance(),
    ),
    MetricInfo(
        canonical_name="twe",
        aka={"twe", "time warped edit"},
        dist_instance=_TweDistance(),
    ),
]

_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info) for info in _METRIC_INFOS for alias in info.aka)
_METRIC_CALLABLES = dict(
    (info.canonical_name, info.dist_instance.distance) for info in _METRIC_INFOS
)
_METRICS_NAMES = list(_METRICS.keys())

def _resolve_metric_to_instance(
        metric_str: str
) -> Union[BaseDistance, ElasticDistance]:
    if metric_str in _METRICS:
        return _METRICS[metric_str].dist_instance
    elif metric_str in _METRIC_ALIAS:
        return _METRIC_ALIAS[metric_str].dist_instance
    else:
        raise ValueError(f"Metric '{metric_str}' not found")

def distance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str,
        strategy: str = 'independent',
        **kwargs: Any,
) -> float:
    _metric_instance = _resolve_metric_to_instance(metric)
    return _metric_instance.distance(x, y, strategy=strategy, **kwargs)


def distance_factory(
        x: np.ndarray = None,
        y: np.ndarray = None,
        metric: str = 'euclidean',
        strategy: str = 'independent',
        **kwargs: Any,
) -> DistanceCallable:
    if x is None:
        x = np.zeros((1, 10))
    if y is None:
        y = np.zeros((1, 10))

    _metric_instance = _resolve_metric_to_instance(metric)
    return _metric_instance.distance_factory(x, y, strategy=strategy, **kwargs)


def distance_alignment_path(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = 'euclidean',
        strategy: str = 'independent',
        return_distance: bool = False,
        return_cost_matrix: bool = False,
        **kwargs: Any,
) -> AlignmentPathCallableReturn:
    _metric_instance = _resolve_metric_to_instance(metric)

    if not isinstance(_metric_instance, ElasticDistance):
        raise ValueError(f"Metric '{metric}' is not an elastic distance")


    return _metric_instance.alignment_path(
        x,
        y,
        strategy=strategy,
        return_distance=return_distance,
        return_cost_matrix=return_cost_matrix,
        **kwargs
    )

def distance_alignment_path_factory(
        x: np.ndarray = None,
        y: np.ndarray = None,
        metric: str = 'euclidean',
        strategy: str = 'independent',
        return_distance: bool = False,
        return_cost_matrix: bool = False,
        **kwargs: Any,
) -> AlignmentPathCallable:
    if x is None:
        x = np.zeros((1, 10))
    if y is None:
        y = np.zeros((1, 10))

    _metric_instance = _resolve_metric_to_instance(metric)

    if not isinstance(_metric_instance, ElasticDistance):
        raise ValueError(f"Metric '{metric}' is not an elastic distance")

    return _metric_instance.alignment_path_factory(
        x,
        y,
        strategy=strategy,
        return_distance=return_distance,
        return_cost_matrix=return_cost_matrix,
        **kwargs
    )


def pairwise_distance(
        x: np.ndarray,
        y: np.ndarray = None,
        metric: str = "euclidean",
        strategy: str = 'independent',
        **kwargs: Any,
) -> np.ndarray:
    _x = _make_3d_series(x)
    if y is None:
        y = x
    _y = _make_3d_series(y)
    symmetric = np.array_equal(_x, _y)
    if isinstance(metric, str):
        _metric_instance = _resolve_metric_to_instance(metric)
        example_x = _make_3d_series(x)[0]
        _dist_callable = _metric_instance.distance_factory(example_x, example_x, strategy=strategy, **kwargs)

        return _compute_pairwise_distance(_x, _y, symmetric, _dist_callable)
    else:
        return _compute_pairwise_distance(x, y, symmetric, metric)



if __name__ == '__main__':
    x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    y = np.array([11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])
    test = distance(x, y, 'euclidean')
    test1 = distance_factory(x, y, 'euclidean')(x, y)

    test2 = distance_alignment_path(x, y, 'dtw')
    test3 = distance_alignment_path_factory(x, y, 'dtw')(x, y)

    # test4, dist1 = distance_alignment_path(x, y, 'dtw', return_distance=True)
    # test5, dist2 = distance_alignment_path_factory(x, y, 'dtw', return_distance=True)(x, y)
    #
    # test6, cm1 = distance_alignment_path(x, y, 'dtw', return_cost_matrix=True)
    # test7, cm2 = distance_alignment_path_factory(x, y, 'dtw', return_cost_matrix=True)(x, y)
    #
    # test8, dist3, cm3 = distance_alignment_path(x, y, 'dtw', return_distance=True, return_cost_matrix=True)
    # test9, dist4, cm4 = distance_alignment_path_factory(x, y, 'dtw', return_distance=True, return_cost_matrix=True)(x, y)

    pw1 = pairwise_distance(x, y, 'euclidean')
    pw2 = pairwise_distance(x, y, 'dtw')

    joe = ''
