# -*- coding: utf-8 -*-
__author__ = ["Christopher Holder"]

import numpy as np
from typing import Set, Type, Optional
from dataclasses import dataclass

from sktime.metrics.distances.base._base import BaseDistance
from sktime.metrics.distances._dtw import _DtwDistance, _DtwDistanceCostMatrix


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding=None,
    sakoe_chiba_window_radius=None,
    itakura_max_slope=None,
):
    """
    Method used to check the incoming parameters and ensure they are the correct
    format for dtw

    x: np.ndarray
        first time series

    y: np.ndarray
        second time series

    lower_bounding: LowerBounding or int, defaults = NO_BOUNDING
        Lower bounding algorithm to use. The following describes the potential
        parameters:
        no bounding if LowerBounding.NO_BOUNDING or 1
        sakoe chiba bounding if LowerBounding.SAKOE_CHIBA or 2
        itakura parallelogram if LowerBounding.ITAKURA_PARALLELOGRAM or 3


    sakoe_chiba_window_radius: int, defaults = 2
        Integer that is the radius of the sakoe chiba window

    itakura_max_slope: float, defaults = 2.
        Gradient of the slope fo itakura

    Returns
    -------
        float that is the distance between the two time series
    """
    kwargs = {
        "lower_bounding": lower_bounding,
        "sakoe_chiba_window_radius": sakoe_chiba_window_radius,
        "itakura_max_slopes": itakura_max_slope,
    }
    return _DtwDistance().distance(x, y, **kwargs)


def dtw_with_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    lower_bounding=None,
    sakoe_chiba_window_radius=None,
    itakura_max_slope=None,
):
    """
    Method used to calculate the dtw distance between two time series

    Parameters
    ----------
    x: np.ndarray
        time series to find distance from

    y: np.ndarray
        time series to find distance from

    lower_bounding: LowerBounding or int, defaults = NO_BOUNDING
        Lower bounding algorithm to use. The following describes the potential
        parameters:
        no bounding if LowerBounding.NO_BOUNDING or 1
        sakoe chiba bounding if LowerBounding.SAKOE_CHIBA or 2
        itakura parallelogram if LowerBounding.ITAKURA_PARALLELOGRAM or 3


    sakoe_chiba_window_radius: int, defaults = 2
        Integer that is the radius of the sakoe chiba window

    itakura_max_slope: float, defaults = 2.
        Gradient of the slope fo itakura

    Returns
    -------
        Tuple[float, np.ndarray]. The first return value is the distance between
        the two time series and the second return value is the cost matrix used
        to generate the distance
    """

    kwargs = {
        "lower_bounding": lower_bounding,
        "sakoe_chiba_window_radius": sakoe_chiba_window_radius,
        "itakura_max_slopes": itakura_max_slope,
    }
    return _DtwDistanceCostMatrix().distance(x, y, **kwargs)


def dtw_pairwise(
    x: np.ndarray,
    y: np.ndarray = None,
    lower_bounding=None,
    sakoe_chiba_window_radius=None,
    itakura_max_slope=None,
):
    kwargs = {
        "lower_bounding": lower_bounding,
        "sakoe_chiba_window_radius": sakoe_chiba_window_radius,
        "itakura_max_slopes": itakura_max_slope,
    }
    return _DtwDistance().pairwise(x, y, **kwargs)


@dataclass(frozen=True)
class DistanceInfo:
    """
    Dataclass used to register valid distance metrics. This contains all the
    info you need for cdists and pdists and additional info such as str values
    for the distance metric
    """

    # Name of python distance function
    canonical_name: str
    # All aliases, including canonical_name
    aka: Set[str]
    # Base distance class
    distance_class: Optional[Type[BaseDistance]]


# Registry of implemented metrics:
DISTANCE_INFO = [
    DistanceInfo(
        canonical_name="dtw",
        aka={"dtw", "dynamic time warping"},
        distance_class=_DtwDistance,
    ),
    DistanceInfo(
        canonical_name="dtw cost matrix",
        aka={"dtw cost matrix", "dynamic time warping cost matrix"},
        distance_class=_DtwDistanceCostMatrix,
    ),
]


@dataclass(frozen=True)
class CDistWrapper:
    metric_name: str

    def __call__(self, x, y, **kwargs):
        pass


@dataclass(frozen=True)
class PDistWrapper:
    metric_name: str

    def __call__(self, x, y, **kwargs):
        pass


_scipy_distances = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
