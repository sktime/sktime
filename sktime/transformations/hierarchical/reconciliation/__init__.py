"""Hierarchical reconciliation transformations."""

from .bottom_up import BottomUpReconciler
from .forecast_proportions import ForecastProportions
from .full_hierarchy import FullHierarchyReconciler, NonNegativeFullHierarchyReconciler
from .middle_out import MiddleOutReconciler
from .topdown_share import TopdownShareReconciler

__all__ = [
    "MiddleOutReconciler",
    "TopdownShareReconciler",
    "BottomUpReconciler",
    "FullHierarchyReconciler",
    "NonNegativeFullHierarchyReconciler",
    "ForecastProportions",
]


METHOD_MAP = {
    "bu": BottomUpReconciler(),
    "ols": FullHierarchyReconciler(),
    "ols:nonneg": NonNegativeFullHierarchyReconciler(),
    "wls_str": FullHierarchyReconciler("wls_str"),
    "wls_str:nonneg": NonNegativeFullHierarchyReconciler("wls_str"),
    "td_fcst": ForecastProportions(),
    "td_share": TopdownShareReconciler(),
}
