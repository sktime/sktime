"""TDE classifiers - numba methods."""

__author__ = ["MatthewMiddlehurst"]

from sktime.utils.numba.njit import njit
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("numba", severity="none"):
    from numba import types


@njit(fastmath=True, cache=True)
def _histogram_intersection_dict(first, second):
    sim = 0
    for word, val_a in first.items():
        val_b = second.get(word, types.uint32(0))
        sim += min(val_a, val_b)
    return sim
