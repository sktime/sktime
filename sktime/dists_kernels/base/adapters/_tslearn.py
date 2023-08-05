# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for tslearn distances and kernels."""
import pandas

__all__ = ["_TslearnPwTrafoAdapter"]
__author__ = ["fkiraly"]


class _GeneralisedStatsForecastAdapter:
    """Base adapter mixin for tslearn distances and kernels."""

    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "X_inner_mtype": "numpy3D",  # which mtype is used internally in _transform?
        "fit_is_empty": True,  # is "fit" empty? Yes, for all pairwise transforms
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
        "python_dependencies": ["tslearn"],
    }
