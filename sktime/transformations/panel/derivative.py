# -*- coding: utf-8 -*-
"""Derivative transformer."""
import numpy as np

from sktime.distances._ddtw import average_of_slope
from sktime.transformations.base import BaseTransformer


class DerivativeTransformer(BaseTransformer):
    """Derivative transformer of a time series."""

    _tags = {
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit-in-transform": True,
    }

    def _transform(self, X: np.ndarray, y=None):
        derivative_X = []
        for val in X:
            derivative_X.append(average_of_slope(val))
        return np.array(derivative_X)
