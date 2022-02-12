# -*- coding: utf-8 -*-
"""Factory module that adds 3D vectorization to all exported distances.

Applies pairwise_distance to all distanes exported by module, so
    exported distances function with 1D, 2D and 3D input.
"""

import numpy as np

from sktime.distances._distance import ALL_DISTANCES, pairwise_distance

assign = [(x.__name__, i) for i, x in enumerate(ALL_DISTANCES)]


def _vectorize_distance(distance):
    """Turn 2D distance in one that can handle 1D, 2D and 3D input."""

    def _3D_distance(X, X2, **params):
        msg = "X and X2 must be np.ndarray, of dim 1, 2, or 3"
        if not isinstance(X, np.ndarray) or not isinstance(X2, np.ndarray):
            raise TypeError(msg)
        if X.ndim > 3 or X2.ndim > 3 or X.ndim < 1 or X2.ndim < 1:
            raise TypeError(msg)
        if X.ndim < 3 and X2.ndim < 3:
            return distance(X, X2, **params)
        else:
            return pairwise_distance(X, X2, metric=distance, **params)

    return _3D_distance


for a in assign:
    # wrap all distances in pairwise_distance to add 3D compatibility
    exec("%s = _vectorize_distance(ALL_DISTANCES[%d])" % a)
    # copy the docstring over
    exec("%s.__doc__ = ALL_DISTANCES[%d].__doc__" % a)
