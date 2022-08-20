# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

"""Factory module that adds 3D vectorization to all exported distances.

Applies pairwise_distance to all distances exported by module, so
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


def _extend_docstring_3d(docstring):
    """Change docstring to include 3D numpy support."""
    # the indentation must be this way, since tabs are included in the string
    #   if one more tab is added, the replacement is not correctly done
    to_replace = """x: np.ndarray (1d or 2d array)
        First time series.
    y: np.ndarray (1d or 2d array)
        Second time series.
    """  # noqa
    replace_by = """x: np.ndarray (1d, 2d, or 3d array)
        First time series or panel of time series.
        Indices are (n_instances, n_variables, n_series).
        If index is not present, n_variables=1 resp n_series=1 is assumed.
    y: np.ndarray (1d or 2d, or 3d array)
        Second time series or panel of time series.
        Indices are (n_instances, n_variables, n_series).
        If index is not present, n_variables=1 resp n_series=1 is assumed.
    """  # noqa
    docstring = docstring.replace(to_replace, replace_by)

    to_replace = """Returns
    -------
    float
    """  # noqa
    replace_by = """Returns
    -------
    float if x, y are both 1d or 2d
        distance between single series x and y
    2d np.ndarray if x and y are both 3d
        (i, j)-th entry is distance between i-th instance in x and j-th in y
    """  # noqa
    docstring = docstring.replace(to_replace, replace_by)

    return docstring


for a in assign:
    # wrap all distances in pairwise_distance to add 3D compatibility
    exec("%s = _vectorize_distance(ALL_DISTANCES[%d])" % a)
    # copy the docstring over
    exec("%s.__doc__ = _extend_docstring_3d(ALL_DISTANCES[%d].__doc__)" % a)
