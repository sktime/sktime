"""Euclidean distance."""
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any

import numpy as np

from sktime.distances.base import DistanceCallable, NumbaDistance


class _EuclideanDistance(NumbaDistance):
    """Euclidean distance between two time series."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled euclidean distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Requires equal length series.

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First time series.
        y: np.ndarray (1d or 2d array)
            Second times eries.
        kwargs: Any
            Extra kwargs. For euclidean there are none however, this is kept for
            consistency.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled euclidean distance callable.
        """
        from sktime.distances._euclidean_numba import _numba_euclidean_distance

        return _numba_euclidean_distance
