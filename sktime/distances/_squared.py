__author__ = ["chrisholder", "TonyBagnall"]


from typing import Any

import numpy as np

from sktime.distances.base import DistanceCallable, NumbaDistance


class _SquaredDistance(NumbaDistance):
    """Squared distance between two time series."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled Squared distance callable.

        Parameters
        ----------
        x: np.ndarray (1d or 2d array)
            First time series.
        y: np.ndarray (1d or 2d array)
            Second time series.
        kwargs: Any
            Extra kwargs. For squared there are none however, this is kept for
            consistency.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Squared distance callable.
        """
        from sktime.distances._squared_numba import _numba_squared_distance

        return _numba_squared_distance
