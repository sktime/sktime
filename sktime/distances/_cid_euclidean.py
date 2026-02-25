"""Complexity-Invariant Distance (CID)."""

__author__ = ["jgyasu"]

from typing import Any

import numpy as np

from sktime.distances.base import DistanceCallable, NumbaDistance


class _CIDDistance(NumbaDistance):
    """Complexity-Invariant Distance (CID)."""

    def _distance_factory(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> DistanceCallable:
        """Create a no_python compiled CID distance callable.

        Parameters
        ----------
        x, y : np.ndarray
            Time series of shape (d, m).
        kwargs : Any
            Unused (kept for API consistency).

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Numba-compiled CID distance.
        """
        from sktime.distances._cid_euclidean_numba import _numba_cid_distance

        return _numba_cid_distance
