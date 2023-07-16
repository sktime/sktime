__author__ = ["chrisholder", "TonyBagnall"]

from typing import Any, List, Tuple

import numpy as np

from sktime.distances.base import DistanceCallable, NumbaDistance
from sktime.distances.base._types import DistanceAlignmentPathCallable


class _DtwDistance(NumbaDistance):
    r"""Dynamic time warping (dtw) between two time series.

    DTW is the most widely researched and used elastic distance measure. It mitigates
    distortions in the time axis by realligning (warping) the series to best match
    each other. A good background into DTW can be found in [1]. For two series
    :math:'\mathbf{a}=\{a_1,a_2,\ldots,a_m\}' and :math:'\mathbf{b}=\{b_1,b_2,\ldots,
    b_m\}',  (assumed equal length for simplicity), DTW first calculates  :math:'M(
    \mathbf{a},\mathbf{b})', the :math:'m \times m'
    pointwise distance matrix between series :math:'\mathbf{a}' and :math:'\mathbf{b}',
    where :math:'M_{i,j}=   (a_i-b_j)^2'. A warping path
    .. math::  P=<(e_1,f_1),(e_2,f_2),\ldots, (e_s,f_s)>
    is a set of pairs of indices that  define a traversal of matrix :math:'M'. A
    valid warping path must start at location :math:'(1,1)' and end at point :math:'(
    m,m)' and not backtrack, i.e. :math:'0 \leq e_{i+1}-e_{i} \leq 1' and :math:'0
    \leq f_{i+1}- f_i \leq 1' for all :math:'1< i < m'. The DTW distance between
    series is the path through :math:'M' that minimizes the total distance. The
    distance for any path :math:'P' of length :math:'s' is
    .. math::  D_P(\mathbf{a},\mathbf{b}, M) =\sum_{i=1}^s M_{e_i,f_i}.
    If :math:'\mathcal{P}' is the space of all possible paths, the DTW path :math:'P^*'
    is the path that has the minimum distance, hence the DTW distance between series is
    .. math::  d_{dtw}(\mathbf{a}, \mathbf{b}) =D_{P*}(\mathbf{a},\mathbf{b}, M).
    The optimal warping path $P^*$ can be found exactly through a dynamic programming
    formulation. This can be a time consuming operation, and it is common to put a
    restriction on the amount of warping allowed. This is implemented through
    the bounding_matrix structure, that supplies a mask for allowable warpings.
    Common bounding strategies include the Sakoe-Chiba band [2] and the Itakura
    parallelogram [3]. The Sakoe-Chiba band creates a warping path window that has
    the same width along the diagonal of :math:'M'. The Itakura paralleogram allows
    for less warping at the start or end of the series than in the middle.

    References
    ----------
    .. [1] Ratanamahatana C and Keogh E.: Three myths about dynamic time warping data
    mining Proceedings of 5th SIAM International Conference on Data Mining, 2005
    .. [2] Sakoe H. and Chiba S.: Dynamic programming algorithm optimization for
    spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal
    Processing 26(1):43–49, 1978
    .. [3] Itakura F: Minimum prediction residual principle applied to speech
    recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing 23(
    1):67–72, 1975
    """

    def _distance_alignment_path_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_cost_matrix: bool = False,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: Any,
    ) -> DistanceAlignmentPathCallable:
        """Create a no_python compiled dtw path distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        return_cost_matrix: bool, defaults = False
            Boolean that when true will also return the cost matrix.
        window: Float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]
            No_python compiled Dtw distance path callable.

        Raises
        ------
        ValueError
            If the input time series are not numpy array.
            If the input time series do not have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        from sktime.distances._distance_alignment_paths import compute_min_return_path
        from sktime.distances._dtw_numba import _cost_matrix
        from sktime.distances.lower_bounding import resolve_bounding_matrix
        from sktime.utils.numba.njit import njit

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        if return_cost_matrix is True:

            @njit(cache=True)
            def numba_dtw_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float, np.ndarray]:
                cost_matrix = _cost_matrix(_x, _y, _bounding_matrix)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1], cost_matrix

        else:

            @njit(cache=True)
            def numba_dtw_distance_alignment_path(
                _x: np.ndarray,
                _y: np.ndarray,
            ) -> Tuple[List, float]:
                cost_matrix = _cost_matrix(_x, _y, _bounding_matrix)
                path = compute_min_return_path(cost_matrix, _bounding_matrix)
                return path, cost_matrix[-1, -1]

        return numba_dtw_distance_alignment_path

    def _distance_factory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: float = None,
        itakura_max_slope: float = None,
        bounding_matrix: np.ndarray = None,
        **kwargs: Any,
    ) -> DistanceCallable:
        """Create a no_python compiled dtw distance callable.

        Series should be shape (d, m), where d is the number of dimensions, m the series
        length. Series can be different lengths.

        Parameters
        ----------
        x: np.ndarray (2d array of shape (d,m1)).
            First time series.
        y: np.ndarray (2d array of shape (d,m2)).
            Second time series.
        window: Float, defaults = None
            Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
            lower bounding). Must be between 0 and 1.
        itakura_max_slope: float, defaults = None
            Gradient of the slope for itakura parallelogram (if using Itakura
            Parallelogram lower bounding). Must be between 0 and 1.
        bounding_matrix: np.ndarray (2d array of shape (m1,m2)), defaults = None
            Custom bounding matrix to use. If defined then other lower_bounding params
            are ignored. The matrix should be structure so that indexes considered in
            bound should be the value 0. and indexes outside the bounding matrix should
            be infinity.
        kwargs: any
            extra kwargs.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            No_python compiled Dtw distance callable.

        Raises
        ------
        ValueError
            If the input time series are not numpy array.
            If the input time series do not have exactly 2 dimensions.
            If the sakoe_chiba_window_radius is not an integer.
            If the itakura_max_slope is not a float or int.
        """
        from sktime.distances._dtw_numba import _cost_matrix
        from sktime.distances.lower_bounding import resolve_bounding_matrix
        from sktime.utils.numba.njit import njit

        _bounding_matrix = resolve_bounding_matrix(
            x, y, window, itakura_max_slope, bounding_matrix
        )

        @njit(cache=True)
        def numba_dtw_distance(
            _x: np.ndarray,
            _y: np.ndarray,
        ) -> float:
            cost_matrix = _cost_matrix(_x, _y, _bounding_matrix)
            return cost_matrix[-1, -1]

        return numba_dtw_distance
