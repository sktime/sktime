# -*- coding: utf-8 -*-
"""BaseEstimator interface to sktime edit distances in distances module."""

__author__ = ["fkiraly"]

from typing import Union

import numpy as np
from deprecated.sphinx import deprecated

import sktime.dists_kernels.distances.edit_dist as new_class_loc


# TODO: remove file in v0.15.0
@deprecated(
    version="0.13.4",
    reason="EditDist has moved and this import will be removed in 0.15.0. Import from sktime.dists_kernels.distances",  # noqa: E501
    category=FutureWarning,
)
class EditDist(new_class_loc.EditDist):
    r"""Interface to sktime native edit distances.

    Interface to the following edit distances:
    LCSS - longest common subsequence distance
    ERP - Edit distance for real penalty
    EDR - Edit distance for real sequences
    TWE - Time warp edit distance

    LCSS [1]_ attempts to find the longest common sequence between two time series and
    returns a value that is the percentage that longest common sequence assumes.
    LCSS is computed by matching indexes that are
    similar up until a defined threshold (epsilon).

    The value returned will be between 0.0 and 1.0, where 0.0 means the two time series
    are exactly the same and 1.0 means they are complete opposites.

    EDR [2]_ computes the minimum number of elements (as a percentage) that must be
    removed from x and y so that the sum of the distance between the remaining
    signal elements lies within the tolerance (epsilon).

    The value returned will be between 0 and 1 per time series. The value will
    represent as a percentage of elements that must be removed for the time series to
    be an exact match.

    ERP [3]_ attempts align time series
    by better considering how indexes are carried forward through the cost matrix.
    Usually in the dtw cost matrix, if an alignment can't be found the previous value
    is carried forward. ERP instead proposes the idea of gaps or sequences of points
    that have no matches. These gaps are then punished based on their distance from 'g'.

    TWE [4]_ is a distance measure for discrete time series
    matching with time 'elasticity'. In comparison to other distance measures, (e.g.
    DTW (Dynamic Time Warping) or LCS (Longest Common Subsequence Problem)), TWE is a
    metric. Its computational time complexity is O(n^2), but can be drastically reduced
    in some specific situation by using a corridor to reduce the search space. Its
    memory space complexity can be reduced to O(n).

    Parameters
    ----------
    distance: str, one of ["lcss", "edr", "erp", "twe"], optional, default = "lcss"
        name of the distance that is calculated
    window: float, default = None
        Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding). Value must be between 0. and 1.
    itakura_max_slope: float, default = None
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding)
    bounding_matrix: 2D np.ndarray, optional, default = None
        if passed, must be of shape (len(X), len(X2)) for X, X2 in `transform`
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should
        be infinity.
    epsilon : float, defaults = 1.
        Used in LCSS, EDR, ERP, otherwise ignored
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'.
    g: float, defaults = 0.
        Used in ERP, otherwise ignored.
        The reference value to penalise gaps.
    lmbda: float, optional, default = 1.0
        Used in TWE, otherwise ignored.
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    nu: float optional, default = 0.001
        Used in TWE, otherwise ignored.
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    p: int optional, default = 2
        Used in TWE, otherwise ignored.
        Order of the p-norm for local cost.

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
        Similar Multidimensional Trajectories", In Proceedings of the
        18th International Conference on Data Engineering (ICDE '02).
        IEEE Computer Society, USA, 673.
    .. [2] Lei Chen, M. Tamer Özsu, and Vincent Oria. 2005. Robust and fast similarity
        search for moving object trajectories. In Proceedings of the 2005 ACM SIGMOD
        international conference on Management of data (SIGMOD '05). Association for
        Computing Machinery, New York, NY, USA, 491–502.
        DOI:https://doi.org/10.1145/1066157.1066213
    .. [3] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
        In Proceedings of the Thirtieth international conference on Very large data
        bases - Volume 30 (VLDB '04). VLDB Endowment, 792–803.
    .. [4] Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment
        for Time Series Matching". IEEE Transactions on Pattern Analysis and Machine
        Intelligence. 31 (2): 306–318.
    """

    def __init__(
        self,
        distance: str = "lcss",
        window: Union[int, None] = None,
        itakura_max_slope: Union[float, None] = None,
        bounding_matrix: np.ndarray = None,
        epsilon: float = 1.0,
        g: float = 0.0,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
    ):
        super(EditDist, self).__init__(
            distance=distance,
            window=window,
            itakura_max_slope=itakura_max_slope,
            bounding_matrix=bounding_matrix,
            epsilon=epsilon,
            g=g,
            lmbda=lmbda,
            nu=nu,
            p=p,
        )
