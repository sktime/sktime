"""BaseAligner interface to sktime edit distance aligners in distances module."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerEditNumba(BaseAligner):
    r"""Interface to sktime native edit distance aligners.

    Interface to the following edit distance aligners:
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
        if passed, must be of shape (len(X), len(X2)) for X, X2 in ``transform``
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
        Computing Machinery, New York, NY, USA, 491-502.
        DOI:https://doi.org/10.1145/1066157.1066213
    .. [3] Lei Chen and Raymond Ng. 2004. On the marriage of Lp-norms and edit distance.
        In Proceedings of the Thirtieth international conference on Very large data
        bases - Volume 30 (VLDB '04). VLDB Endowment, 792-803.
    .. [4] Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment
        for Time Series Matching". IEEE Transactions on Pattern Analysis and Machine
        Intelligence. 31 (2): 306-318.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.dists_kernels.edit_dist import EditDist
    >>>
    >>> X, _ = load_unit_test(return_type="pd-multiindex")  # doctest: +SKIP
    >>> d = EditDist("edr")  # doctest: +SKIP
    >>> distmat = d.transform(X)  # doctest: +SKIP

    distances are also callable, this does the same:
    >>> distmat = d(X)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["chrisholder", "TonyBagnall", "fkiraly"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "symmetric": True,  # all the distances are symmetric
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": True,  # does compute/return overall distance?
        "capability:distance-matrix": True,  # does compute/return distance matrix?
        "capability:unequal_length": False,  # can align sequences of unequal length?
        "alignment_type": "partial",
        "X_inner_mtype": "numpy3D",
    }

    ALLOWED_DISTANCE_STR = ["lcss", "edr", "erp", "twe"]

    def __init__(
        self,
        distance: str = "lcss",
        window=None,
        itakura_max_slope=None,
        bounding_matrix: np.ndarray = None,
        epsilon: float = 1.0,
        g: float = 0.0,
        lmbda: float = 1.0,
        nu: float = 0.001,
        p: int = 2,
    ):
        self.distance = distance
        self.window = window
        self.itakura_max_slope = itakura_max_slope
        self.bounding_matrix = bounding_matrix
        self.epsilon = epsilon
        self.g = g
        self.lmbda = lmbda
        self.nu = nu
        self.p = p

        super().__init__()

        kwargs = {
            "window": window,
            "itakura_max_slope": itakura_max_slope,
            "bounding_matrix": bounding_matrix,
        }

        if distance not in self.ALLOWED_DISTANCE_STR:
            raise ValueError(
                "distance must be one of the strings"
                f"{self.ALLOWED_DISTANCE_STR}, but found"
                f" {distance}"
            )

        # epsilon is used only for lcss, edr, erp
        if distance in ["lcss", "edr", "erp"]:
            kwargs["epsilon"] = epsilon

        # g is used only for erp
        if distance == "erp":
            kwargs["g"] = g

        # twe has three unique params
        if distance == "twe":
            kwargs["lmbda"] = lmbda
            kwargs["nu"] = nu
            kwargs["p"] = p

        self.kwargs = kwargs

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Writes to self:
            alignment : computed alignment from dtw package (nested struct)

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X
        """
        from sktime.distances import distance_alignment_path

        X1 = X[0]
        X2 = X[1]

        metric_key = self.distance
        kwargs = self.kwargs

        path, dist = distance_alignment_path(X1, X2, metric=metric_key, **kwargs)
        self.path_ = path
        self.dist_ = dist

        return self

    def _get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        # retrieve alignment
        path = self.path_
        ind0, ind1 = zip(*path)

        # convert to required data frame format and return
        aligndf = pd.DataFrame({"ind0": ind0, "ind1": ind1})

        return aligndf

    def _get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        return self.dist_

    def _get_distance_matrix(self):
        """Return distance matrix of alignment.

        Behaviour: returns pairwise distance matrix of alignment distances
            not all aligners will return or implement this (optional)

        Returns
        -------
        distmat: a (2 x 2) np.array of floats
            [i,j]-th entry is alignment distance between X[i] and X[j] passed to fit
        """
        # since dtw does only pairwise alignments, this is always a 2x2 matrix
        distmat = np.zeros((2, 2), dtype="float")
        distmat[0, 1] = self.dist_
        distmat[1, 0] = self.dist_

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for aligners.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {}
        params1 = {"distance": "twe"}

        return [params0, params1]
