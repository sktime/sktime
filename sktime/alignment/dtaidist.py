"""Interface module to dtaidist package.

Exposes basic interface, excluding multivariate case.
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerDtwDtai(BaseAligner):
    """Aligner interface for dtaidistance time warping alignment.

    Direct interface to ``dtaidistance.dtw_ndim.warping_path`` and
    ``dtaidistance.dtw_ndim.warping_path_fast``.

    This distance is covers multivariate data and
    arbitrary scalar distances as components.

    Parameters
    ----------
    use_c: bool, optional, default=False
        Whether to use the faster C variant: ``True`` for C, ``False`` for Python.
        ``True`` requires a C compiled installation of ``dtaidistance``.

        * If False, uses ``dtaidistance.dtw_ndim.distance_matrix``.
        * If True, uses ``dtaidistance.dtw_ndim.distance_matrix_fast``.

    window : integer, optional, default=infinite
        Sakoe Chiba window width, from diagonal to boundary.
        Only allow for maximal shifts from the two diagonals smaller than this number.
        The maximally allowed warping, thus difference between indices i
        in series 1 and j in series 2,
        is thus |i-j| < 2*window + |len(s1) - len(s2)|.
        It includes the diagonal, meaning that Euclidean distance is obtained by setting
        ``window=1.``
        If the two series are of equal length, this means that the band appearing
        on the cumulative cost matrix is of width 2*window-1. In other definitions of
        DTW this number may be referred to as the window instead.
    max_dist: float, optional, default=infinite
        Stop if the returned values will be larger than this value.
    max_step: float, optional, default=infinite
        Do not allow steps larger than this value.
        If the difference between two values in the two series is larger than this, thus
        if |s1[i]-s2[j]| > max_step, replace that value with infinity.
    max_length_diff: int, optional, default=infinite
        Return infinity if difference of length of two series is larger than this value.
    penalty: float, optional, default=0
        Penalty to add if compression or expansion is applied
    psi: integer or 4-tuple of integers or none, optional, default=none
        Psi relaxation parameter (ignore start and end of matching).
        If psi is a single integer, it is used for both start and end relaxations
        for both series in a pair of series.
        If psi is a 4-tuple, it is used as the psi-relaxation for
        (begin series1, end series1, begin series2, end series2).
        Useful for cyclical series.

    inner_dist: str, or sktime BasePairwiseTransformer, default="squared euclidean"
        Distance between two points in the time series.

        * If str, must be one of 'squared euclidean' (default), 'euclidean'.
        * if estimator, must follow sktime BasePairwiseTransformer API.
          For a range of distances from scipy, see ``ScipyDist``.

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["wannesm", "probberechts", "fkiraly"],
        # wannesm, probberechts credit for interfaced code
        "python_dependencies": ["dtaidistance"],
        # estimator type
        # --------------
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": True,  # does compute/return overall distance?
        "capability:distance-matrix": True,  # does compute/return distance matrix?
        "capability:unequal_length": True,  # can align sequences of unequal length?
        "alignment_type": "full",
    }

    def __init__(
        self,
        use_c=False,
        window=None,
        max_dist=None,
        max_step=None,
        max_length_diff=None,
        penalty=None,
        psi=None,
        inner_dist="squared euclidean",
    ):
        self.window = window
        self.max_dist = max_dist
        self.max_step = max_step
        self.max_length_diff = max_length_diff
        self.penalty = penalty
        self.psi = psi
        self.use_c = use_c
        self.inner_dist = inner_dist

        super().__init__()

        self._dtai_params = self.get_params()

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
        # soft dependency import of dtw
        from dtaidistance.dtw import warping_path

        dtai_params = self._dtai_params

        # shorthands for 1st and 2nd series
        # dtaidistances requires 2D np.array (time, variable)
        s1 = X[0].values
        s2 = X[1].values

        path, dist = warping_path(
            s1, s2, include_distance=True, use_ndim=True, **dtai_params
        )

        self._path = path
        self._dist = dist

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
        path = self._path

        # convert to required data frame format and return
        aligndf = pd.DataFrame(path, columns=["ind0", "ind1"])

        return aligndf

    def _get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        return self._dist

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
        distmat[0, 1] = self._dist
        distmat[1, 0] = self._dist

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for aligner."""
        params0 = {}
        params1 = {"window": 1, "max_length_diff": 1}
        params2 = {"penalty": 0.1}

        return [params0, params1, params2]
