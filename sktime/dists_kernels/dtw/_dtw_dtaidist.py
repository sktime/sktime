"""Dynamic time warping distance, from dtaidistance."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel


class DtwDtaidistUniv(BasePairwiseTransformerPanel):
    """Univariate dynamic time warping distance, from dtaidistance.

    Direct interface to ``dtaidistance.dtw.distance_matrix`` and
    ``dtaidistance.dtw.distance_matrix_fast``.

    Parameters
    ----------
    use_c: bool, optional, default=False
        Use the faster C variant.
        Require a C compiled installation of ``dtaidistance``.

        * If False, uses ``dtaidistance.dtw.distance_matrix``.
        * If True, uses ``dtaidistance.dtw.distance_matrix_fast``.

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
    use_pruning: bool, optional, default=False
        Prune values based on Euclidean distance.

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
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
        "symmetric": True,  # all the distances are symmetric
        "capability:multivariate": False,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        "X_inner_mtype": "df-list",
    }

    def __init__(
        self,
        window=None,
        use_pruning=False,
        max_dist=None,
        max_step=None,
        max_length_diff=None,
        penalty=None,
        psi=None,
        use_c=False,
    ):
        self.window = window
        self.use_pruning = use_pruning
        self.max_dist = max_dist
        self.max_step = max_step
        self.max_length_diff = max_length_diff
        self.penalty = penalty
        self.psi = psi
        self.use_c = use_c

        super().__init__()

        self._dtai_params = self.get_params()

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: list of pd.DataFrame of length n
        X2: list of pd.DataFrame of length m, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        from dtaidistance.dtw import distance_matrix

        dtai_params = self._dtai_params

        if X2 is None:
            X_np = [x.values.flatten() for x in X]
            distmat = distance_matrix(X_np, **dtai_params)
            return distmat

        # else we know X, X2 are lists of df
        X_np = [x.values.flatten() for x in X]
        X2_np = [x.values.flatten() for x in X2]

        len_X = len(X)
        len_X2 = len(X2)
        block = ((0, len_X - 1), (len_X, len_X + len_X2 - 1))
        X_all = X_np + X2_np

        distmat = distance_matrix(X_all, block=block, **dtai_params)
        distmat_ss = distmat[:len_X, -len_X2:]
        return distmat_ss

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for distance/kernel transformers.

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
        params1 = {"window": 1, "use_pruning": True, "max_length_diff": 1}
        params2 = {"penalty": 0.1, "psi": 2}

        return [params0, params1, params2]
