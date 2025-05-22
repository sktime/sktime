"""BaseEstimator interface to dynamic time warping distances in dtw_python."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base._delegate import _DelegatedPairwiseTransformerPanel


class DtwPythonDist(_DelegatedPairwiseTransformerPanel):
    r"""Interface to dynamic time warping distances in the dtw-python package.

    Computes the dynamic time warping distance between series, using
    the dtw-python package.

    Parameters
    ----------
    dist: str, or estimator following sktime BasePairwiseTransformer API
        distance to use, a distance on real n-space, default = "euclidean"
        if str, must be name of one of the functions in ``scipy.spatial.distance.cdist``
        if estimator, must follow sktime BasePairwiseTransformer API
    step_pattern : str, optional, default = "symmetric2",
        or dtw_python stepPattern object, optional
        step pattern to use in time warping
        one of: 'symmetric1', 'symmetric2' (default), 'asymmetric',
        and dozens of other more non-standard step patterns;
        list can be displayed by calling help(stepPattern) in dtw
    window_type: str  optional, default = "none"
        the chosen windowing function
        "none", "itakura", "sakoechiba", or "slantedband"
        "none" (default) - no windowing
        "sakoechiba" - a band around main diagonal
        "slantedband" - a band around slanted diagonal
        "itakura" - Itakura parallelogram
    open_begin : boolean, optional, default=False
    open_end: boolean, optional, default=False
        whether to perform open-ended alignments
        open_begin = whether alignment open ended at start (low index)
        open_end = whether alignment open ended at end (high index)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["tonigi", "fkiraly"],  # tonigi for dtw-python
        "python_dependencies": "dtw-python",
        # estimator type
        # --------------
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
        "symmetric": True,  # all the distances are symmetric
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        "X_inner_mtype": "df-list",
    }

    def __init__(
        self,
        dist="euclidean",
        step_pattern="symmetric2",
        window_type="none",
        open_begin=False,
        open_end=False,
    ):
        self.dist = dist
        self.step_pattern = step_pattern
        self.window_type = window_type
        self.open_begin = open_begin
        self.open_end = open_end

        super().__init__()

        params = {
            "step_pattern": step_pattern,
            "window_type": window_type,
            "open_begin": open_begin,
            "open_end": open_end,
        }

        from sktime.alignment.dtw_python import AlignerDTW, AlignerDTWfromDist
        from sktime.dists_kernels.compose_from_align import DistFromAligner

        if isinstance(dist, str):
            params["dist_method"] = dist
            delegate = DistFromAligner(AlignerDTW(**params))
        else:
            params["dist_trafo"] = dist
            delegate = DistFromAligner(AlignerDTWfromDist(**params))

        self.estimator_ = delegate

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
        from sktime.dists_kernels import ScipyDist

        params0 = {}
        params1 = {"dist": "cityblock"}
        params2 = {"dist": ScipyDist()}
        params3 = {"dist": ScipyDist("cityblock"), "step_pattern": "symmetric1"}

        return [params0, params1, params2, params3]
