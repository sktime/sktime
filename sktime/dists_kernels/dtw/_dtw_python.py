"""BaseEstimator interface to dynamic time warping distances in dtw_python."""

__author__ = ["fkiraly"]

from sktime.dists_kernels.base._delegate import _DelegatedPairwiseTransformerPanel


class DtwPythonDist(_DelegatedPairwiseTransformerPanel):
    r"""Interface to dynamic time warping distances in the dtw-python package.

    Computes the dynamic time warping distance between series, using
    the dtw-python package.

    Parameters
    ----------
    dist_trafo: str, or estimator following sktime BasePairwiseTransformer API
        distance function to use, a distance on real n-space, default = "euclidean"
        if str, must be name of one of the functions in `scipy.spatial.distance.cdist`
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
        "symmetric": True,  # all the distances are symmetric
        "X_inner_mtype": "df-list",
        "python_dependencies": "dtw-python",
        "python_dependencies_alias": {"dtw-python": "dtw"},
    }

    def __init__(
        self,
        dist_trafo,
        step_pattern="symmetric2",
        window_type="none",
        open_begin=False,
        open_end=False,
    ):
        super().__init__()

        self.dist_trafo = dist_trafo
        self.dist_trafo_ = self.dist_trafo.clone()
        self.step_pattern = step_pattern
        self.window_type = window_type
        self.open_begin = open_begin
        self.open_end = open_end
