"""The collective and point anomalies (CAPA) algorithm."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record(["skchange.anomaly_detectors", "skchange.anomaly_detectors.capa"])
class CAPA(BaseDetector):
    """CAPA = Collective and point anomaly detection, from skchange.

    Redirects to ``skchange.anomaly_detectors.CAPA``.

    An efficient implementation of the CAPA algorithm [1]_ for anomaly detection.
    It is implemented using the 'savings' formulation of the problem given in [2]_ and
    [3]_.

    ``CAPA`` can be applied to both univariate and multivariate data, but does not infer
    the subset of affected components for each anomaly in the multivariate case. See
    ``MVCAPA`` if such inference is desired.

    Parameters
    ----------
    segment_saving : BaseIntervalScorer, optional, default=L2Saving()
        The saving to use for segment anomaly detection.
        If a cost is given, the saving is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
        If a penalised saving is given, it must be constructed from `PenalisedScore`.
    segment_penalty : np.ndarray or float, optional, default=None
        The penalty to use for segment anomaly detection. If the segment penalty is
        penalised (`segment_penalty.get_tag("is_penalised")`) the penalty will
        be ignored. The different types of penalties are as follows:

        * ``float``: A constant penalty applied to the sum of scores across all
          variables in the data.
        * ``np.ndarray``: A penalty array of the same length as the number of
          columns in the data, where element ``i`` of the array is the penalty for
          ``i+1`` variables being affected by an anomaly. The penalty array
          must be positive and increasing (not strictly). A penalised score with a
          linear penalty array is faster to evaluate than a nonlinear penalty array.
        * ``None``: A default constant penalty is created in `predict` based on the
          fitted score using the `make_chi2_penalty` function.

    point_saving : BaseIntervalScorer, optional, default=L2Saving()
        The saving to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a cost is given, the saving is constructed from the cost. The
        cost must have a fixed parameter that represents the baseline cost.
        If a penalised saving is given, it must be constructed from `PenalisedScore`.
    point_penalty : np.ndarray or float, optional, default=None
        The penalty to use for point anomaly detection. See the documentation for
        `segment_penalty` for details. For ``None`` input, the default is set using the
        `make_linear_chi2_penalty` function.
    min_segment_length : int, optional, default=2
        Minimum length of a segment. This may be overridden by the `min_size` of the
        fitted `segment_saving`.
    max_segment_length : int, optional, default=1000
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional, default=False
        If ``True``, detected point anomalies are not returned by `predict`. I.e., only
        segment anomalies are returned. If ``False``, point anomalies are included in
        the output as segment anomalies of length 1.
    find_affected_components : bool, optional, default=False
        If ``True``, the affected components for each segment anomaly are returned in
        the `"icolumns"` key of the `predict` output.
        Only relevant for multivariate data in combination with a penalty array.
        The affected components are sorted from the highest to lowest evidence
        of an anomaly being present in the variable.

    See Also
    --------
    MVCAPA : Multivariate CAPA with subset inference.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method\
        for the detection of collective and point anomalies. Statistical Analysis and\
        DataMining: The ASA Data Science Journal, 15(4), 494-508.

    .. [2] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate\
        collective and point anomaly detection. Journal of Computational and Graphical\
        Statistics, 31(2), 574-585.

    .. [3] Tveten, M., Eckley, I. A., & Fearnhead, P. (2022). Scalable change-point and\
        anomaly detection in cross-correlated data with an application to condition\
        monitoring. The Annals of Applied Statistics, 16(2), 721-743.

    Examples
    --------
    >>> from skchange.anomaly_detectors import CAPA
    >>> from skchange.datasets.generate import generate_alternating_data
    >>> df = generate_alternating_data(n_segments=5, mean=10, segment_length=100)
    >>> detector = CAPA()
    >>> detector.fit_predict(df)
    0    [100, 200)
    1    [300, 400)
    Name: anomaly_interval, dtype: interval
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Tveten"],
        "maintainers": ["Tveten"],
        "python_dependencies": "skchange>=0.14.3",
        # estimator type
        # --------------
        "task": "segmentation",
        "learning_type": "unsupervised",
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        segment_saving=None,
        segment_penalty=None,
        point_saving=None,
        point_penalty=None,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
        find_affected_components = False,
    ):
        self.segment_saving=segment_saving
        self.segment_penalty=segment_penalty
        self.point_saving=point_saving
        self.point_penalty=point_penalty
        self.min_segment_length=min_segment_length
        self.max_segment_length=max_segment_length
        self.ignore_point_anomalies=ignore_point_anomalies
        self.find_affected_components=find_affected_components
        super().__init__()
