"""The collective and point anomalies (CAPA) algorithm."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record("skchange.anomaly_detectors.capa")
class CAPA(BaseDetector):
    """CAPA = Collective and point anomaly detection, from skchange.

    Redirects to ``skchange.anomaly_detectors.capa``.

    An efficient implementation of the CAPA algorithm [1]_ for anomaly detection.
    It is implemented using the 'savings' formulation of the problem given in [2]_ and
    [3]_.

    ``CAPA`` can be applied to both univariate and multivariate data, but does not infer
    the subset of affected components for each anomaly in the multivariate case. See
    ``MVCAPA`` if such inference is desired.

    Parameters
    ----------
    collective_saving : BaseSaving or BaseCost, optional (default=L2Cost(0.0))
        The saving function to use for collective anomaly detection.
        If a ``BaseCost`` is given, the saving function is constructed from the cost.
        The cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseSaving or BaseCost, optional (default=L2Cost(0.0))
        The saving function to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a ``BaseCost`` is given, the saving function is constructed from the cost.
        The cost must have a fixed parameter that represents the baseline cost.
    collective_penalty_scale : float, optional (default=2.0)
        Scaling factor for the collective penalty.
    point_penalty_scale : float, optional (default=2.0)
        Scaling factor for the point penalty.
    min_segment_length : int, optional (default=2)
        Minimum length of a segment.
    max_segment_length : int, optional (default=1000)
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional (default=False)
        If True, detected point anomalies are not returned by `predict`. I.e., only
        collective anomalies are returned. If False, point anomalies are included in the
        output as collective anomalies of length 1.

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
        "python_dependencies": "skchange>=0.6.0",
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
        collective_saving=None,
        point_saving=None,
        collective_penalty_scale: float = 2.0,
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.collective_saving = collective_saving
        self.point_saving = point_saving
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()
