"""The subset multivariate collective and point anomalies (MVCAPA) algorithm."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record(["skchange.anomaly_detectors", "skchange.anomaly_detectors.capa"])
class MVCAPA(BaseDetector):
    """MVCAPA = Multivariate collective and point anomaly detection, from skchange.

    Redirects to ``skchange.anomaly_detectors.mvcapa``.

    An efficient implementation of the MVCAPA algorithm [1]_ for anomaly detection.

    Parameters
    ----------
    collective_saving : BaseSaving or BaseCost, optional (default=L2Cost(0.0))
        The saving function to use for collective anomaly detection.
        Only univariate savings are permitted (see the `evaluation_type` attribute).
        If a ``BaseCost`` is given, the saving function is constructed from the cost.
        The cost must have a fixed parameter that represents the baseline cost.
    point_saving : BaseSaving or BaseCost, optional (default=L2Cost(0.0))
        The saving function to use for point anomaly detection. Only savings with a
        minimum size of 1 are permitted.
        If a ``BaseCost`` is given, the saving function is constructed from the cost.
        The cost must have a fixed parameter that represents the baseline cost.
    collective_penalty : str or Callable, optional, default="combined"
        Penalty function to use for collective anomalies. If a string, must be one of
        "dense", "sparse", "intermediate" or "combined". If a Callable, must be a
        function returning a penalty and per-component penalties, given n, p, n_params
        and scale.
    collective_penalty_scale : float, optional, default=1.0
        Scaling factor for the collective penalty.
    point_penalty : str or Callable, optional, default="sparse"
        Penalty function to use for point anomalies. See ``collective_penalty``.
    point_penalty_scale : float, optional, default=1.0
        Scaling factor for the point penalty.
    min_segment_length : int, optional, default=2
        Minimum length of a segment.
    max_segment_length : int, optional, default=1000
        Maximum length of a segment.
    ignore_point_anomalies : bool, optional, default=False
        If True, detected point anomalies are not returned by `predict`. I.e., only
        collective anomalies are returned.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). Subset multivariate
       collective and point anomaly detection. Journal of Computational and Graphical
       Statistics, 31(2), 574-585.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.anomaly_detectors import MVCAPA
    >>> from skchange.datasets.generate import generate_anomalous_data
    >>> n = 300
    >>> means = [np.array([8.0, 0.0, 0.0]), np.array([2.0, 3.0, 5.0])]
    >>> df = generate_anomalous_data(
    >>>     n, anomalies=[(100, 120), (250, 300)], means=means, random_state=3
    >>> )
    >>> detector = MVCAPA()
    >>> detector.fit_predict(df)
      anomaly_interval anomaly_columns
    0       [100, 120)             [0]
    1       [250, 300)       [2, 1, 0]

    Notes
    -----
    The MVCAPA algorithm assumes the input data is centered before fitting and
    predicting.
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
        collective_penalty="combined",
        collective_penalty_scale: float = 2.0,
        point_penalty="sparse",
        point_penalty_scale: float = 2.0,
        min_segment_length: int = 2,
        max_segment_length: int = 1000,
        ignore_point_anomalies: bool = False,
    ):
        self.collective_saving = collective_saving
        self.point_saving = point_saving
        self.collective_penalty = collective_penalty
        self.collective_penalty_scale = collective_penalty_scale
        self.point_penalty = point_penalty
        self.point_penalty_scale = point_penalty_scale
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.ignore_point_anomalies = ignore_point_anomalies
        super().__init__()
