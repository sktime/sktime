"""Compatibility wrapper for MVCAPA."""

import numpy as np

from sktime.detection._skchange.anomaly_detectors import CAPA


class MVCAPA(CAPA):
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
    >>> import numpy as np  # doctest: +SKIP
    >>> from skchange.anomaly_detectors import MVCAPA  # doctest: +SKIP
    >>> from skchange.datasets.generate import generate_anomalous_data  # doctest: +SKIP
    >>> n = 300  # doctest: +SKIP
    >>> means = [np.array([8.0, 0.0, 0.0]), np.array([2.0, 3.0, 5.0])]  # doctest: +SKIP
    >>> df = generate_anomalous_data(  # doctest: +SKIP
    ...     n, anomalies=[(100, 120), (250, 300)], means=means, random_state=3
    ... )  # doctest: +SKIP
    >>> detector = MVCAPA()  # doctest: +SKIP
    >>> detector.fit_predict(df)  # doctest: +SKIP
      anomaly_interval anomaly_columns
    0       [100, 120)             [0]
    1       [250, 300)       [2, 1, 0]

    Notes
    -----
    The MVCAPA algorithm assumes the input data is centered before fitting and
    predicting.
    """

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
        segment_saving=None,
        segment_penalty=None,
        find_affected_components: bool = True,
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
        self.segment_saving = segment_saving
        self.segment_penalty = segment_penalty
        self.find_affected_components = find_affected_components

        segment_saving = collective_saving if segment_saving is None else segment_saving
        segment_penalty = (
            self._resolve_penalty(
                collective_penalty,
                collective_penalty_scale,
                default_name="combined",
            )
            if segment_penalty is None
            else segment_penalty
        )
        point_penalty_resolved = self._resolve_penalty(
            point_penalty,
            point_penalty_scale,
            default_name="sparse",
        )

        super().__init__(
            segment_saving=segment_saving,
            segment_penalty=segment_penalty,
            point_saving=point_saving,
            point_penalty=point_penalty_resolved,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            ignore_point_anomalies=ignore_point_anomalies,
            find_affected_components=find_affected_components,
        )

    @staticmethod
    def _resolve_penalty(penalty, scale: float, default_name: str):
        if isinstance(penalty, str):
            if penalty == default_name:
                return None
            raise ValueError(
                "Only default string penalties are supported by this compatibility "
                f"wrapper: expected '{default_name}', got '{penalty}'."
            )

        if penalty is None:
            return None

        if np.isscalar(penalty):
            return penalty * scale

        return np.asarray(penalty) * scale
