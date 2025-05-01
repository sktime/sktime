"""Anomaly detectors composed of change detectors and some conversion logic."""

import numpy as np

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record(
    ["skchange.anomaly_detectors", "skchange.anomaly_detectors.anomalisers"]
)
class StatThresholdAnomaliser(BaseDetector):
    """Anomaly detection based on thresholding values of segment statistics, skchange.

    Redirects to ``skchange.anomaly_detectors.anomalisers``.

    Parameters
    ----------
    change_detector : ``ChangeDetector``
        Change detector to use for detecting segments.
    stat : callable, optional (default=``np.mean``)
        Statistic to calculate per segment.
    stat_lower : float, optional (default=-1.0)
        Segments with a statistic lower than this value are considered anomalous.
    stat_upper : float, optional (default=1.0)
        Segments with a statistic higher than this value are considered anomalous.
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
        "capability:multivariate": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        change_detector,
        stat=np.mean,
        stat_lower: float = -1.0,
        stat_upper: float = 1.0,
    ):
        self.change_detector = change_detector
        self.stat = stat
        self.stat_lower = stat_lower
        self.stat_upper = stat_upper
        super().__init__()

    def get_test_params():
        """Return dummy test params for the placeholder."""
        return {"change_detector": None}
