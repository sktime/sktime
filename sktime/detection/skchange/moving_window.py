"""Moving Window detector."""

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _placeholder_record


@_placeholder_record(
    "skchange.change_detectors.moving_window", dependencies="skchange>=0.6.0"
)
class MovingWindow(BaseDetector):
    """
    Placeholder for the MovingWindow detector from skchange.

    Detects change points using the Moving Window algorithm.
    """

    _tags = {
        "capability:missing_values": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }
