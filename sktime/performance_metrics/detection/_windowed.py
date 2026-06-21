"""Shared windowed point-detection metric utilities."""

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils import _count_windowed_matches


def _resolve_margins(margin, margin_backward, margin_forward):
    """Resolve symmetric and directional window parameters."""
    if margin_backward is None and margin_forward is None:
        return margin, margin

    if margin_backward is None:
        margin_backward = margin
    if margin_forward is None:
        margin_forward = margin

    return margin_backward, margin_forward


def _point_values(y, X=None, use_loc=False):
    """Return point-event locations in iloc or loc units."""
    ilocs = y["ilocs"].values

    if X is not None and use_loc:
        return X.index[ilocs].to_numpy()

    return ilocs


class _BaseWindowedDetectionScore(BaseDetectionMetric):
    """Base class for margin-based point event detection scores."""

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": False,
    }

    def __init__(
        self, margin=0, margin_backward=None, margin_forward=None, use_loc=False
    ):
        self.margin = margin
        self.margin_backward = margin_backward
        self.margin_forward = margin_forward
        self.use_loc = use_loc
        super().__init__()

    def _get_match_counts(self, y_true, y_pred, X=None):
        true_values = _point_values(y_true, X=X, use_loc=self.use_loc)
        pred_values = _point_values(y_pred, X=X, use_loc=self.use_loc)

        true_count = len(true_values)
        pred_count = len(pred_values)
        margin_backward, margin_forward = _resolve_margins(
            self.margin, self.margin_backward, self.margin_forward
        )
        matched_count = _count_windowed_matches(
            true_values,
            pred_values,
            margin_backward=margin_backward,
            margin_forward=margin_forward,
        )

        return matched_count, true_count, pred_count

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        param0 = {}
        param1 = {"margin": 1}
        param2 = {"margin_backward": 0, "margin_forward": 2}
        param3 = {"margin": 1, "use_loc": True}

        return [param0, param1, param2, param3]
