"""Naive thresholding detector."""

import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.detection.utils._arr_to_seg import sparse_pts_to_seg
from sktime.detection.utils._seg_middle import seg_middlepoint


class ThresholdDetector(BaseDetector):
    """Naive detector which detects all points outside a threshold.

    Detects all events that lie outside a threshold interval.
    Naive method that is typically used as a pipeline component.

    Detects all events that are above ``upper`` and below ``lower``.
    By default, ``upper=1`` and ``lower=-upper``.
    To remove one of these bounds, set it to ``None``.

    The parameter ``mode`` determines whether segments are returned,
    or midpoints of segments.

    Parameters
    ----------
    upper : float or None, optional, default=1
        Upper bound of the threshold interval.
        If None, no upper bound is applied.
    lower : float or None, optional, default=-upper
        Lower bound of the threshold interval.
        If None, no lower bound is applied.
    mode : str, optional, one of "segments", "points", default="segments"
        Type of detection returned.

        * ``"segments"``: returns detected segments as ``pd.Interval`` iloc values.
        * ``"points"``: returns midpoints of detected segments, integer iloc values.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.naive import ThresholdDetector
    >>> y = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 42, 43, 1, 2])
    >>> d = ThresholdDetector(upper=10, mode="segments")
    >>> y_pred = d.fit_predict(y)
    """

    _tags = {
        "authors": ["fkiraly"],
        "capability:multivariate": False,
        "capability:missing_values": False,
        "fit_is_empty": True,
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    def __init__(self, upper=1, lower="-upper", mode="segments"):
        self.upper = upper
        self.lower = lower
        self.mode = mode

        super().__init__()

        if lower == "-upper" and upper is not None:
            self._lower = -upper
        else:
            self._lower = None

        if mode == "segments":
            self.set_tags(**{"task": "segmentation"})
        elif mode == "points":
            self.set_tags(**{"task": "anomaly_detection"})
        else:
            raise ValueError(f"Error in ThresholdDetector: unknown mode {mode}")

    def _predict(self, X, y=None):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        X = X.reset_index(drop=True)

        if self.upper is not None:
            above_thresh = X > self.upper
        else:
            above_thresh = pd.Series(False, index=X.index)

        if self._lower is not None:
            below_thresh = X < self._lower
        else:
            below_thresh = pd.Series(False, index=X.index)

        outside_thresh = above_thresh | below_thresh

        outside_tresh_ilocs = outside_thresh[outside_thresh].index.values

        # deal with "no detections" case
        if len(outside_tresh_ilocs) == 0:
            if self.mode == "segments":
                return self._empty_segments()
            else:
                return self._empty_sparse()

        segs = sparse_pts_to_seg(outside_tresh_ilocs)

        if self.mode == "points":
            segs = seg_middlepoint(segs)

        return segs

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params0 = {}
        params1 = {"upper": 10e6}
        params2 = {"upper": 0.42, "lower": -0.41}
        params3 = {"upper": 0.42, "lower": None}
        params4 = {"upper": None, "lower": -0.41}
        params5 = {"upper": None, "lower": None}
        params6 = {"mode": "points"}
        params7 = {"mode": "points", "lower": None, "upper": 0.42}
        params8 = {"upper": 0.42}

        return [
            params0,
            params1,
            params2,
            params3,
            params4,
            params5,
            params6,
            params7,
            params8,
        ]
