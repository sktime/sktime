"""Naive quantile-based detector."""

import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.detection.utils._arr_to_seg import sparse_pts_to_seg
from sktime.detection.utils._seg_middle import seg_middlepoint


class QuantileDetector(BaseDetector):
    """Naive detector which learns thresholds from training data quantiles.

    Fits upper and lower thresholds based on quantiles of the training data,
    then detects all events in new data that lie outside the learned interval.

    This is a data-driven complement to ``ThresholdDetector``, which uses
    fixed, user-specified thresholds. ``QuantileDetector`` instead learns
    the thresholds from training data.

    The parameter ``mode`` determines whether segments are returned,
    or midpoints of segments.

    Parameters
    ----------
    upper_quantile : float or None, optional, default=0.99
        Upper quantile for the threshold, must be in [0, 1].
        If None, no upper bound is applied.
    lower_quantile : float or None, optional, default=0.01
        Lower quantile for the threshold, must be in [0, 1].
        If None, no lower bound is applied.
    mode : str, optional, one of "segments", "points", default="segments"
        Type of detection returned.

        * ``"segments"``: returns detected segments as ``pd.Interval`` iloc values.
        * ``"points"``: returns midpoints of detected segments, integer iloc values.

    Attributes
    ----------
    upper_threshold_ : float or None
        Upper threshold learned from training data. Set in ``_fit``.
    lower_threshold_ : float or None
        Lower threshold learned from training data. Set in ``_fit``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.naive import QuantileDetector
    >>> y_train = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 2, 1, 2])
    >>> y_test = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 42, 43, 1])
    >>> d = QuantileDetector(upper_quantile=0.95, lower_quantile=0.05)
    >>> y_pred = d.fit_predict(y_test)
    >>> d.predict(y_test)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["Maya-Mohamed"],
        "maintainers": ["Maya-Mohamed"],
        "capability:multivariate": False,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    def __init__(self, upper_quantile=0.99, lower_quantile=0.01, mode="segments"):
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.mode = mode

        super().__init__()

        # Validate quantile values
        if upper_quantile is not None and not 0 <= upper_quantile <= 1:
            raise ValueError(
                f"upper_quantile must be between 0 and 1, got {upper_quantile}."
            )
        if lower_quantile is not None and not 0 <= lower_quantile <= 1:
            raise ValueError(
                f"lower_quantile must be between 0 and 1, got {lower_quantile}."
            )
        if (
            upper_quantile is not None
            and lower_quantile is not None
            and upper_quantile <= lower_quantile
        ):
            raise ValueError(
                f"upper_quantile ({upper_quantile}) must be strictly greater "
                f"than lower_quantile ({lower_quantile})."
            )

        # Dynamically set task tag based on mode, same as ThresholdDetector
        if mode == "segments":
            self.set_tags(**{"task": "segmentation"})
        elif mode == "points":
            self.set_tags(**{"task": "anomaly_detection"})
        else:
            raise ValueError(f"Error in QuantileDetector: unknown mode {mode}")

    def _fit(self, X, y=None):
        """Fit the detector by learning quantile thresholds from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to (time series).
        y : pd.DataFrame, optional
            Not used, present for API consistency.

        Returns
        -------
        self : reference to self.
        """
        if isinstance(X, pd.DataFrame):
            X_vals = X.iloc[:, 0]
        else:
            X_vals = X

        if self.upper_quantile is not None:
            self.upper_threshold_ = X_vals.quantile(self.upper_quantile)
        else:
            self.upper_threshold_ = None

        if self.lower_quantile is not None:
            self.lower_threshold_ = X_vals.quantile(self.lower_quantile)
        else:
            self.lower_threshold_ = None

        return self

    def _predict(self, X, y=None):
        """Create labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection.

        Returns
        -------
        y : pd.DataFrame with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"``,
              the values are integer indices of the anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        if isinstance(X, pd.DataFrame):
            X_vals = X.iloc[:, 0]
        else:
            X_vals = X
        X_vals = X_vals.reset_index(drop=True)

        if self.upper_threshold_ is not None:
            above_thresh = X_vals > self.upper_threshold_
        else:
            above_thresh = pd.Series(False, index=X_vals.index)

        if self.lower_threshold_ is not None:
            below_thresh = X_vals < self.lower_threshold_
        else:
            below_thresh = pd.Series(False, index=X_vals.index)

        outside_thresh = above_thresh | below_thresh
        outside_thresh_ilocs = outside_thresh[outside_thresh].index.values

        # deal with "no detections" case
        if len(outside_thresh_ilocs) == 0:
            if self.mode == "segments":
                return self._empty_segments()
            else:
                return self._empty_sparse()

        segs = sparse_pts_to_seg(outside_thresh_ilocs)

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
        params1 = {"upper_quantile": 0.95, "lower_quantile": 0.05}
        params2 = {"upper_quantile": 0.99, "lower_quantile": None}
        params3 = {"upper_quantile": None, "lower_quantile": 0.01}
        params4 = {"upper_quantile": None, "lower_quantile": None}
        params5 = {"mode": "points"}
        params6 = {
            "upper_quantile": 0.9,
            "lower_quantile": 0.1,
            "mode": "points",
        }

        return [params0, params1, params2, params3, params4, params5, params6]
