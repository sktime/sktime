"""Detector that fires where a wrapped detector's score is high for its stream."""

__author__ = ["yash-sangwan"]
__all__ = ["StreamCalibrateFPR"]

import math
from collections import deque
from numbers import Integral, Real

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class StreamCalibrateFPR(BaseDetector):
    """Fire where a detector's score is above a quantile of its recent scores.

    Wraps an anomaly detector that scores every time point, for instance
    ``ReducerPretrainDetector``, and replaces its fixed alarm threshold with one
    taken from the stream itself: a point fires if its score is above the
    ``1 - fpr`` quantile of the last ``window_length`` scores. So about a share
    ``fpr`` of the points fire, whatever the score level of the series.

    In ``pretrain``, a clone of ``detector`` is pretrained and kept as
    ``pretrained_detector_``. In ``fit``, a clone of ``pretrained_detector_``, or
    of ``detector`` if ``pretrain`` was not called, is fitted, and the scores of
    the fitted points fill the buffer of recent scores. No alarm is fired in
    ``fit``.

    In ``update``, the wrapped detector is updated, and the new points are scored
    and decided one at a time, in time order: a point fires if its score is above
    the ``1 - fpr`` quantile of the buffer, and its score is then added to the
    buffer, which keeps the last ``window_length`` scores. ``predict`` on the
    points of the last ``update`` returns these decisions. ``predict`` on other
    points compares their scores with the current quantile, and does not change
    the buffer.

    Nothing fires until the buffer holds ``window_length`` scores. A point
    without a score (NaN), for instance before the window of the wrapped detector
    is full, never fires and is not added to the buffer.

    Parameters
    ----------
    detector : sktime detector
        Anomaly detector that returns a score for every point, via
        ``transform_scores``. It is cloned, and not changed.
    fpr : float
        Share of points that should fire, between 0 and 1, exclusive.
        For instance, one alarm per minute at 10 points per second is
        ``fpr = 1 / 600``.
    window_length : int
        Number of recent scores the quantile is taken from. Must be at least
        ``ceil(1 / fpr)``, as fewer scores cannot tell a ``1 - fpr`` quantile
        apart from their maximum.

    Attributes
    ----------
    pretrained_detector_ : sktime detector
        Clone of ``detector``, pretrained in ``pretrain``. Only set if
        ``pretrain`` was called.
    detector_ : sktime detector
        Clone of ``pretrained_detector_``, or of ``detector``, fitted in ``fit``
        and updated in ``update``.
    scores_buffer_ : collections.deque
        The last ``window_length`` scores, oldest first, without NaN scores.
    threshold_ : float
        The ``1 - fpr`` quantile of ``scores_buffer_``, the smallest buffer
        value with at most a share ``fpr`` of the buffer above it. NaN while the
        buffer holds fewer than ``window_length`` scores.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.detection.compose import StreamCalibrateFPR
    >>> from sktime.detection.reduce import ReducerPretrainDetector
    >>> X = pd.DataFrame({"x": np.random.default_rng(0).normal(size=300)})
    >>> y = pd.DataFrame({"ilocs": [40, 90, 140]})
    >>> detector = StreamCalibrateFPR(
    ...     ReducerPretrainDetector(window_length=5), fpr=0.1, window_length=50
    ... )
    >>> detector = detector.fit(X.iloc[:200], y)
    >>> alarms = detector.update_predict(X.iloc[200:])
    """

    _tags = {
        "authors": ["yash-sangwan"],
        "maintainers": ["yash-sangwan"],
        "task": "anomaly_detection",
        "learning_type": "supervised",
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:pretrain": True,
        "capability:update": True,
        # the buffer of recent scores needs every update, so fit is not empty
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # tags that describe the data the wrapped detector takes, and what it does
    _tags_to_clone = [
        "task",
        "learning_type",
        "capability:multivariate",
        "capability:missing_values",
        "capability:pretrain",
        "X_inner_mtype",
    ]

    def __init__(self, detector, fpr, window_length):
        self.detector = detector
        self.fpr = fpr
        self.window_length = window_length
        super().__init__()

        if isinstance(detector, BaseDetector):
            self.clone_tags(detector, self._tags_to_clone)

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sklearn.tree import DecisionTreeClassifier

        from sktime.detection.reduce import ReducerPretrainDetector

        params = [
            {
                "detector": ReducerPretrainDetector(window_length=3),
                "fpr": 0.5,
                "window_length": 2,
            },
            {
                "detector": ReducerPretrainDetector(
                    estimator=DecisionTreeClassifier(random_state=0),
                    window_length=2,
                    detection_offset=1,
                ),
                "fpr": 0.25,
                "window_length": 8,
            },
        ]
        return params

    def _pretrain(self, X, y=None):
        """Pretrain a clone of the wrapped detector.

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets ``pretrained_detector_``.

        Parameters
        ----------
        X : time series panel
            Series to pretrain on, passed to ``pretrain`` of the wrapped detector.
        y : pd.DataFrame, optional
            Known events of ``X``, passed to ``pretrain`` of the wrapped detector.

        Returns
        -------
        self :
            Reference to self.
        """
        self._check_params()
        self.pretrained_detector_ = self.detector.clone().pretrain(X, y)
        return self

    def _fit(self, X, y=None):
        """Fit a clone of the (pretrained) wrapped detector, and fill the buffer.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets ``detector_``, ``scores_buffer_`` and ``threshold_``.

        Parameters
        ----------
        X : pd.DataFrame
            Start of the time series. Its scores fill the buffer, and no alarm
            is fired for it.
        y : pd.DataFrame, optional
            Known events in ``X``, passed to ``fit`` of the wrapped detector.

        Returns
        -------
        self :
            Reference to self.
        """
        self._check_params()

        # each fit starts from the pretrained detector, not from a previous fit
        if hasattr(self, "pretrained_detector_"):
            self.detector_ = self.pretrained_detector_.clone()
        else:
            self.detector_ = self.detector.clone()
        self.detector_.fit(X, y)

        scores = self._scores(X)
        self.scores_buffer_ = deque(
            scores[~np.isnan(scores)], maxlen=self.window_length
        )
        self.threshold_ = self._threshold()
        self._last_update = None
        return self

    def _update(self, X, y=None):
        """Update the wrapped detector, and decide the new points one at a time.

        private _update containing the core logic, called from update

        Writes to self:
            Adds the scores of ``X`` to ``scores_buffer_``, sets ``threshold_``,
            and keeps the decisions for ``X``, returned by ``predict``.

        Parameters
        ----------
        X : pd.DataFrame
            New points of the time series, following the points seen so far.
        y : pd.DataFrame, optional
            Known events in ``X``, passed to ``update`` of the wrapped detector.

        Returns
        -------
        self :
            Reference to self.
        """
        self.detector_.update(X, y)

        fired = []
        for iloc, score in enumerate(self._scores(X)):
            if np.isnan(score):
                continue
            # the threshold is NaN until the buffer is full, and nothing is above NaN
            if score > self._threshold():
                fired.append(iloc)
            self.scores_buffer_.append(score)

        self.threshold_ = self._threshold()
        self._last_update = (X.copy(), np.asarray(fired, dtype="int64"))
        return self

    def _predict(self, X):
        """Return the alarms in X.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection. The points of the last ``update``,
            or points that follow the points seen so far.

        Returns
        -------
        y : pd.Series with RangeIndex
            Alarms in ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.
        """
        last = self._last_update
        if last is not None and X.equals(last[0]):
            ilocs = last[1]
        else:
            # points not seen in update are compared with the current quantile,
            # and the buffer is not changed
            scores = np.nan_to_num(self._scores(X), nan=-np.inf)
            ilocs = np.flatnonzero(scores > self.threshold_)

        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()

        return pd.Series(ilocs, dtype="int64")

    def _scores(self, X):
        """Return the score of every point of X, from the wrapped detector."""
        scores = self.detector_.transform_scores(X)
        if isinstance(scores, pd.DataFrame):
            scores = scores.iloc[:, 0]
        scores = np.asarray(scores, dtype="float64")

        if len(scores) != len(X):
            raise ValueError(
                "detector must return one score per point of X from "
                f"transform_scores, but returned {len(scores)} scores for "
                f"{len(X)} points."
            )
        return scores

    def _threshold(self):
        """Return the 1 - fpr quantile of the buffer, NaN if it is not full.

        The quantile is a buffer value, the one at position
        ``ceil((1 - fpr) * (window_length - 1))`` in sorted order, so at most a
        share ``fpr`` of the buffer is above it.
        """
        n_scores = len(self.scores_buffer_)
        if n_scores < self.window_length:
            return np.nan

        # small tolerance, so that an exact position is not pushed up by rounding
        position = math.ceil((1 - self.fpr) * (n_scores - 1) - 1e-9)
        values = np.fromiter(self.scores_buffer_, dtype="float64", count=n_scores)
        return float(np.partition(values, position)[position])

    def _check_params(self):
        """Check the wrapped detector, fpr and window_length."""
        detector = self.detector
        if not isinstance(detector, BaseDetector):
            raise TypeError(
                "detector must be an sktime detector, an instance of "
                f"BaseDetector, but found {type(detector).__name__}."
            )
        task = detector.get_tag("task")
        if task != "anomaly_detection":
            raise ValueError(
                "detector must be an anomaly detector, with the task tag "
                f"'anomaly_detection', as only point alarms are calibrated, "
                f"but found task {task!r}."
            )
        if type(detector)._transform_scores is BaseDetector._transform_scores:
            raise TypeError(
                "detector must return a score for every point, via "
                f"transform_scores, but {type(detector).__name__} does not "
                "implement it."
            )

        fpr = self.fpr
        if isinstance(fpr, bool) or not isinstance(fpr, Real) or not 0 < fpr < 1:
            raise ValueError(
                f"fpr must be a number between 0 and 1, exclusive, but found {fpr!r}."
            )

        window_length = self.window_length
        min_window_length = math.ceil(1 / fpr - 1e-9)
        if (
            isinstance(window_length, bool)
            or not isinstance(window_length, Integral)
            or window_length < min_window_length
        ):
            raise ValueError(
                "window_length must be an integer of at least ceil(1 / fpr) = "
                f"{min_window_length}, but found {window_length!r}."
            )
