"""Anomaly detection by reduction to tabular classification of windows."""

__author__ = ["yash-sangwan"]

from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.detection.base import BaseDetector


class ReducerPretrainDetector(BaseDetector):
    """Anomaly detector which classifies the window ending at each time point.

    Reduces stream anomaly detection to tabular classification. Every time point
    ``t`` is described by the window of the last ``window_length`` points up to
    and including ``t``, flattened to one row. An sklearn classifier learns from
    such rows which windows come ``detection_offset`` points before an event.

    In ``pretrain``, the classifier is fitted to a collection of time series and
    their known events:

    * positive rows are the windows ending ``detection_offset`` points before
      each event, so the detector can fire before the event happens.
      An event is skipped unless the window ending ``detection_offset`` points
      before it is full, and so is an event outside its own series.
    * negative rows are windows with no event in them, and no event within
      ``detection_offset`` points after them. A ``negative_window_fraction`` of
      all such windows of a series is used, equally spaced.

    No window reaches from one series into another. A second call to
    ``pretrain`` replaces the classifier of the first.

    In ``fit`` and ``update``, nothing is learnt if the detector was pretrained:
    only the last ``window_length - 1`` points are kept, so the windows of the
    next points can be completed. If ``pretrain`` was not called, the classifier
    is fitted in ``fit``, to the known events ``y`` of the series fitted to. If
    no ``y`` is passed to ``fit`` either, no alarms are fired.

    In ``predict``, every point of ``X`` is scored with the probability of the
    positive class for its window, which uses only that point and the points
    before it. An alarm is fired where the score is above
    ``detection_threshold``. Points without a full window yet get no score, and
    no alarm. ``update_predict`` on consecutive chunks fires the same alarms as
    ``predict`` on the whole series at once, as each chunk passed to ``update``
    follows the points already seen, with no overlap and no gap.

    Parameters
    ----------
    estimator : sklearn classifier with ``predict_proba``, optional, default=None
        Classifier of the windows, cloned before fitting.
        If None, ``LogisticRegression(max_iter=1000)`` is used.
    window_length : int, default=10
        Number of points in a window. A row has ``window_length`` times the
        number of variables features, in time order.
    detection_offset : int, default=0
        Number of points between the end of a positive window and its event.
        With 0, the window ends at the event.
    detection_threshold : float, default=0.5
        An alarm is fired where the score is above this value, between 0 and 1.
    negative_window_fraction : float, default=0.1
        Fraction of the windows without an event that are used as negative rows,
        in each series. Must be above 0, and at most 1.

    Attributes
    ----------
    pretrain_estimator_ : sklearn classifier, or None
        Classifier fitted in ``pretrain``. If only one class was seen, a
        classifier of that single class. None if no series was long enough
        for a window.
    n_pretrain_positive_ : int
        Number of positive rows seen in ``pretrain``.
    n_pretrain_negative_ : int
        Number of negative rows seen in ``pretrain``.
    estimator_ : sklearn classifier, or None
        Classifier used by ``predict``, set in ``fit``. ``pretrain_estimator_``
        if the detector was pretrained, otherwise the classifier fitted to the
        ``y`` passed to ``fit``, otherwise None, and nothing is fired.
    context_ : np.ndarray
        The last ``window_length - 1`` points before the latest points passed
        to ``fit`` or ``update``, which complete the windows in ``predict``.
    tail_ : np.ndarray
        The last ``window_length - 1`` points passed to ``fit`` and ``update``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sktime.detection.reduce import ReducerPretrainDetector

    Three series of 30 points, each with a spike 2 points before its event:

    >>> values = np.zeros((3, 30))
    >>> events = [10, 15, 20]
    >>> for i, event in enumerate(events):
    ...     values[i, event - 2] = 1.0
    >>> X = pd.DataFrame(
    ...     {"value": values.ravel()},
    ...     index=pd.MultiIndex.from_product([["a", "b", "c"], range(30)]),
    ... )
    >>> y = pd.DataFrame(
    ...     {"ilocs": events},
    ...     index=pd.MultiIndex.from_tuples([("a", 0), ("b", 0), ("c", 0)]),
    ... )
    >>> d = ReducerPretrainDetector(
    ...     DecisionTreeClassifier(random_state=0),
    ...     window_length=2,
    ...     detection_offset=1,
    ...     negative_window_fraction=1.0,
    ... )
    >>> d = d.pretrain(X, y)
    >>> d.n_pretrain_positive_
    3

    A new series with a spike at 5 gets an alarm at 6, one point before the
    event it announces:

    >>> x_new = np.zeros(20)
    >>> x_new[5] = 1.0
    >>> X_new = pd.DataFrame({"value": x_new})
    >>> d.fit(X_new).predict(X_new)
       ilocs
    0      6
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
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        estimator=None,
        window_length=10,
        detection_offset=0,
        detection_threshold=0.5,
        negative_window_fraction=0.1,
    ):
        self.estimator = estimator
        self.window_length = window_length
        self.detection_offset = detection_offset
        self.detection_threshold = detection_threshold
        self.negative_window_fraction = negative_window_fraction
        super().__init__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        from sklearn.tree import DecisionTreeClassifier

        return [
            {"window_length": 3},
            {
                "estimator": DecisionTreeClassifier(random_state=0),
                "window_length": 2,
                "detection_offset": 1,
                "detection_threshold": 0.4,
                "negative_window_fraction": 0.5,
            },
        ]

    def _pretrain(self, X, y=None):
        """Fit the classifier to the windows of a collection of time series.

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets ``pretrain_estimator_``, ``n_pretrain_positive_``, and
            ``n_pretrain_negative_``.

        Parameters
        ----------
        X : pd.DataFrame with 2-level row MultiIndex (instance, time)
            Time series to pretrain on. Hierarchical data has been flattened
            to this format by ``pretrain``.
        y : pd.DataFrame, optional
            Known events in ``X``, one row per event, with an ``"ilocs"``
            column of integer positions, each counted from the start of its own
            series, and row ``MultiIndex`` ``(instance, event_no)``.
            Events of instances that are not in ``X`` are ignored.
            For hierarchical ``X``, the instances of ``y`` must use the
            flattened labels, for instance ``"h0_0__h1_0"``, as the instance
            levels of ``y`` are not flattened yet.
            If None, there are no positive rows, and no alarms will be fired.

        Returns
        -------
        self :
            Reference to self.

        Raises
        ------
        ValueError
            If a parameter is out of its range, or ``y`` has an ``"ilocs"``
            entry that is not an integer position.
        TypeError
            If ``estimator`` is not a classifier with ``predict_proba``.
        """
        self._check_params()
        events = self._events_by_instance(y)
        no_events = np.empty(0, dtype="int64")

        rows, labels = [], []
        for instance, X_series in X.groupby(level=0, sort=False):
            series_rows, series_labels = self._training_rows(
                self._values(X_series), events.get(instance, no_events)
            )
            rows.append(series_rows)
            labels.append(series_labels)

        rows = np.vstack(rows)
        labels = np.concatenate(labels)

        self.n_pretrain_positive_ = int(labels.sum())
        self.n_pretrain_negative_ = int(len(labels) - labels.sum())
        self.pretrain_estimator_ = self._fit_classifier(rows, labels)
        return self

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets ``estimator_``, the classifier used by ``predict``.
            Sets ``context_`` to no points, and ``tail_`` to the last
            ``window_length - 1`` points of ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        y : pd.DataFrame, optional
            Known events in ``X``, one row per event, with an ``"ilocs"``
            column. Only used if the detector was not pretrained.

        Returns
        -------
        self :
            Reference to self.

        Raises
        ------
        ValueError
            If a parameter is out of its range, or ``y`` has an ``"ilocs"``
            entry that is not an integer position.
        TypeError
            If ``estimator`` is not a classifier with ``predict_proba``.
        """
        self._check_params()
        values = self._values(X)

        if hasattr(self, "pretrain_estimator_"):
            self.estimator_ = self.pretrain_estimator_
        elif y is not None:
            rows, labels = self._training_rows(values, self._event_ilocs(y))
            self.estimator_ = self._fit_classifier(rows, labels)
        else:
            self.estimator_ = None

        # the stream starts with the series fitted to
        self.context_ = values[:0]
        self.tail_ = self._last_points(values)
        return self

    def _update(self, X, y=None):
        """Add the points in X to the stream, without refitting.

        private _update containing the core logic, called from update

        Writes to self:
            Sets ``context_`` to the points kept before ``X``, and ``tail_`` to
            the last ``window_length - 1`` points, including those of ``X``.

        Parameters
        ----------
        X : pd.DataFrame
            New points of the time series, following the points seen so far.
        y : pd.DataFrame, optional
            Known events in ``X``. Ignored, nothing is refitted.

        Returns
        -------
        self :
            Reference to self.
        """
        values = self._values(X)
        self.context_ = self.tail_
        self.tail_ = self._last_points(np.vstack([self.tail_, values]))
        return self

    def _predict(self, X):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.
            It follows the points in ``context_``.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.
        """
        ilocs = self._alarm_ilocs(self._scores(X))

        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()

        return pd.Series(ilocs, dtype="int64")

    def _predict_scores(self, X):
        """Return the scores of the alarms fired by predict.

        private _predict_scores containing the core logic, called from
        predict_scores

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        scores : pd.Series with RangeIndex
            Score of every alarm returned by ``predict``, in the same order.
        """
        scores = self._scores(X)
        return pd.Series(scores[self._alarm_ilocs(scores)], dtype="float64")

    def _transform_scores(self, X):
        """Return the score of every point of X.

        private _transform_scores containing the core logic, called from
        transform_scores

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Probability of the positive class for the window ending at each
            point, NaN where the window is not full yet.
        """
        return pd.DataFrame(self._scores(X), index=X.index, columns=["scores"])

    def _scores(self, X):
        """Return the score of every point of X, NaN without a full window.

        The window of a point uses only that point and the points before it,
        from ``X`` and from ``context_``. Nothing is written to self.
        """
        values = self._values(X)
        scores = np.full(len(values), np.nan)

        window_length = self.window_length
        data = np.vstack([self.context_, values])
        if len(data) < window_length:
            return scores

        # X[i] is data[first + i], whose window is row first + i - window_length + 1
        first = len(self.context_)
        start = max(window_length - 1 - first, 0)
        rows = self._windows(data)[first + start - window_length + 1 :]

        scores[start:] = self._positive_proba(rows)
        return scores

    def _alarm_ilocs(self, scores):
        """Return the positions whose score is above detection_threshold."""
        # NaN scores, of points without a full window, are never above it
        scores = np.nan_to_num(scores, nan=-np.inf)
        return np.flatnonzero(scores > self.detection_threshold)

    def _positive_proba(self, rows):
        """Return the probability of the positive class for each row."""
        if self.estimator_ is None:
            return np.zeros(len(rows))

        classes = list(self.estimator_.classes_)
        if 1 not in classes:
            return np.zeros(len(rows))

        return self.estimator_.predict_proba(rows)[:, classes.index(1)]

    def _training_rows(self, values, ilocs):
        """Return the rows and labels of one series.

        Parameters
        ----------
        values : np.ndarray of shape (n_timepoints, n_variables)
            Values of one time series.
        ilocs : np.ndarray of int
            Positions of the known events of that series.

        Returns
        -------
        rows : np.ndarray of shape (n_rows, window_length * n_variables)
            Positive rows first, then negative rows.
        labels : np.ndarray of int, of shape (n_rows,)
            1 for positive rows, 0 for negative rows.
        """
        window_length = self.window_length
        offset = self.detection_offset
        n_timepoints = len(values)
        n_features = window_length * values.shape[1]

        if n_timepoints < window_length:
            return np.empty((0, n_features)), np.empty(0, dtype="int64")

        # row k of windows is the window ending at k + window_length - 1
        windows = self._windows(values)
        ends = np.arange(window_length - 1, n_timepoints)

        # an event outside its own series is ignored
        ilocs = ilocs[(ilocs >= 0) & (ilocs < n_timepoints)]

        positive = np.unique(ilocs - offset)
        positive = positive[positive >= window_length - 1]

        # a window ending at t contains an event at e if e <= t < e + window_length,
        # and is followed by it within the offset if e - offset <= t < e
        near_event = np.zeros(n_timepoints, dtype=bool)
        for iloc in ilocs:
            near_event[max(iloc - offset, 0) : iloc + window_length] = True
        candidates = ends[~near_event[ends]]

        n_negative = int(np.ceil(self.negative_window_fraction * len(candidates)))
        if n_negative > 0:
            spaced = np.linspace(0, len(candidates) - 1, n_negative).round()
            negative = candidates[np.unique(spaced.astype("int64"))]
        else:
            negative = candidates[:0]

        first_end = window_length - 1
        rows = np.vstack([windows[positive - first_end], windows[negative - first_end]])
        labels = np.concatenate(
            [np.ones(len(positive), dtype="int64"), np.zeros(len(negative), "int64")]
        )
        return rows, labels

    def _fit_classifier(self, rows, labels):
        """Return a classifier fitted to the rows, or None if there are none.

        With only one class, a classifier of that single class is returned, as
        the classifiers of sklearn need at least two.
        """
        if len(labels) == 0:
            return None

        if len(np.unique(labels)) == 1:
            from sklearn.dummy import DummyClassifier

            return DummyClassifier(strategy="most_frequent").fit(rows, labels)

        if self.estimator is None:
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(max_iter=1000)
        else:
            estimator = clone(self.estimator)

        return estimator.fit(rows, labels)

    def _windows(self, values):
        """Return every full window of values, one flattened row per window end."""
        window_length = self.window_length
        n_windows = len(values) - window_length + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            values, window_length, axis=0
        )
        # (window end, variable, time) to (window end, time, variable), flattened
        return windows.transpose(0, 2, 1).reshape(n_windows, -1)

    @staticmethod
    def _values(X):
        """Return the values of X as a 2D float array, one column per variable.

        ``predict_points`` and ``predict_segments`` of ``BaseDetector`` pass
        ``X`` as it is, which can be a ``pd.Series``.
        """
        values = np.asarray(X, dtype="float64")
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values

    def _last_points(self, values):
        """Return the last window_length - 1 points of values."""
        return values[max(len(values) - (self.window_length - 1), 0) :]

    def _check_params(self):
        """Raise if a parameter is out of its range, or the estimator unfit."""
        window_length = self.window_length
        if (
            isinstance(window_length, bool)
            or not isinstance(window_length, Integral)
            or window_length < 1
        ):
            raise ValueError(
                "window_length must be an integer of at least 1, "
                f"but found {window_length!r}."
            )

        offset = self.detection_offset
        if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 0:
            raise ValueError(
                "detection_offset must be an integer of at least 0, "
                f"but found {offset!r}."
            )

        threshold = self.detection_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "detection_threshold must be a number between 0 and 1, "
                f"but found {threshold!r}."
            )

        fraction = self.negative_window_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or not 0 < fraction <= 1
        ):
            raise ValueError(
                "negative_window_fraction must be a number above 0 and at most 1, "
                f"but found {fraction!r}."
            )

        estimator = self.estimator
        if estimator is not None and not (
            hasattr(estimator, "fit") and hasattr(estimator, "predict_proba")
        ):
            raise TypeError(
                "estimator must be an sklearn classifier with predict_proba, "
                f"but found {type(estimator).__name__}."
            )

    @classmethod
    def _events_by_instance(cls, y):
        """Return the event positions in y, keyed by instance.

        The instance of an event is its row index without the last level. A
        ``y`` without an instance level has no instance of ``X``, so its events
        are ignored.
        """
        if y is None or len(y) == 0:
            return {}

        instances = y.index
        if isinstance(instances, pd.MultiIndex):
            instances = instances.droplevel(-1)

        events = {}
        for instance, iloc in zip(instances, cls._event_ilocs(y)):
            events.setdefault(instance, []).append(iloc)
        return {instance: np.asarray(ilocs) for instance, ilocs in events.items()}

    @staticmethod
    def _event_ilocs(y):
        """Return the ``"ilocs"`` of events y as int, refusing non-integer entries."""
        if isinstance(y, pd.DataFrame):
            y = y["ilocs"]

        ilocs = np.asarray(y, dtype="float64")
        if not np.all(np.isfinite(ilocs)) or not np.all(ilocs == np.round(ilocs)):
            raise ValueError(
                "The ilocs of y must be integer positions, but found non-integer "
                "or missing entries."
            )
        return ilocs.astype("int64")
