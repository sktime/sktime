"""Dummy anomaly detector which fires at the average pretrained event rate."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector


class DummyRateAnomalies(BaseDetector):
    """Dummy anomaly detector which fires at the average event rate seen in pretrain.

    Naive method that can serve as a chance level baseline, or as an API test
    for the pretrain API of detectors.

    In ``pretrain``, learns the average event rate, i.e., the number of events
    per time point, over a collection of time series and their known events.
    A second call to ``pretrain`` replaces the rate learnt by the first call.

    In ``predict``, fires alarms at that rate, at regular distance
    ``step = round(1 / rate)``: at every time-from-start ``t`` with
    ``(t + 1) % step == 0``, so the first alarm of a series is at ``iloc``
    position ``step - 1``. Alarms are ``iloc`` references to the time series
    passed to ``predict``.

    If ``pretrain`` was not called, the rate is learnt in ``fit``, from the
    known events ``y`` of the series fitted to. If no ``y`` is passed to
    ``fit`` either, no alarms are fired.

    The series is treated as a stream. ``fit`` starts the stream at
    time-from-start 0, and ``update`` adds the points it is passed to the
    stream, without refitting. ``predict`` labels the latest points passed to
    ``fit`` or ``update``, at their time-from-start, so ``update_predict`` on
    consecutive chunks fires the same alarms as ``predict`` on the whole
    series at once. Each chunk passed to ``update`` must follow the points
    already seen, with no overlap and no gap, as time-from-start is counted
    in points.

    Attributes
    ----------
    pretrain_event_rate_ : float
        Average number of events per time point, learnt in ``pretrain``.
    n_pretrain_events_ : int
        Number of events seen in ``pretrain``.
    n_pretrain_timepoints_ : int
        Number of time points seen in ``pretrain``.
    event_rate_ : float
        Rate used by ``predict``, set in ``fit``. The rate from ``pretrain``
        if the detector was pretrained, otherwise the rate of the ``y``
        passed to ``fit``, otherwise 0.
    cursor_ : int
        Time-from-start of the first point passed in the latest call to
        ``fit`` or ``update``, where ``predict`` starts. 0 after ``fit``.
    n_timepoints_seen_ : int
        Number of time points passed to ``fit`` and ``update`` so far.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyRateAnomalies
    >>> X_pretrain = pd.DataFrame(
    ...     {"value": range(40)},
    ...     index=pd.MultiIndex.from_product([["a", "b"], range(20)]),
    ... )
    >>> y_pretrain = pd.DataFrame(
    ...     {"ilocs": [5, 15, 10]},
    ...     index=pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ... )
    >>> d = DummyRateAnomalies().pretrain(X_pretrain, y_pretrain)
    >>> d.pretrain_event_rate_
    0.075
    >>> X = pd.Series(range(30))
    >>> d.fit(X).predict(X)
       ilocs
    0     12
    1     25
    """

    _tags = {
        "authors": ["yash-sangwan"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:pretrain": True,
        "capability:update": True,
        "fit_is_empty": False,
        "task": "anomaly_detection",
        "learning_type": "supervised",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self):
        super().__init__()

    def _pretrain(self, X, y=None):
        """Learn the average event rate from a collection of time series.

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets ``pretrain_event_rate_``, ``n_pretrain_events_``, and
            ``n_pretrain_timepoints_``.

        Parameters
        ----------
        X : pd.DataFrame with 2-level row MultiIndex (instance, time)
            Time series to pretrain on. Hierarchical data has been flattened
            to this format by ``pretrain``.
        y : pd.DataFrame, optional
            Known events in ``X``, one row per event, with an ``"ilocs"``
            column, and row ``MultiIndex`` ``(instance, event_no)``.
            ``pretrain`` has flattened the instance levels, so the instance
            ``("h0_0", "h1_0")`` is ``"h0_0__h1_0"`` here.
            Events of instances that are not in ``X`` are ignored.
            If None, no events are seen, and no alarms will be fired.

        Returns
        -------
        self :
            Reference to self.
        """
        instances = X.index.droplevel(-1).unique()

        self.n_pretrain_events_ = self._count_events(y, instances)
        self.n_pretrain_timepoints_ = len(X)
        self.pretrain_event_rate_ = self._rate(
            self.n_pretrain_events_, self.n_pretrain_timepoints_
        )
        return self

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets ``event_rate_``, the rate used by ``predict``.
            Sets ``cursor_`` to 0, and ``n_timepoints_seen_`` to the length of
            ``X``.

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
        """
        if hasattr(self, "pretrain_event_rate_"):
            self.event_rate_ = self.pretrain_event_rate_
        elif y is not None:
            self.event_rate_ = self._rate(len(y), len(X))
        else:
            self.event_rate_ = 0.0

        # the stream starts with the series fitted to
        self.cursor_ = 0
        self.n_timepoints_seen_ = len(X)
        return self

    def _update(self, X, y=None):
        """Add the points in X to the stream, without refitting.

        private _update containing the core logic, called from update

        Writes to self:
            Sets ``cursor_`` to the number of points seen before ``X``, the
            time-from-start of the first point of ``X``, then adds the length
            of ``X`` to ``n_timepoints_seen_``.

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
        self.cursor_ = self.n_timepoints_seen_
        self.n_timepoints_seen_ += len(X)
        return self

    def _predict(self, X):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.
            It starts at time-from-start ``cursor_``.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.
        """
        rate = self.event_rate_

        if rate <= 0:
            return BaseDetector._empty_sparse()

        step = max(1, int(round(1 / rate)))
        # alarms are at every time-from-start t with (t + 1) % step == 0, and X
        # starts at time-from-start cursor_, so the first alarm in X is at
        first = (step - 1 - self.cursor_) % step
        ilocs = np.arange(first, len(X), step)

        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()

        return pd.Series(ilocs, dtype="int64")

    @staticmethod
    def _rate(n_events, n_timepoints):
        """Return events per time point, 0 if there are no time points."""
        if n_timepoints == 0:
            return 0.0
        return n_events / n_timepoints

    @staticmethod
    def _count_events(y, instances):
        """Count events in y that belong to one of the instances.

        ``pretrain`` has flattened the instance levels of ``y``, so the
        instance of an event is its row index without the last level.

        Parameters
        ----------
        y : pd.DataFrame, or None
            Known events, with row ``MultiIndex`` ``(instance, event_no)``.
        instances : pd.Index
            Instances of the time series pretrained on.

        Returns
        -------
        int
            Number of events in ``y`` belonging to an instance in ``instances``.
        """
        if y is None or len(y) == 0:
            return 0

        y_instances = y.index
        if isinstance(y_instances, pd.MultiIndex):
            y_instances = y_instances.droplevel(-1)

        return int(y_instances.isin(instances).sum())
