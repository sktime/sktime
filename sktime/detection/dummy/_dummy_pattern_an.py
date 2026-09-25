"""Dummy anomaly detector which replays an event pattern seen in pretrain."""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.detection.base import BaseDetector


class DummyPatternAnomalies(BaseDetector):
    """Dummy anomaly detector which replays an event pattern seen in pretrain.

    Naive method that can serve as a baseline, or as an API test for the
    pretrain API of detectors.

    In ``pretrain``, stores the event pattern of every time series seen, i.e.,
    the positions of its known events, counted from the start of the series.
    A second call to ``pretrain`` replaces the patterns learnt by the first call.

    In ``fit``, picks one of the stored patterns at random. In ``predict``,
    replays that pattern on the time series passed, ignoring its values.
    Alarms are ``iloc`` references to the time series passed to ``predict``,
    and alarms beyond the end of that series are dropped.

    If ``pretrain`` was not called, the pattern is taken in ``fit``, from the
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

    Parameters
    ----------
    random_state : int, np.random.RandomState, or None, optional, default=None
        Seed used to pick a pattern in ``fit``. If None, the pick can differ
        between runs.

    Attributes
    ----------
    patterns_ : list of tuple of int
        Event pattern of every series seen in ``pretrain``, in the order in
        which the series appear in ``X``. One pattern holds the positions of
        the events of one series, counted from the start of that series, in
        increasing order. Series without known events have an empty pattern.
    pattern_ : tuple of int
        Pattern replayed by ``predict``, picked in ``fit``. One of
        ``patterns_`` if the detector was pretrained, otherwise the pattern
        of the ``y`` passed to ``fit``, otherwise empty.
    cursor_ : int
        Time-from-start of the first point passed in the latest call to
        ``fit`` or ``update``, where ``predict`` starts. 0 after ``fit``.
    n_timepoints_seen_ : int
        Number of time points passed to ``fit`` and ``update`` so far.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyPatternAnomalies
    >>> X_pretrain = pd.DataFrame(
    ...     {"value": range(40)},
    ...     index=pd.MultiIndex.from_product([["a", "b"], range(20)]),
    ... )
    >>> y_pretrain = pd.DataFrame(
    ...     {"ilocs": [5, 15, 10]},
    ...     index=pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ... )
    >>> d = DummyPatternAnomalies(random_state=42)
    >>> d = d.pretrain(X_pretrain, y_pretrain)
    >>> d.patterns_
    [(5, 15), (10,)]
    >>> X = pd.Series(range(30))
    >>> d.fit(X).predict(X)
       ilocs
    0      5
    1     15
    """

    _tags = {
        "authors": ["yash-sangwan"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:pretrain": True,
        "capability:update": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        "fit_is_empty": False,
        "task": "anomaly_detection",
        "learning_type": "supervised",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, random_state=None):
        self.random_state = random_state
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
        return [{}, {"random_state": 42}]

    def _pretrain(self, X, y=None):
        """Store the event pattern of every time series in X.

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets ``patterns_``, one event pattern per series in ``X``.

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
            If None, all patterns are empty, and no alarms will be fired.

        Returns
        -------
        self :
            Reference to self.
        """
        instances = X.index.droplevel(-1).unique()
        events = self._events_by_instance(y)

        self.patterns_ = [events.get(instance, ()) for instance in instances]
        return self

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets ``pattern_``, the pattern replayed by ``predict``.
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
        if hasattr(self, "patterns_"):
            self.pattern_ = self._pick_pattern(self.patterns_)
        elif y is not None:
            self.pattern_ = self._to_pattern(y)
        else:
            self.pattern_ = ()

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
            Only the length of ``X`` is used, its values are ignored.
            It starts at time-from-start ``cursor_``.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.
        """
        # the pattern holds times-from-start, and X starts at cursor_
        start = self.cursor_
        ilocs = [t - start for t in self.pattern_ if start <= t < start + len(X)]

        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()

        return pd.Series(ilocs, dtype="int64")

    def _pick_pattern(self, patterns):
        """Pick one of the stored patterns at random, seeded by random_state."""
        if len(patterns) == 0:
            return ()

        rng = check_random_state(self.random_state)
        return patterns[rng.randint(len(patterns))]

    @classmethod
    def _events_by_instance(cls, y):
        """Group known events y into one event pattern per instance.

        ``pretrain`` has flattened the instance levels of ``y``, so the
        instance of an event is its row index without the last level.

        Parameters
        ----------
        y : pd.DataFrame, or None
            Known events, with row ``MultiIndex`` ``(instance, event_no)``.

        Returns
        -------
        dict
            Event pattern of every instance seen in ``y``, keyed by instance.
        """
        if y is None or len(y) == 0:
            return {}

        instances = y.index
        if isinstance(instances, pd.MultiIndex):
            instances = instances.droplevel(-1)

        grouped = cls._ilocs(y).groupby(np.asarray(instances))
        return {instance: cls._to_pattern(events) for instance, events in grouped}

    @classmethod
    def _to_pattern(cls, y):
        """Return the event positions in y, as a tuple of int in increasing order."""
        return tuple(sorted(int(iloc) for iloc in cls._ilocs(y)))

    @staticmethod
    def _ilocs(y):
        """Return the ``"ilocs"`` column of events ``y``, as a pd.Series."""
        if isinstance(y, pd.DataFrame):
            return y["ilocs"]
        return y
