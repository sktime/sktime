"""Dummy anomaly detector which fires as learnt in pretrain, by one of 3 strategies."""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.detection.base import BaseDetector

_STRATEGIES = ("rate", "pattern", "timefreq")


class DummyPretrainedAnomalies(BaseDetector):
    """Dummy anomaly detector which fires as learnt in pretrain.

    Naive method that can serve as a baseline, or as an API test for the
    pretrain API of detectors.

    In ``pretrain``, learns from a collection of time series and their known
    events, and in ``predict``, fires alarms as learnt, ignoring the values of
    the time series passed. What is learnt depends on ``strategy``:

    * ``"rate"``: the average event rate, i.e., the number of events per time
      point. ``predict`` fires at regular distance ``step = round(1 / rate)``:
      at every time-from-start ``t`` with ``(t + 1) % step == 0``, so the first
      alarm of a series is at ``iloc`` position ``step - 1``.
    * ``"pattern"``: the event pattern of every time series, i.e., the
      positions of its known events, counted from the start of the series.
      ``fit`` picks one of the patterns at random, and ``predict`` replays it.
    * ``"timefreq"``: for every time-from-start, how often an event happened
      there. The count is divided by the number of series that are long enough
      to reach that time-from-start, so a short series does not shrink the
      probability of a late time-from-start. An event after the end of its own
      series is ignored. ``predict`` fires at each time-from-start with the
      probability learnt for it, and fires nothing beyond the last
      time-from-start seen in ``pretrain``.

    A second call to ``pretrain`` replaces what the first call learnt.
    Alarms are ``iloc`` references to the time series passed to ``predict``,
    and alarms beyond the end of that series are dropped.

    If ``pretrain`` was not called, the detector learns in ``fit``, from the
    known events ``y`` of the series fitted to: the rate of that series for
    ``"rate"``, its pattern for ``"pattern"``, and probability one at each
    time-from-start where it had an event, zero elsewhere, for ``"timefreq"``.
    If no ``y`` is passed to ``fit`` either, no alarms are fired.

    The series is treated as a stream. ``fit`` starts the stream at
    time-from-start 0, and ``update`` adds the points it is passed to the
    stream, without refitting. ``predict`` labels the latest points passed to
    ``fit`` or ``update``, at their time-from-start, so ``update_predict`` on
    consecutive chunks fires the same alarms as ``predict`` on the whole
    series at once. For ``"timefreq"``, this holds only if ``random_state`` is
    an integer, as otherwise every call draws new random numbers.
    Each chunk passed to ``update`` must follow the points already seen, with
    no overlap and no gap, as time-from-start is counted in points.

    Parameters
    ----------
    strategy : str, one of "rate", "pattern", "timefreq", default="rate"
        What is learnt in ``pretrain``, and how alarms are fired, see above.
    random_state : int, np.random.RandomState, or None, optional, default=None
        Seed for the random parts of the strategy. Not used by ``"rate"``.
        For ``"pattern"``, seeds the pick of a pattern in ``fit``.
        For ``"timefreq"``, seeds the alarms drawn in ``predict``, one draw per
        time-from-start. With an integer seed, a time-from-start gets the same
        draw in every call, so ``predict`` fires the same alarms for the same
        ``cursor_`` and length of ``X``.
        If None, the pick and the alarms can differ between runs.

    Attributes
    ----------
    pretrain_event_rate_ : float
        ``"rate"`` only. Average number of events per time point, learnt in
        ``pretrain``.
    n_pretrain_events_ : int
        ``"rate"`` only. Number of events seen in ``pretrain``.
    n_pretrain_timepoints_ : int
        ``"rate"`` only. Number of time points seen in ``pretrain``.
    event_rate_ : float
        ``"rate"`` only. Rate used by ``predict``, set in ``fit``. The rate
        from ``pretrain`` if the detector was pretrained, otherwise the rate
        of the ``y`` passed to ``fit``, otherwise 0.
    patterns_ : list of tuple of int
        ``"pattern"`` only. Event pattern of every series seen in ``pretrain``,
        in the order in which the series appear in ``X``. One pattern holds
        the positions of the events of one series, counted from the start of
        that series, in increasing order. Series without known events have an
        empty pattern.
    pattern_ : tuple of int
        ``"pattern"`` only. Pattern replayed by ``predict``, picked in ``fit``.
        One of ``patterns_`` if the detector was pretrained, otherwise the
        pattern of the ``y`` passed to ``fit``, otherwise empty.
    event_counts_ : np.ndarray of int
        ``"timefreq"`` only. Number of series seen in ``pretrain`` that had an
        event at a given time-from-start. Position ``t`` is the
        time-from-start ``t``, and the length is that of the longest series
        seen. A series with more than one event at the same time-from-start is
        counted once.
    n_at_risk_ : np.ndarray of int
        ``"timefreq"`` only. Number of series seen in ``pretrain`` that are
        long enough to reach a given time-from-start, i.e., the denominator of
        ``event_prob_``.
    event_prob_ : np.ndarray of float
        ``"timefreq"`` only. ``event_counts_`` divided by ``n_at_risk_``, and
        zero where no series is at risk. The probability of firing at a
        time-from-start.
    prob_ : np.ndarray of float
        ``"timefreq"`` only. Probabilities used by ``predict``, set in ``fit``.
        ``event_prob_`` if the detector was pretrained, otherwise one at each
        time-from-start where the ``y`` passed to ``fit`` had an event,
        otherwise empty.
    cursor_ : int
        Time-from-start of the first point passed in the latest call to
        ``fit`` or ``update``, where ``predict`` starts. 0 after ``fit``.
    n_timepoints_seen_ : int
        Number of time points passed to ``fit`` and ``update`` so far.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyPretrainedAnomalies
    >>> X_pretrain = pd.DataFrame(
    ...     {"value": range(40)},
    ...     index=pd.MultiIndex.from_product([["a", "b"], range(20)]),
    ... )
    >>> y_pretrain = pd.DataFrame(
    ...     {"ilocs": [5, 15, 10]},
    ...     index=pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ... )
    >>> X = pd.Series(range(30))

    With ``"rate"``, 3 events in 40 time points, so an alarm every 13 points:

    >>> d = DummyPretrainedAnomalies(strategy="rate")
    >>> d = d.pretrain(X_pretrain, y_pretrain)
    >>> d.pretrain_event_rate_
    0.075
    >>> d.fit(X).predict(X)
       ilocs
    0     12
    1     25

    With ``"pattern"``, the events of one of the pretrained series are replayed:

    >>> d = DummyPretrainedAnomalies(strategy="pattern", random_state=42)
    >>> d = d.pretrain(X_pretrain, y_pretrain)
    >>> d.patterns_
    [(5, 15), (10,)]
    >>> d.fit(X).predict(X)
       ilocs
    0      5
    1     15

    With ``"timefreq"``, one series out of two had an event at 5, 10 and 15:

    >>> d = DummyPretrainedAnomalies(strategy="timefreq", random_state=42)
    >>> d = d.pretrain(X_pretrain, y_pretrain)
    >>> [float(d.event_prob_[t]) for t in [5, 10, 15]]
    [0.5, 0.5, 0.5]
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

    def __init__(self, strategy="rate", random_state=None):
        self.strategy = strategy
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
        return [
            {},
            {"strategy": "pattern", "random_state": 42},
            {"strategy": "timefreq", "random_state": 42},
        ]

    def _check_strategy(self):
        """Return strategy if it is one of the known strategies, otherwise raise."""
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy of {type(self).__name__} must be one of "
                f"{_STRATEGIES}, but found {self.strategy!r}."
            )
        return self.strategy

    def _pretrain(self, X, y=None):
        """Learn from a collection of time series, as per strategy.

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            ``"rate"``: sets ``pretrain_event_rate_``, ``n_pretrain_events_``,
            and ``n_pretrain_timepoints_``.
            ``"pattern"``: sets ``patterns_``, one event pattern per series.
            ``"timefreq"``: sets ``event_counts_``, ``n_at_risk_``, and
            ``event_prob_``.

        Parameters
        ----------
        X : pd.DataFrame with 2-level row MultiIndex (instance, time)
            Time series to pretrain on. Hierarchical data has been flattened
            to this format by ``pretrain``.
        y : pd.DataFrame, optional
            Known events in ``X``, one row per event, with an ``"ilocs"``
            column, and row ``MultiIndex`` ``(instance, event_no)``.
            An ``"ilocs"`` entry is a time-from-start of its own series.
            ``pretrain`` has flattened the instance levels, so the instance
            ``("h0_0", "h1_0")`` is ``"h0_0__h1_0"`` here.
            Events of instances that are not in ``X`` are ignored, and for
            ``"timefreq"``, so are events after the end of their own series.
            If None, no events are seen, and no alarms will be fired.

        Returns
        -------
        self :
            Reference to self.

        Raises
        ------
        ValueError
            If ``strategy`` is not one of ``"rate"``, ``"pattern"``,
            ``"timefreq"``.
        """
        strategy = self._check_strategy()

        if strategy == "rate":
            self._pretrain_rate(X, y)
        elif strategy == "pattern":
            self._pretrain_pattern(X, y)
        else:
            self._pretrain_timefreq(X, y)
        return self

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            ``"rate"``: sets ``event_rate_``, the rate used by ``predict``.
            ``"pattern"``: sets ``pattern_``, the pattern replayed by
            ``predict``.
            ``"timefreq"``: sets ``prob_``, the probabilities used by
            ``predict``.
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

        Raises
        ------
        ValueError
            If ``strategy`` is not one of ``"rate"``, ``"pattern"``,
            ``"timefreq"``.
        """
        strategy = self._check_strategy()

        if strategy == "rate":
            self._fit_rate(X, y)
        elif strategy == "pattern":
            self._fit_pattern(X, y)
        else:
            self._fit_timefreq(X, y)

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

        Nothing is written to self. For ``"timefreq"``, the random numbers are
        drawn from a generator made inside this method.

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
        if self.strategy == "rate":
            ilocs = self._predict_rate(X)
        elif self.strategy == "pattern":
            ilocs = self._predict_pattern(X)
        else:
            ilocs = self._predict_timefreq(X)

        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()

        return pd.Series(ilocs, dtype="int64")

    # strategy "rate"
    # ---------------

    def _pretrain_rate(self, X, y):
        """Learn the average event rate from a collection of time series."""
        instances = X.index.droplevel(-1).unique()

        self.n_pretrain_events_ = self._count_events(y, instances)
        self.n_pretrain_timepoints_ = len(X)
        self.pretrain_event_rate_ = self._rate(
            self.n_pretrain_events_, self.n_pretrain_timepoints_
        )

    def _fit_rate(self, X, y):
        """Set the rate used by predict, from pretrain, else from y, else 0."""
        if hasattr(self, "pretrain_event_rate_"):
            self.event_rate_ = self.pretrain_event_rate_
        elif y is not None:
            self.event_rate_ = self._rate(len(y), len(X))
        else:
            self.event_rate_ = 0.0

    def _predict_rate(self, X):
        """Return ilocs in X of the alarms fired at regular distance."""
        rate = self.event_rate_

        if rate <= 0:
            return []

        step = max(1, int(round(1 / rate)))
        # alarms are at every time-from-start t with (t + 1) % step == 0, and X
        # starts at time-from-start cursor_, so the first alarm in X is at
        first = (step - 1 - self.cursor_) % step
        return np.arange(first, len(X), step)

    @staticmethod
    def _rate(n_events, n_timepoints):
        """Return events per time point, 0 if there are no time points."""
        if n_timepoints == 0:
            return 0.0
        return n_events / n_timepoints

    @classmethod
    def _count_events(cls, y, instances):
        """Count events in y that belong to one of the instances.

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

        return int(cls._instances_of(y).isin(instances).sum())

    # strategy "pattern"
    # ------------------

    def _pretrain_pattern(self, X, y):
        """Store the event pattern of every time series in X."""
        instances = X.index.droplevel(-1).unique()
        events = self._events_by_instance(y)

        self.patterns_ = [events.get(instance, ()) for instance in instances]

    def _fit_pattern(self, X, y):
        """Set the pattern replayed by predict, picked from pretrain, else y."""
        if hasattr(self, "patterns_"):
            self.pattern_ = self._pick_pattern(self.patterns_)
        elif y is not None:
            self.pattern_ = self._to_pattern(y)
        else:
            self.pattern_ = ()

    def _predict_pattern(self, X):
        """Return ilocs in X of the events of the replayed pattern."""
        # the pattern holds times-from-start, and X starts at cursor_
        start = self.cursor_
        return [t - start for t in self.pattern_ if start <= t < start + len(X)]

    def _pick_pattern(self, patterns):
        """Pick one of the stored patterns at random, seeded by random_state."""
        if len(patterns) == 0:
            return ()

        rng = check_random_state(self.random_state)
        return patterns[rng.randint(len(patterns))]

    @classmethod
    def _events_by_instance(cls, y):
        """Group known events y into one event pattern per instance.

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

        instances = np.asarray(cls._instances_of(y))
        grouped = cls._raw_ilocs(y).groupby(instances)
        return {instance: cls._to_pattern(events) for instance, events in grouped}

    @classmethod
    def _to_pattern(cls, y):
        """Return the event positions in y, as a tuple of int in increasing order.

        Positions are converted one by one with ``int``, so a NaN position
        raises a ``ValueError``.
        """
        return tuple(sorted(int(iloc) for iloc in cls._raw_ilocs(y)))

    @staticmethod
    def _raw_ilocs(y):
        """Return the ``"ilocs"`` column of events ``y`` as it is, without casting."""
        if isinstance(y, pd.DataFrame):
            return y["ilocs"]
        return y

    # strategy "timefreq"
    # -------------------

    def _pretrain_timefreq(self, X, y):
        """Count how often an event happened at each time-from-start."""
        lengths = X.index.droplevel(-1).value_counts()
        n_offsets = int(lengths.max())

        # a series is at risk at time-from-start t if it is longer than t
        at_risk = lengths.to_numpy()[:, None] > np.arange(n_offsets)

        self.n_at_risk_ = at_risk.sum(axis=0)
        self.event_counts_ = self._event_counts(y, lengths, n_offsets)
        self.event_prob_ = self._to_prob(self.event_counts_, self.n_at_risk_)

    def _fit_timefreq(self, X, y):
        """Set the probabilities used by predict, from pretrain, else from y."""
        if hasattr(self, "event_prob_"):
            self.prob_ = self.event_prob_
        elif y is not None:
            self.prob_ = self._prob_of_series(y, len(X))
        else:
            self.prob_ = np.zeros(0, dtype="float64")

    def _predict_timefreq(self, X):
        """Return ilocs in X of the alarms drawn with the learnt probabilities."""
        # X starts at time-from-start cursor_, and nothing is fired beyond
        # the last time-from-start seen in pretrain
        start = self.cursor_
        prob = self.prob_[start : start + len(X)]

        if len(prob) == 0:
            return []

        # one draw per time-from-start, so a time-from-start gets the same
        # draw whichever chunk it is in
        rng = check_random_state(self.random_state)
        draws = rng.uniform(size=len(self.prob_))[start : start + len(prob)]
        return np.flatnonzero(draws < prob)

    @classmethod
    def _event_counts(cls, y, lengths, n_offsets):
        """Count series with an event at each time-from-start.

        Parameters
        ----------
        y : pd.DataFrame, or None
            Known events, with row ``MultiIndex`` ``(instance, event_no)``.
        lengths : pd.Series
            Length of every time series pretrained on, indexed by instance.
        n_offsets : int
            Length of the longest series pretrained on.

        Returns
        -------
        np.ndarray of int, of length ``n_offsets``
            Number of series with an event at a given time-from-start.
        """
        counts = np.zeros(n_offsets, dtype="int64")

        if y is None or len(y) == 0:
            return counts

        events = pd.DataFrame(
            {"instance": np.asarray(cls._instances_of(y)), "offset": cls._ilocs(y)}
        )
        # events of unknown instances are ignored, and one series counts
        # at most once per time-from-start
        events = events[events["instance"].isin(lengths.index)].drop_duplicates()

        offsets = events["offset"].to_numpy()
        # an event cannot happen after the end of its own series
        own_length = events["instance"].map(lengths).to_numpy()
        offsets = offsets[(offsets >= 0) & (offsets < own_length)]

        return np.bincount(offsets, minlength=n_offsets).astype("int64")

    @classmethod
    def _prob_of_series(cls, y, n_timepoints):
        """Return probability one at each time-from-start with an event in y."""
        prob = np.zeros(n_timepoints, dtype="float64")

        offsets = cls._ilocs(y)
        offsets = offsets[(offsets >= 0) & (offsets < n_timepoints)]
        prob[offsets] = 1.0

        return prob

    @staticmethod
    def _to_prob(counts, n_at_risk):
        """Divide counts by the number at risk, and return zero where none is."""
        prob = np.zeros(len(counts), dtype="float64")
        np.divide(counts, n_at_risk, out=prob, where=n_at_risk > 0)
        return prob

    @staticmethod
    def _ilocs(y):
        """Return the ``"ilocs"`` column of events ``y``, as an int np.ndarray."""
        if isinstance(y, pd.DataFrame):
            y = y["ilocs"]
        return np.asarray(y, dtype="int64")

    # known events y, shared by all strategies
    # ----------------------------------------

    @staticmethod
    def _instances_of(y):
        """Return the instance of every event in y.

        ``pretrain`` has flattened the instance levels of ``y``, so the
        instance of an event is its row index without the last level.
        """
        instances = y.index
        if isinstance(instances, pd.MultiIndex):
            instances = instances.droplevel(-1)
        return instances
