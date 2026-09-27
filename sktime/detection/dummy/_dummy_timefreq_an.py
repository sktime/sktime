"""Dummy anomaly detector firing at the event frequency of each time-from-start."""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.detection.base import BaseDetector


class DummyTimeFreqAnomalies(BaseDetector):
    """Dummy anomaly detector firing at the event frequency of each time-from-start.

    Naive method that can serve as a baseline, or as an API test for the
    pretrain API of detectors.

    In ``pretrain``, counts, for every time-from-start, how often an event
    happened there, over the time series seen. The count is divided by the
    number of series that are long enough to reach that time-from-start, so
    a short series does not shrink the probability of a late time-from-start.
    A second call to ``pretrain`` replaces the counts of the first call.

    In ``predict``, fires at each time-from-start with the probability learnt
    for it, ignoring the values of the time series passed. Alarms are ``iloc``
    references to the time series passed to ``predict``. Nothing is fired
    beyond the last time-from-start seen in ``pretrain``.

    If ``pretrain`` was not called, the probabilities are taken in ``fit``,
    from the known events ``y`` of the series fitted to: probability one at
    each time-from-start where that series had an event, zero elsewhere.
    If no ``y`` is passed to ``fit`` either, no alarms are fired.

    Parameters
    ----------
    random_state : int, np.random.RandomState, or None, optional, default=None
        Seed used to draw the alarms in ``predict``. With an integer seed,
        ``predict`` fires the same alarms every time it is passed a series of
        the same length. If None, the alarms can differ between runs.

    Attributes
    ----------
    event_counts_ : np.ndarray of int
        Number of series seen in ``pretrain`` that had an event at a given
        time-from-start. Position ``t`` is the time-from-start ``t``, and the
        length is that of the longest series seen. A series with more than one
        event at the same time-from-start is counted once.
    n_at_risk_ : np.ndarray of int
        Number of series seen in ``pretrain`` that are long enough to reach a
        given time-from-start, i.e., the denominator of ``event_prob_``.
    event_prob_ : np.ndarray of float
        ``event_counts_`` divided by ``n_at_risk_``, and zero where no series
        is at risk. The probability of firing at a time-from-start.
    prob_ : np.ndarray of float
        Probabilities used by ``predict``, set in ``fit``. ``event_prob_`` if
        the detector was pretrained, otherwise one at each time-from-start
        where the ``y`` passed to ``fit`` had an event, otherwise empty.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyTimeFreqAnomalies
    >>> X_pretrain = pd.DataFrame(
    ...     {"value": range(40)},
    ...     index=pd.MultiIndex.from_product([["a", "b"], range(20)]),
    ... )
    >>> y_pretrain = pd.DataFrame(
    ...     {"ilocs": [5, 5, 10]},
    ...     index=pd.MultiIndex.from_tuples([("a", 0), ("b", 0), ("b", 1)]),
    ... )
    >>> d = DummyTimeFreqAnomalies(random_state=42)
    >>> d = d.pretrain(X_pretrain, y_pretrain)
    >>> float(d.event_prob_[5])  # both series had an event at time-from-start 5
    1.0
    >>> float(d.event_prob_[10])  # one series out of two
    0.5
    >>> X = pd.Series(range(20))
    >>> d.fit(X).predict(X)
       ilocs
    0      5
    1     10
    """

    _tags = {
        "authors": ["yash-sangwan"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:pretrain": True,
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
        """Count how often an event happened at each time-from-start.

        private _pretrain containing the core logic, called from pretrain

        Writes to self:
            Sets ``event_counts_``, ``n_at_risk_``, and ``event_prob_``.

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
            Events of instances that are not in ``X`` are ignored, and so
            are events after the end of their own series.
            If None, all probabilities are zero, and no alarms will be fired.

        Returns
        -------
        self :
            Reference to self.
        """
        lengths = X.index.droplevel(-1).value_counts()
        n_offsets = int(lengths.max())

        # a series is at risk at time-from-start t if it is longer than t
        at_risk = lengths.to_numpy()[:, None] > np.arange(n_offsets)

        self.n_at_risk_ = at_risk.sum(axis=0)
        self.event_counts_ = self._event_counts(y, lengths, n_offsets)
        self.event_prob_ = self._to_prob(self.event_counts_, self.n_at_risk_)
        return self

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets ``prob_``, the probabilities used by ``predict``.

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
        if hasattr(self, "event_prob_"):
            self.prob_ = self.event_prob_
        elif y is not None:
            self.prob_ = self._prob_of_series(y, len(X))
        else:
            self.prob_ = np.zeros(0, dtype="float64")
        return self

    def _predict(self, X):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Draws one random number per time-from-start, from a generator made
        inside this method, so nothing is written to self.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.
            Only the length of ``X`` is used, its values are ignored.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.
        """
        # nothing is fired beyond the last time-from-start seen in pretrain
        prob = self.prob_[: len(X)]

        if len(prob) == 0:
            return BaseDetector._empty_sparse()

        rng = check_random_state(self.random_state)
        ilocs = np.flatnonzero(rng.uniform(size=len(prob)) < prob)

        if len(ilocs) == 0:
            return BaseDetector._empty_sparse()

        return pd.Series(ilocs, dtype="int64")

    @classmethod
    def _event_counts(cls, y, lengths, n_offsets):
        """Count series with an event at each time-from-start.

        ``pretrain`` has flattened the instance levels of ``y``, so the
        instance of an event is its row index without the last level.

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

        y_instances = y.index
        if isinstance(y_instances, pd.MultiIndex):
            y_instances = y_instances.droplevel(-1)

        events = pd.DataFrame(
            {"instance": np.asarray(y_instances), "offset": cls._ilocs(y)}
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
