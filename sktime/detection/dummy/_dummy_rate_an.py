"""Dummy anomaly detector which fires at the average pretrained event rate."""

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.utils.multiindex import flatten_multiindex


class DummyRateAnomalies(BaseDetector):
    """Dummy anomaly detector which fires at the average event rate seen in pretrain.

    Naive method that can serve as a chance level baseline, or as an API test
    for the pretrain API of detectors.

    In ``pretrain``, learns the average event rate, i.e., the number of events
    per time point, over a collection of time series and their known events.
    A second call to ``pretrain`` replaces the rate learnt by the first call.

    In ``predict``, fires alarms at that rate on the time series passed, at
    regular distance ``step = round(1 / rate)``, with the first alarm at
    ``iloc`` position ``step - 1``. Alarms are ``iloc`` references to the
    time series passed to ``predict``.

    If ``pretrain`` was not called, the rate is learnt in ``fit``, from the
    known events ``y`` of the series fitted to. If no ``y`` is passed to
    ``fit`` either, no alarms are fired.

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

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.dummy import DummyRateAnomalies
    >>> X_pretrain = pd.DataFrame(
    ...     {"value": range(40)},
    ...     index=pd.MultiIndex.from_product([["a", "b"], range(20)]),
    ... )
    >>> y_pretrain = pd.DataFrame({"ilocs": [5, 15, 10]}, index=["a", "a", "b"])
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
            column, indexed by the instance the event belongs to.
            Instance labels are flattened as ``pretrain`` flattens ``X``,
            so the instance ``("h0_0", "h1_0")`` becomes ``"h0_0__h1_0"``.
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
        return self

    def _predict(self, X):
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
        """
        rate = self.event_rate_

        if rate <= 0:
            return BaseDetector._empty_sparse()

        step = max(1, int(round(1 / rate)))
        ilocs = np.arange(step - 1, len(X), step)

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

        Instance labels of ``y`` are flattened the same way as ``pretrain``
        flattens the instance levels of ``X``.

        Parameters
        ----------
        y : pd.DataFrame, or None
            Known events, indexed by the instance the event belongs to.
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
            y_instances = flatten_multiindex(y_instances)

        return int(y_instances.isin(instances).sum())
