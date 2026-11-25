"""Compositors that control stream and refitting behaviour of detector updates."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly", "yash-sangwan", "Adnan9-63"]
__all__ = ["UpdateRefitsEvery", "RefitDetector"]

import pandas as pd

from sktime.datatypes import VectorizedDF
from sktime.datatypes._utilities import get_window
from sktime.detection.base import BaseDetector


class _DetectorStreamDataPoolMixin:
    """Pool training data as ``_X`` / ``_y`` on detector stream compositors.

    BaseDetector does not retain fit/update data when remember_data=False.
    Stream wrappers that need a pooled history (for periodic refit / interval
    logic) own that storage.
    """

    def _init_stream_data_pool(self):
        self._X = None
        self._y = None

    def _update_stream_data_pool(self, X, y=None):
        """Append ``X`` / ``y`` to the stream pool."""
        if X is not None:
            if isinstance(X, VectorizedDF):
                X = X.X_multiindex
            if self._X is None or not self.is_fitted:
                self._X = X
            else:
                self._X = X.combine_first(self._X)

        if y is not None:
            if isinstance(y, VectorizedDF):
                y = y.X_multiindex
            if self._y is None or not self.is_fitted:
                self._y = y
            else:
                self._y = y.combine_first(self._y)


def _lag_arg(lag):
    """Map lag=0 to None for ``get_window`` (``Timestamp - 0`` is invalid)."""
    return None if lag == 0 else lag


def _is_time_offset(obj):
    """Check whether obj is a pd.DateOffset."""
    return hasattr(pd, "DateOffset") and isinstance(obj, pd.DateOffset)


def _is_time_difference(obj):
    """Check whether obj is a time difference (pd.Timedelta or pd.DateOffset)."""
    if isinstance(obj, pd.Timedelta):
        return True
    if _is_time_offset(obj):
        return True
    return False


def _geq(a, b):
    """Check whether time difference a is bigger equal than b."""
    if b is None:
        return False
    if _is_time_offset(a) and _is_time_offset(b):
        return a.n >= b.n
    return a >= b


def _extract_cutoff_scalar(cutoff):
    """Unwrap scalar from cutoff or index-like container."""
    if cutoff is None:
        return None
    if isinstance(cutoff, (pd.Index, pd.Series, list, tuple)):
        return cutoff[0]
    return cutoff


class UpdateRefitsEvery(_DetectorStreamDataPoolMixin, BaseDetector):
    """Refits a detector periodically when update is called.

    If update is called and ``refit_interval`` or more has elapsed since the last
    ``fit``, refits the wrapped ``detector`` instead (call to ``fit``).

    Refitting is done on (potentially) all data seen so far.

    ``refit_window_size`` and ``refit_window_lag`` control the lookback window
    on which refitting is done.

    Parameters
    ----------
    detector : BaseDetector
        The detector to be refit/updated regularly.
    refit_interval : int or timedelta, optional, default=0
        Interval that needs to elapse after which the update triggers a refit.
        Default = 0, i.e., always refits, never updates.
    refit_window_size : int or timedelta, optional, default=None
        Length of the data window to refit to in case update calls fit;
        default = None, i.e., refits to entire training data seen so far.
    refit_window_lag : int or timedelta, optional, default=0
        Lag of the data window to refit to, in case update calls fit;
        default = 0, i.e., refit window ends with the latest cutoff.

    Examples
    --------
    >>> from sktime.detection.dummy import DummyRegularAnomalies
    >>> from sktime.detection.compose import UpdateRefitsEvery
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({"a": np.arange(20)})
    >>> X0 = X.iloc[:10]
    >>> X1 = X.iloc[10:15]
    >>> X2 = X.iloc[15:]
    >>> detector = UpdateRefitsEvery(DummyRegularAnomalies(), refit_interval=5)
    >>> detector.fit(X0)
    UpdateRefitsEvery(...)
    >>> detector.update(X1)
    UpdateRefitsEvery(...)
    >>> y_pred = detector.predict(X2)
    """

    _tags = {
        "authors": ["fkiraly", "yash-sangwan", "Adnan9-63"],
        "category": ["composition"],
        "learning_type": "unsupervised",
        "fit_is_empty": False,
        "capability:update": True,
        "tests:core": True,
    }

    def __init__(
        self,
        detector,
        refit_interval=0,
        refit_window_size=None,
        refit_window_lag=0,
    ):
        self.detector = detector
        self.detector_ = detector.clone()
        self.refit_interval = refit_interval
        self.refit_window_size = refit_window_size
        self.refit_window_lag = refit_window_lag

        super().__init__()
        self._init_stream_data_pool()

        tags_to_clone = [
            "task",
            "learning_type",
            "capability:missing_values",
            "capability:multivariate",
            "capability:unequal_length",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(self.detector_, tag_names=tags_to_clone)
        self.set_tags(**{"capability:update": True, "fit_is_empty": False})

    def _fit(self, X, y=None):
        """Fit detector to training data."""
        self._init_stream_data_pool()
        self._X = X
        self._y = y
        cutoff = getattr(self, "_cutoff", None)
        if cutoff is not None:
            self._last_refit_cutoff = cutoff
        elif hasattr(X, "index") and len(X.index) > 0:
            self._last_refit_cutoff = X.index[-1]
        else:
            self._last_refit_cutoff = len(X) - 1
        self.last_fit_cutoff_ = _extract_cutoff_scalar(self._last_refit_cutoff)
        self.detector_ = self.detector.clone()
        self.detector_.fit(X=X, y=y)
        return self

    def _update(self, X, y=None):
        """Update detector with incremental training data."""
        self._update_stream_data_pool(X=X, y=y)
        cutoff = getattr(self, "_cutoff", None)
        if cutoff is not None:
            curr_cutoff = cutoff
        elif hasattr(X, "index") and len(X.index) > 0:
            curr_cutoff = X.index[-1]
        else:
            curr_cutoff = len(self._X) - 1

        curr_scalar = _extract_cutoff_scalar(curr_cutoff)
        last_scalar = _extract_cutoff_scalar(self.last_fit_cutoff_)
        delta_cutoff = curr_scalar - last_scalar
        delta_cutoff = _extract_cutoff_scalar(delta_cutoff)

        time_since_last_fit = delta_cutoff
        refit_interval = self.refit_interval
        refit_window_size = self.refit_window_size
        refit_window_lag = self.refit_window_lag

        _X = self._X
        _y = self._y

        if _is_time_difference(time_since_last_fit):
            if isinstance(refit_window_lag, int) and refit_window_lag != 0:
                lag = min(refit_window_lag, len(_X))
                refit_window_lag = curr_scalar - _X.index[-lag]
            if isinstance(refit_window_size, int):
                _X_lag = get_window(_X, lag=_lag_arg(refit_window_lag))
                window_size = min(refit_window_size + 1, len(_X_lag))
                refit_window_size = curr_scalar - _X_lag.index[-window_size]
            if isinstance(refit_interval, int):
                index = min(refit_interval + 1, len(_X))
                refit_interval = curr_scalar - _X.index[-index]

        if _geq(time_since_last_fit, refit_interval):
            if refit_window_size is not None or refit_window_lag != 0:
                _X = get_window(
                    _X,
                    window_length=refit_window_size,
                    lag=_lag_arg(refit_window_lag),
                )
                if _y is not None:
                    _y = get_window(
                        _y,
                        window_length=refit_window_size,
                        lag=_lag_arg(refit_window_lag),
                    )
            self.detector_.fit(X=_X, y=_y)
            self._last_refit_cutoff = curr_cutoff
            self.last_fit_cutoff_ = curr_scalar
        else:
            self.detector_.update(X=X, y=y)
        return self

    def _predict(self, X):
        return self.detector_.predict(X)

    def _transform(self, X):
        return self.detector_.transform(X)

    def _predict_points(self, X):
        return self.detector_.predict_points(X)

    def _predict_segments(self, X):
        return self.detector_.predict_segments(X)

    def _predict_scores(self, X):
        return self.detector_.predict_scores(X)

    def _transform_scores(self, X):
        return self.detector_.transform_scores(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection.dummy import DummyRegularAnomalies

        detector = DummyRegularAnomalies()
        param1 = {"detector": detector, "refit_interval": 0}
        param2 = {"detector": detector, "refit_interval": 5}
        return [param1, param2]


class RefitDetector(UpdateRefitsEvery):
    """Refits a detector on every update.

    Shorthand for ``UpdateRefitsEvery(detector, refit_interval=0)``.

    Parameters
    ----------
    detector : BaseDetector
        The detector to refit on every update.
    refit_window_size : int or timedelta, optional, default=None
        Length of the data window to refit to.
    refit_window_lag : int or timedelta, optional, default=0
        Lag of the data window to refit to.

    Examples
    --------
    >>> from sktime.detection.dummy import DummyRegularAnomalies
    >>> from sktime.detection.compose import RefitDetector
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({"a": np.arange(20)})
    >>> X0 = X.iloc[:10]
    >>> X1 = X.iloc[10:]
    >>> detector = RefitDetector(DummyRegularAnomalies())
    >>> detector.fit(X0)
    RefitDetector(...)
    >>> detector.update(X1)
    RefitDetector(...)
    >>> y_pred = detector.predict(X1)
    """

    _tags = {
        "authors": ["fkiraly", "yash-sangwan", "Adnan9-63"],
        "category": ["composition"],
        "learning_type": "unsupervised",
        "fit_is_empty": False,
        "capability:update": True,
        "tests:core": True,
    }

    def __init__(self, detector, refit_window_size=None, refit_window_lag=0):
        super().__init__(
            detector=detector,
            refit_interval=0,
            refit_window_size=refit_window_size,
            refit_window_lag=refit_window_lag,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.detection.dummy import DummyRegularAnomalies

        detector = DummyRegularAnomalies()
        param1 = {"detector": detector}
        param2 = {"detector": detector, "refit_window_size": 10}
        return [param1, param2]
