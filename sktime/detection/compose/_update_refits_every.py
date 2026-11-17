"""Detector wrapper that refits on update."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd

from sktime.detection.base import BaseDetector

__all__ = ["UpdateRefitsEveryDetector"]


class UpdateRefitsEveryDetector(BaseDetector):
    """Refit a detector on all stored data when update is called.

    Parameters
    ----------
    detector : sktime detector, i.e., estimator inheriting from BaseDetector
        the detector to be refit on update

    Attributes
    ----------
    detector_ : sktime detector
        clone of ``detector``, fitted to all data seen so far

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.compose import UpdateRefitsEveryDetector
    >>> from sktime.detection.dummy import DummyRegularAnomalies
    >>> X = pd.DataFrame({"a": range(20)}, dtype=float)
    >>> X0 = X.iloc[:10]
    >>> X1 = X.iloc[10:]
    >>> detector = UpdateRefitsEveryDetector(DummyRegularAnomalies())
    >>> detector.fit(X0)
    UpdateRefitsEveryDetector(...)
    >>> # update refits the inner detector on X0 and X1 together
    >>> detector.update(X1)
    UpdateRefitsEveryDetector(...)
    >>> y_pred = detector.predict(X)
    """

    _tags = {
        "authors": "yash-sangwan",
        "capability:update": True,
        "fit_is_empty": False,
        "learning_type": "unsupervised",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, detector):
        self.detector = detector

        super().__init__()

        tags_to_clone = [
            "learning_type",
            "task",
            "capability:multivariate",
            "capability:missing_values",
        ]
        self.clone_tags(self.detector, tags_to_clone)

        # own data store, BaseDetector writes to _X and _y itself
        self._X_pool = None
        self._y_pool = None

    def _fit(self, X, y=None):
        """Reset the store to ``X`` and ``y``, then fit a clone of ``detector``."""
        self._X_pool = X
        self._y_pool = y

        self.detector_ = self.detector.clone()
        self.detector_.fit(X=self._X_pool, y=self._y_pool)
        return self

    def _update(self, X, y=None):
        """Refit on all stored data."""
        n_stored = len(self._X_pool)
        self._X_pool = _append_X(self._X_pool, X)
        self._y_pool = _append_y(self._y_pool, y, offset=n_stored)

        self.detector_ = self.detector.clone()
        self.detector_.fit(X=self._X_pool, y=self._y_pool)
        return self

    def _predict(self, X):
        """Call ``predict`` of the fitted ``detector_``."""
        return self.detector_.predict(X)

    def _predict_points(self, X):
        """Call ``predict_points`` of the fitted ``detector_``."""
        return self.detector_.predict_points(X)

    def _predict_segments(self, X):
        """Call ``predict_segments`` of the fitted ``detector_``."""
        return self.detector_.predict_segments(X)

    def _predict_scores(self, X):
        """Call ``predict_scores`` of the fitted ``detector_``."""
        return self.detector_.predict_scores(X)

    def _transform_scores(self, X):
        """Call ``transform_scores`` of the fitted ``detector_``."""
        return self.detector_.transform_scores(X)

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
            Parameters to create testing instances of the class
        """
        from sktime.detection.dummy import DummyRegularAnomalies

        params0 = {"detector": DummyRegularAnomalies()}
        params1 = {"detector": DummyRegularAnomalies(step_size=3)}

        return [params0, params1]


def _append_X(X_old, X_new):
    """Append the rows of ``X_new`` after the rows of ``X_old``.

    If the index of ``X_new`` restarts or overlaps the stored index,
    the result gets a new ``RangeIndex``.
    """
    if X_old is None:
        return X_new
    X_all = pd.concat([X_old, X_new])
    if not X_all.index.is_unique:
        X_all = X_all.reset_index(drop=True)
    return X_all


def _append_y(y_old, y_new, offset):
    """Append the events of ``y_new`` after the events of ``y_old``.

    The ``"ilocs"`` of ``y_new`` point into the new chunk of ``X``,
    so they are shifted by ``offset``, the number of rows stored before it.
    """
    if y_new is None:
        return y_old

    y_new = y_new.copy()
    ilocs = y_new["ilocs"]
    if isinstance(ilocs.dtype, pd.IntervalDtype):
        y_new["ilocs"] = pd.arrays.IntervalArray.from_arrays(
            ilocs.array.left + offset,
            ilocs.array.right + offset,
            closed=ilocs.array.closed,
        )
    else:
        y_new["ilocs"] = ilocs + offset

    if y_old is None:
        return y_new
    return pd.concat([y_old, y_new], ignore_index=True)
