# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for multivariate capability guard in BaseDetector._check_X."""

__author__ = ["rupeshca007"]

import pandas as pd
import pytest

from sktime.detection.dummy import DummyRegularAnomalies
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_input_raises_for_univariate_detector():
    """Univariate detectors should raise ValueError on multivariate input.

    Before this fix, passing a 2-column DataFrame to a detector with
    ``capability:multivariate=False`` would silently produce wrong results.
    After the fix, a clear ValueError is raised immediately, guiding the user
    to switch to a multivariate-capable detector.

    This is especially important for multi-sensor time series (e.g., vibration,
    acoustics, mechanics) where users may inadvertently pass all sensor channels
    to a univariate-only algorithm.
    """
    # DummyRegularAnomalies has capability:multivariate = True — use a
    # minimal subclass that sets it to False to test the guard properly.
    from sktime.detection.base import BaseDetector

    class _UnivariateDummy(BaseDetector):
        """Minimal detector that only supports univariate input."""

        _tags = {
            "task": "anomaly_detection",
            "learning_type": "unsupervised",
            "capability:multivariate": False,  # univariate only
            "fit_is_empty": True,
        }

        def __init__(self):
            super().__init__()

        def _predict(self, X):
            return self._empty_sparse()

        @classmethod
        def get_test_params(cls, parameter_set="default"):
            return {}

    X_multivariate = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    detector = _UnivariateDummy()

    with pytest.raises(ValueError, match="does not support multivariate"):
        detector.fit(X_multivariate)


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_input_accepted_for_multivariate_detector():
    """Multivariate detectors must accept multi-column DataFrames without error."""
    X_multivariate = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    # DummyRegularAnomalies sets capability:multivariate = True
    detector = DummyRegularAnomalies(step_size=2)
    # Should not raise
    detector.fit(X_multivariate)
    result = detector.predict(X_multivariate)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_univariate_input_accepted_for_univariate_detector():
    """Univariate detector must still accept single-column DataFrames."""
    X_univariate = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    detector = DummyRegularAnomalies(step_size=2)
    detector.fit(X_univariate)
    result = detector.predict(X_univariate)
    assert isinstance(result, pd.DataFrame)
