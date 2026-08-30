# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for multivariate capability guard in BaseDetector._check_X.

Covers all four combinations of detector capability vs input dimensionality,
as requested in https://github.com/sktime/sktime/issues/9943
"""

__author__ = ["rupeshca007"]

import pandas as pd
import pytest

from sktime.detection.base import BaseDetector
from sktime.detection.dummy import DummyRegularAnomalies
from sktime.tests.test_switch import run_test_for_class


# ---------------------------------------------------------------------------
# Shared fixtures: minimal detector stubs
# ---------------------------------------------------------------------------


class _UnivariateDummy(BaseDetector):
    """Minimal detector that declares capability:multivariate = False."""

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": False,
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__()

    def _predict(self, X):
        return self._empty_sparse()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}


class _MultivariateDummy(BaseDetector):
    """Minimal detector that declares capability:multivariate = True."""

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__()

    def _predict(self, X):
        return self._empty_sparse()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}


# Shared data
X_univariate = pd.DataFrame({"sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0]})
X_multivariate = pd.DataFrame(
    {
        "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "acoustics": [5.0, 4.0, 3.0, 2.0, 1.0],
    }
)


# ---------------------------------------------------------------------------
# Combo 1: univariate detector + univariate input → OK
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_univariate_detector_univariate_input_ok():
    """Combo 1: univariate-only detector accepts single-column input without error."""
    detector = _UnivariateDummy()
    # Should not raise
    detector.fit(X_univariate)
    result = detector.predict(X_univariate)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Combo 2: univariate detector + multivariate input → ValueError  ← core fix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_univariate_detector_multivariate_input_raises():
    """Combo 2: univariate-only detector raises ValueError on multi-column input.

    Before the fix, this silently produced incorrect results.
    After the fix, a clear ValueError is raised immediately at fit() time,
    guiding the user to switch to a multivariate-capable detector.
    """
    detector = _UnivariateDummy()
    with pytest.raises(ValueError, match="does not support multivariate"):
        detector.fit(X_multivariate)


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_univariate_detector_multivariate_input_raises_on_predict():
    """Combo 2b: guard also fires at predict() time if fit was with univariate."""
    detector = _UnivariateDummy()
    detector.fit(X_univariate)  # fit on univariate — OK
    with pytest.raises(ValueError, match="does not support multivariate"):
        detector.predict(X_multivariate)  # predict on multivariate — should raise


# ---------------------------------------------------------------------------
# Combo 3: multivariate detector + univariate input → OK
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_detector_univariate_input_ok():
    """Combo 3: multivariate-capable detector accepts single-column input."""
    detector = _MultivariateDummy()
    detector.fit(X_univariate)
    result = detector.predict(X_univariate)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Combo 4: multivariate detector + multivariate input → OK
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_detector_multivariate_input_ok():
    """Combo 4: multivariate-capable detector accepts multi-column input."""
    detector = _MultivariateDummy()
    detector.fit(X_multivariate)
    result = detector.predict(X_multivariate)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_error_message_contains_column_count():
    """ValueError message should include the detected column count."""
    detector = _UnivariateDummy()
    with pytest.raises(ValueError, match="2"):
        detector.fit(X_multivariate)


@pytest.mark.skipif(
    not run_test_for_class(DummyRegularAnomalies),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_error_message_contains_class_name():
    """ValueError message should include the detector class name."""
    detector = _UnivariateDummy()
    with pytest.raises(ValueError, match="_UnivariateDummy"):
        detector.fit(X_multivariate)
