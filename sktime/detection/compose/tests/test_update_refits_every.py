"""Tests for UpdateRefitsEveryDetector."""

import pandas as pd
import pytest

from sktime.detection.compose import UpdateRefitsEveryDetector
from sktime.detection.dummy import DummyRegularAnomalies
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(UpdateRefitsEveryDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_update_refits_on_all_data():
    """Fit and update give the same predictions as one fit on all data."""
    X = pd.DataFrame({"a": range(20)}, dtype=float)

    detector = UpdateRefitsEveryDetector(DummyRegularAnomalies())
    detector.fit(X.iloc[:10])
    detector.update(X.iloc[10:])
    y_pred = detector.predict(X)

    y_expected = DummyRegularAnomalies().fit(X).predict(X)

    pd.testing.assert_frame_equal(y_pred, y_expected)
    # the inner detector is refit on all data seen so far
    pd.testing.assert_frame_equal(detector.detector_._X, X)


@pytest.mark.skipif(
    not run_test_for_class(UpdateRefitsEveryDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_update_appends_chunks_with_restarting_index():
    """Chunks whose index restarts at 0 are appended, not overwritten."""
    X0 = pd.DataFrame({"a": range(10)}, dtype=float)
    X1 = pd.DataFrame({"a": range(10, 20)}, dtype=float)
    X = pd.concat([X0, X1], ignore_index=True)

    detector = UpdateRefitsEveryDetector(DummyRegularAnomalies())
    detector.fit(X0)
    detector.update(X1)

    assert len(detector._X_pool) == 20
    pd.testing.assert_frame_equal(detector._X_pool, X)

    y_pred = detector.predict(X)
    y_expected = DummyRegularAnomalies().fit(X).predict(X)
    pd.testing.assert_frame_equal(y_pred, y_expected)
