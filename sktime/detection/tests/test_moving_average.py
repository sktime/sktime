"""Tests for MovingAverageDetector."""

import pandas as pd
import pytest

from sktime.detection.moving_average import MovingAverageDetector
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(MovingAverageDetector),
    reason="skip if test not required for this class",
)
class TestMovingAverageDetector:
    """Tests for MovingAverageDetector."""

    def _make_series_with_spike(self, spike_index=5, spike_value=100.0, n=15):
        """Helper: create a near-zero series with a single spike."""
        values = [0.1] * n
        values[spike_index] = spike_value
        return pd.DataFrame(values, columns=["value"])

    def test_detects_obvious_spike(self):
        """Obvious spike at index 5 must be flagged."""
        X = self._make_series_with_spike(spike_index=5, spike_value=100.0)
        detector = MovingAverageDetector(window_size=3, n_sigma=2.0)
        y_pred = detector.fit_transform(X)
        # Row 5 should be labelled 1 (anomaly)
        assert y_pred.loc[5, "labels"] == 1

    def test_no_anomaly_on_flat_series(self):
        """A perfectly flat series should produce no anomalies."""
        X = pd.DataFrame([1.0] * 20, columns=["value"])
        detector = MovingAverageDetector(window_size=5, n_sigma=3.0)
        y_pred = detector.fit_transform(X)
        assert (y_pred["labels"] == 0).all()

    def test_no_anomaly_on_normal_noise(self):
        """Low-amplitude white noise well within 5-sigma should not be flagged.

        The first ``window_size`` rows are skipped because there is insufficient
        baseline history during the warm-up period.
        """
        import numpy as np

        window = 10
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0.0, scale=0.01, size=50)
        X = pd.DataFrame(values, columns=["value"])
        detector = MovingAverageDetector(window_size=window, n_sigma=5.0)
        y_pred = detector.fit_transform(X)
        # Skip warm-up: first window rows may have zero-std baseline
        assert (y_pred["labels"].iloc[window:] == 0).all()

    def test_multiple_spikes_detected(self):
        """All spikes in the series must be detected."""
        values = [0.0] * 30
        spike_positions = [5, 15, 25]
        for pos in spike_positions:
            values[pos] = 200.0
        X = pd.DataFrame(values, columns=["value"])
        detector = MovingAverageDetector(window_size=4, n_sigma=2.0)
        y_pred = detector.fit_transform(X)
        for pos in spike_positions:
            assert y_pred.loc[pos, "labels"] == 1, (
                f"Spike at index {pos} was not detected."
            )

    def test_center_mode_detects_spike(self):
        """Centered window mode should still detect the spike."""
        X = self._make_series_with_spike(spike_index=7, spike_value=50.0)
        detector = MovingAverageDetector(window_size=5, n_sigma=2.0, center=True)
        y_pred = detector.fit_transform(X)
        assert y_pred.loc[7, "labels"] == 1

    def test_output_shape_matches_input(self):
        """Output must have same number of rows as input."""
        X = self._make_series_with_spike()
        detector = MovingAverageDetector(window_size=5, n_sigma=3.0)
        y_pred = detector.fit_transform(X)
        assert len(y_pred) == len(X)

    def test_output_column_name(self):
        """Output DataFrame must have a 'labels' column."""
        X = self._make_series_with_spike()
        detector = MovingAverageDetector(window_size=5, n_sigma=3.0)
        y_pred = detector.fit_transform(X)
        assert "labels" in y_pred.columns

    def test_output_values_are_binary(self):
        """Labels must be 0 or 1 only."""
        X = self._make_series_with_spike()
        detector = MovingAverageDetector(window_size=5, n_sigma=3.0)
        y_pred = detector.fit_transform(X)
        assert set(y_pred["labels"].unique()).issubset({0, 1})

    def test_get_test_params(self):
        """get_test_params must return a list of two valid param dicts."""
        params = MovingAverageDetector.get_test_params()
        assert isinstance(params, list)
        assert len(params) == 2
        for p in params:
            assert isinstance(p, dict)
            # must instantiate without error
            MovingAverageDetector(**p)
