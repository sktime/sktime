# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Tests for the CAPA detector."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection._anomaly_scores._l2_saving import L2Saving
from sktime.detection._change_scores._from_cost import ChangeScore
from sktime.detection._compose import PenalisedScore
from sktime.detection.capa import CAPA
from sktime.detection.costs import L1Cost, L2Cost, MultivariateGaussianCost
from sktime.tests.test_switch import run_test_module_changed


def _make_anomaly_data(
    n=200, p=1, anomaly_start=80, anomaly_end=100, shift=20.0, seed=42
):
    """Generate test data with a known anomaly segment."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)))
    X.iloc[anomaly_start:anomaly_end] += shift
    return X


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
class TestCAPADetection:
    """Tests for CAPA anomaly detection quality."""

    def test_capa_detects_segment_anomaly(self):
        """Test CAPA detects a strong segment anomaly."""
        X = _make_anomaly_data(n=200, anomaly_start=80, anomaly_end=100, seed=42)
        detector = CAPA(
            min_segment_length=5,
            max_segment_length=100,
            segment_penalty=30.0,
        )
        anomalies = detector.fit_predict(X)["ilocs"]

        assert len(anomalies) >= 1
        assert hasattr(anomalies.iloc[0], "left")
        assert hasattr(anomalies.iloc[0], "right")

    def test_capa_detects_with_l2_cost(self):
        """Test CAPA with L2Cost as segment saving."""
        X = _make_anomaly_data(n=200, anomaly_start=80, anomaly_end=100, seed=42)
        detector = CAPA(
            segment_saving=L2Cost(param=0.0),
            point_saving=L1Cost(param=0.0),
            min_segment_length=5,
            max_segment_length=100,
            segment_penalty=30.0,
        )
        anomalies = detector.fit_predict(X)["ilocs"]
        assert len(anomalies) >= 1

    def test_capa_multivariate_with_affected_components(self):
        """Test CAPA on multivariate data with find_affected_components."""
        X = _make_anomaly_data(
            n=200, p=3, anomaly_start=80, anomaly_end=100, shift=20.0, seed=42
        )
        detector = CAPA(
            min_segment_length=5,
            max_segment_length=100,
            find_affected_components=True,
        )
        anomalies = detector.fit_predict(X)["ilocs"]
        assert len(anomalies) >= 1

    def test_capa_segment_length_enforcement(self):
        """Test that all detected anomalies respect min_segment_length."""
        min_seg = 5
        detector = CAPA(
            segment_penalty=0.0,
            min_segment_length=min_seg,
            max_segment_length=100,
        )
        rng = np.random.default_rng(13)
        X = pd.DataFrame(rng.standard_normal((100, 1)))
        anomalies = detector.fit_predict(X)["ilocs"]

        if len(anomalies) > 0:
            lengths = anomalies.array.right - anomalies.array.left
            assert np.all(lengths >= min_seg)

    def test_capa_point_anomalies(self):
        """Test CAPA detects a point anomaly."""
        rng = np.random.default_rng(134)
        X = pd.DataFrame(rng.standard_normal((200, 1)))
        point_iloc = 100
        X.iloc[point_iloc] += 50.0

        detector = CAPA(min_segment_length=5, max_segment_length=100)
        anomalies = detector.fit_predict(X)["ilocs"]

        assert len(anomalies) >= 1
        found = any(a.left <= point_iloc < a.right for a in anomalies)
        assert found

    def test_capa_ignore_point_anomalies(self):
        """Test that ignore_point_anomalies parameter works without error."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((200, 1)))

        detector = CAPA(
            min_segment_length=5,
            max_segment_length=100,
            ignore_point_anomalies=True,
        )
        result = detector.fit_predict(X)
        assert "ilocs" in result.columns

    def test_capa_empty_result(self):
        """Test CAPA with very high penalty produces no anomalies."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((200, 1)))
        detector = CAPA(
            min_segment_length=5,
            max_segment_length=100,
            segment_penalty=1e6,
        )
        result = detector.fit_predict(X)
        assert "ilocs" in result.columns
        assert len(result) == 0


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
class TestCAPAValidation:
    """Tests for CAPA parameter and input validation."""

    def test_point_saving_min_size(self):
        """Test point_saving must have min_size == 1."""
        cost = MultivariateGaussianCost([0.0, np.eye(2)])
        with pytest.raises(ValueError):
            CAPA(point_saving=cost)

    def test_min_segment_length_too_small(self):
        """Test min_segment_length must be >= 2."""
        with pytest.raises(ValueError):
            CAPA(min_segment_length=1)

    def test_max_less_than_min_segment_length(self):
        """Test max_segment_length must be >= min_segment_length."""
        with pytest.raises(ValueError):
            CAPA(min_segment_length=5, max_segment_length=4)

    def test_invalid_saving_string(self):
        """Test CAPA rejects string as segment_saving."""
        with pytest.raises(ValueError, match="segment_saving"):
            CAPA("l2")

    def test_invalid_saving_change_score(self):
        """Test CAPA rejects ChangeScore as segment_saving."""
        with pytest.raises(ValueError, match="segment_saving"):
            CAPA(ChangeScore(L2Cost()))

    def test_penalised_saving_not_penalised_score(self):
        """Test saving tagged as penalised but not PenalisedScore is rejected."""
        score = L2Saving()
        score.set_tags(is_penalised=True)
        with pytest.raises(ValueError, match="penalised"):
            CAPA(segment_saving=score)
        with pytest.raises(ValueError, match="penalised"):
            CAPA(point_saving=score)

    def test_valid_penalised_point_saving(self):
        """Test that PenalisedScore is accepted as point_saving."""
        score = L2Saving()
        penalised = PenalisedScore(score)
        CAPA(point_saving=penalised)
