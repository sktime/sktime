"""Tests for detection model evaluation."""

__author__ = ["Nischal1425"]

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from sktime.detection.dummy import DummyRegularAnomalies, DummyRegularChangePoints
from sktime.detection.model_evaluation import evaluate
from sktime.performance_metrics.detection import (
    DirectedChamfer,
    DirectedHausdorff,
    WindowedF1Score,
)
from sktime.tests.test_switch import run_test_module_changed


def _make_detection_data(n_timepoints=100, random_state=42):
    """Create synthetic detection data with ground truth events.

    Returns
    -------
    X : pd.DataFrame
        Time series data.
    y : pd.DataFrame
        Ground truth events with ilocs column.
    """
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({"value": rng.randn(n_timepoints)})
    # place ground truth anomalies every 10 steps
    event_ilocs = list(range(9, n_timepoints, 10))
    y = pd.DataFrame({"ilocs": event_ilocs})
    return X, y


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.detection"]),
    reason="run test only if detection module has changed",
)
class TestDetectionEvaluate:
    """Tests for the detection evaluate function."""

    def test_evaluate_basic(self):
        """Test basic evaluate with DummyRegularAnomalies and WindowedF1Score."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)
        scoring = WindowedF1Score(margin=2)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=scoring,
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "test_WindowedF1Score" in results.columns
        assert "fit_time" in results.columns
        assert "pred_time" in results.columns
        assert "len_train_window" in results.columns

    def test_evaluate_multiple_metrics(self):
        """Test evaluate with multiple detection metrics."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)
        scorers = [WindowedF1Score(margin=2), DirectedChamfer()]

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=scorers,
        )

        assert "test_WindowedF1Score" in results.columns
        assert "test_DirectedChamfer" in results.columns
        assert len(results) == 3

    def test_evaluate_return_data(self):
        """Test evaluate with return_data=True returns data columns."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=WindowedF1Score(margin=2),
            return_data=True,
        )

        assert "y_train" in results.columns
        assert "y_test" in results.columns
        assert "y_pred" in results.columns
        assert "X_train" in results.columns
        assert "X_test" in results.columns

    def test_evaluate_different_cv_splits(self):
        """Test evaluate with different numbers of CV splits."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        for n_splits in [2, 3, 5]:
            results = evaluate(
                detector=detector,
                X=X,
                y=y,
                cv=KFold(n_splits=n_splits, shuffle=False),
                scoring=WindowedF1Score(margin=2),
            )
            assert len(results) == n_splits

    def test_evaluate_change_point_detector(self):
        """Test evaluate with DummyRegularChangePoints."""
        X, y = _make_detection_data()
        detector = DummyRegularChangePoints(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=WindowedF1Score(margin=2),
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_evaluate_default_scoring(self):
        """Test that default scoring is WindowedF1Score."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
        )

        assert "test_WindowedF1Score" in results.columns

    def test_evaluate_default_cv(self):
        """Test that default cv is KFold with n_splits=3."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            scoring=WindowedF1Score(margin=2),
        )

        # default is KFold(n_splits=3, shuffle=False)
        assert len(results) == 3

    def test_evaluate_int_cv(self):
        """Test passing an integer for cv parameter."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=5,
            scoring=WindowedF1Score(margin=2),
        )

        assert len(results) == 5

    def test_evaluate_unsupervised(self):
        """Test evaluate without ground truth labels."""
        X, _ = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        # pass y=None (unsupervised)
        results = evaluate(
            detector=detector,
            X=X,
            y=None,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=WindowedF1Score(margin=2),
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_evaluate_timing(self):
        """Test that fit_time and pred_time are positive."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=WindowedF1Score(margin=2),
        )

        assert (results["fit_time"] >= 0).all()
        assert (results["pred_time"] >= 0).all()

    def test_evaluate_series_input(self):
        """Test evaluate with pd.Series input for X."""
        rng = np.random.RandomState(42)
        X = pd.Series(rng.randn(100), name="value")
        y = pd.DataFrame({"ilocs": [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]})
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=WindowedF1Score(margin=2),
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_evaluate_timeseries_split(self):
        """Test evaluate with TimeSeriesSplit for temporal CV."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=TimeSeriesSplit(n_splits=3),
            scoring=WindowedF1Score(margin=2),
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_evaluate_scores_are_numeric(self):
        """Test that all scores are numeric (float)."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=[WindowedF1Score(margin=2), DirectedChamfer()],
        )

        for col in ["test_WindowedF1Score", "test_DirectedChamfer"]:
            assert pd.api.types.is_numeric_dtype(results[col])

    def test_evaluate_error_score_numeric(self):
        """Test error_score with numeric value doesn't raise on bad estimator."""
        X, y = _make_detection_data(n_timepoints=10)

        from sktime.detection.base import BaseDetector

        class FailingDetector(BaseDetector):
            """Detector that always fails on fit."""

            _tags = {
                "task": "anomaly_detection",
                "learning_type": "unsupervised",
                "fit_is_empty": False,
            }

            def _fit(self, X, y=None):
                raise RuntimeError("Intentional failure")

            def _predict(self, X):
                return pd.DataFrame({"ilocs": []})

            @classmethod
            def get_test_params(cls, parameter_set="default"):
                return [{}]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = evaluate(
                detector=FailingDetector(),
                X=X,
                y=y,
                cv=KFold(n_splits=2, shuffle=False),
                scoring=WindowedF1Score(),
                error_score=np.nan,
            )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert results["test_WindowedF1Score"].isna().all()

    def test_evaluate_invalid_detector_type(self):
        """Test that non-BaseDetector raises TypeError."""
        X, y = _make_detection_data()

        with pytest.raises(TypeError, match="must be a BaseDetector"):
            evaluate(detector="not_a_detector", X=X, y=y)

    def test_evaluate_invalid_scorer_type(self):
        """Test that invalid scorer type raises TypeError."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        with pytest.raises(TypeError, match="BaseDetectionMetric"):
            evaluate(
                detector=detector,
                X=X,
                y=y,
                scoring=lambda y_true, y_pred: 0.5,
            )

    def test_evaluate_hausdorff_metric(self):
        """Test evaluate with DirectedHausdorff metric."""
        X, y = _make_detection_data()
        detector = DummyRegularAnomalies(step_size=5)

        results = evaluate(
            detector=detector,
            X=X,
            y=y,
            cv=KFold(n_splits=3, shuffle=False),
            scoring=DirectedHausdorff(),
        )

        assert "test_DirectedHausdorff" in results.columns
        assert len(results) == 3


# need to import warnings for the test above
import warnings  # noqa: E402
