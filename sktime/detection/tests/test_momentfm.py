"""Tests for MomentFMDetector."""

__author__ = ["priyanshuharshbodhi1"]

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class("MomentFMDetector"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_import():
    from sktime.detection.momentfm import MomentFMDetector

    assert MomentFMDetector is not None


def test_instantiation():
    """Check that default and non-default params construct ok."""
    from sktime.detection.momentfm import MomentFMDetector

    det = MomentFMDetector()
    assert det.threshold_percentile == 90
    assert det.anomaly_criterion == "mse"
    assert det.batch_size == 32
    assert det.device == "auto"
    assert det.to_cpu_after_predict is False

    det2 = MomentFMDetector(
        anomaly_criterion="mae",
        threshold_percentile=95,
        batch_size=16,
        to_cpu_after_predict=True,
    )
    assert det2.threshold_percentile == 95
    assert det2.anomaly_criterion == "mae"


def test_tags():
    """Verify the key tags are set correctly."""
    from sktime.detection.momentfm import MomentFMDetector

    assert MomentFMDetector.get_class_tag("task") == "anomaly_detection"
    assert MomentFMDetector.get_class_tag("learning_type") == "unsupervised"
    assert MomentFMDetector.get_class_tag("capability:multivariate") is True


@patch("sktime.detection.momentfm.MOMENTPipeline")
def test_fit_calls_from_pretrained(mock_pipeline_cls):
    """_fit should load the model in RECONSTRUCTION mode."""
    from sktime.detection.momentfm import MomentFMDetector

    mock_model = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_model

    X = pd.DataFrame(np.random.default_rng(0).standard_normal((100, 1)))

    det = MomentFMDetector()
    det._fit(X)

    mock_pipeline_cls.from_pretrained.assert_called_once()
    call_kwargs = mock_pipeline_cls.from_pretrained.call_args
    model_kwargs = call_kwargs[1].get("model_kwargs", {})
    assert "task_name" in model_kwargs

    mock_model.init.assert_called_once()


@patch("sktime.detection.momentfm.MOMENTPipeline")
def test_predict_output_shape(mock_pipeline_cls):
    """_predict should return a Series of 0/1 with same index as X."""
    import torch

    from sktime.detection.momentfm import MomentFMDetector

    n = 600
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((n, 1)))

    mock_model = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_model

    def fake_detect(x_enc, input_mask, anomaly_criterion="mse"):
        b, c, s = x_enc.shape
        out = MagicMock()
        out.anomaly_scores = torch.ones(b, c, s)
        return out

    mock_model.detect_anomalies.side_effect = fake_detect
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)

    det = MomentFMDetector(batch_size=64)
    det.model = mock_model
    det._device = "cpu"
    result = det._predict(X)
    assert isinstance(result, pd.Series)
    assert len(result) == n
    assert result.index.equals(X.index)
    assert set(result.unique()).issubset({0, 1})


@patch("sktime.detection.momentfm.MOMENTPipeline")
def test_predict_multivariate(mock_pipeline_cls):
    """Should handle multivariate input without errors."""
    import torch

    from sktime.detection.momentfm import MomentFMDetector

    n = 200
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((n, 3)))

    mock_model = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_model

    def fake_detect(x_enc, input_mask, anomaly_criterion="mse"):
        b, c, s = x_enc.shape
        out = MagicMock()
        scores = torch.zeros(b, c, s)
        scores[:, :, 10] = 100.0
        out.anomaly_scores = scores
        return out

    mock_model.detect_anomalies.side_effect = fake_detect
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)

    det = MomentFMDetector(threshold_percentile=99)
    det.model = mock_model
    det._device = "cpu"

    result = det._predict(X)
    assert len(result) == n
    assert result.sum() >= 1


def test_get_test_params():
    """get_test_params should return a list of two dicts."""
    from sktime.detection.momentfm import MomentFMDetector

    params = MomentFMDetector.get_test_params()
    assert isinstance(params, list)
    assert len(params) == 2
    for p in params:
        assert isinstance(p, dict)
        det = MomentFMDetector(**p)
        assert det is not None
