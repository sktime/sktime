"""Tests for MomentFMDetector."""

__author__ = ["priyanshuharshbodhi1"]

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sktime.tests.test_switch import run_test_for_class


@patch("sktime.detection.momentfm.MOMENTPipeline")
def test_predict_output_shape(mock_pipeline_cls):
    """_predict returns a sparse Series of valid iloc anomaly positions."""
    import torch

    from sktime.detection.momentfm import MomentFMDetector

    if not run_test_for_class(MomentFMDetector):
        return

    n = 600
    X = pd.DataFrame(np.random.default_rng(1).standard_normal((n, 1)))

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
    assert result.dtype == "int64"
    assert (result >= 0).all() and (result < n).all()


@patch("sktime.detection.momentfm.MOMENTPipeline")
def test_predict_multivariate(mock_pipeline_cls):
    """_predict handles multivariate input and flags injected anomalies."""
    import torch

    from sktime.detection.momentfm import MomentFMDetector

    if not run_test_for_class(MomentFMDetector):
        return

    n = 200
    X = pd.DataFrame(np.random.default_rng(2).standard_normal((n, 3)))

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
    assert isinstance(result, pd.Series)
    assert len(result) >= 1
