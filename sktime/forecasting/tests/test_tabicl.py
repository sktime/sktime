"""Tests for TabICLForecaster."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from sktime.forecasting.tabicl import TabICLForecaster


class _DummyTabICLRegressor(BaseEstimator, RegressorMixin):
    """Small sklearn-compatible stand-in for tabicl.TabICLRegressor."""

    def __init__(
        self,
        n_estimators=8,
        norm_methods=None,
        feat_shuffle_method="latin",
        outlier_threshold=4.0,
        batch_size=8,
        kv_cache=False,
        model_path=None,
        allow_auto_download=True,
        checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
        device=None,
        use_amp="auto",
        use_fa3="auto",
        offload_mode="auto",
        disk_offload_dir=None,
        random_state=42,
        n_jobs=None,
        verbose=False,
        inference_config=None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.inference_config = inference_config

    def fit(self, X, y):
        """Fit dummy regressor."""
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Predict constant values."""
        return np.full(len(X), self.mean_)


def test_tabicl_forecaster_delegates_to_reduction(monkeypatch):
    """TabICLForecaster should wrap TabICLRegressor in a reduction forecaster."""

    monkeypatch.setattr(
        TabICLForecaster,
        "_get_tabicl_regressor_class",
        lambda self: _DummyTabICLRegressor,
    )

    y = pd.Series(np.arange(20, dtype=float))
    forecaster = TabICLForecaster(
        n_estimators=2,
        batch_size=4,
        allow_auto_download=False,
        window_length=5,
        ignore_deps=True,
    )

    forecaster.fit(y, fh=[1, 2])
    y_pred = forecaster.predict(fh=[1, 2])

    assert len(y_pred) == 2
    assert forecaster.forecaster_.estimator.n_estimators == 2
    assert forecaster.forecaster_.estimator.batch_size == 4
    assert forecaster.forecaster_.estimator.allow_auto_download is False
