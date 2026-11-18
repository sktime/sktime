"""Tests for the native PatchTST forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.patch_tst import PatchTSTForecaster

torch = pytest.importorskip("torch")


def test_patch_tst_native_forecaster_predicts_requested_horizon():
    """Native PatchTST fits a noisy multivariate series and preserves FH."""
    rng = np.random.RandomState(7)
    index = pd.RangeIndex(40)
    y = pd.DataFrame(
        {
            "sin": np.sin(np.arange(40) / 3) + rng.normal(0, 0.02, 40),
            "cos": np.cos(np.arange(40) / 3) + rng.normal(0, 0.02, 40),
        },
        index=index,
    )
    forecaster = PatchTSTForecaster(
        patch_len=4,
        stride=2,
        d_model=8,
        n_heads=2,
        dropout=0.0,
        context_length=12,
        max_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        warmup_steps=1,
        random_state=7,
    )

    y_pred = forecaster.fit(y, fh=[1, 3, 5]).predict()

    assert y_pred.shape == (3, 2)
    assert list(y_pred.index) == [40, 42, 44]
    assert forecaster.model_.patches_seen_ > 0


def test_patch_tst_native_backpropagates_through_shared_backbone():
    """PatchTST backbone receives gradients from its forecasting head."""
    model = PatchTSTForecaster(
        patch_len=3, stride=1, d_model=8, n_heads=2, context_length=8
    )._make_model(n_channels=2, prediction_length=2)
    x = torch.randn(4, 8, 2)
    loss = model(x).square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in model.backbone.parameters()]
    assert any(
        gradient is not None and torch.any(gradient != 0) for gradient in gradients
    )
