# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for NNetAR forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.nnetar import NNetAR


def test_nnetar_predicts_requested_horizon():
    """Test that NNetAR returns predictions for the requested horizon."""
    y = pd.Series(np.sin(np.arange(30) / 3) + np.arange(30) * 0.1)
    forecaster = NNetAR(
        p=3, P=0, size=2, repeats=2, max_iter=50, random_state=42
    ).fit(y)

    y_pred = forecaster.predict(fh=[1, 2, 3])

    assert isinstance(y_pred, pd.Series)
    assert list(y_pred.index) == [30, 31, 32]
    assert len(y_pred) == 3
    assert np.isfinite(y_pred).all()


def test_nnetar_uses_seasonal_lags_without_nonseasonal_lags():
    """Test seasonal lag configuration when p=0."""
    y = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0], 5))
    forecaster = NNetAR(
        p=0, P=1, sp=4, size=2, repeats=1, max_iter=50, random_state=7
    ).fit(y)

    assert list(forecaster.lags_) == [4]
    y_pred = forecaster.predict(fh=[1, 2])

    assert len(y_pred) == 2
    assert np.isfinite(y_pred).all()


def test_nnetar_requires_at_least_one_lag():
    """Test error when no autoregressive lag is configured."""
    y = pd.Series(np.arange(10, dtype=float))

    with pytest.raises(ValueError, match="at least one lag"):
        NNetAR(p=0, P=0).fit(y)
