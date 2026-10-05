# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for SavitzkyGolayTransformer."""

__author__ = ["ved197338"]

import numpy as np
import pandas as pd
import pytest

from sktime.transformations.savitzky_golay import SavitzkyGolayTransformer


def test_smoothing_reduces_noise():
    """Smoothing a noisy sine wave should bring it closer to the clean signal."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 4 * np.pi, 100)
    clean = np.sin(t)
    noisy = clean + 0.1 * rng.randn(100)
    series = pd.Series(noisy, name="signal")

    smoother = SavitzkyGolayTransformer(window_length=11, polyorder=2)
    smoothed = smoother.fit_transform(series)

    assert isinstance(smoothed, pd.Series)
    assert len(smoothed) == len(series)
    assert smoothed.name == "signal"

    orig_mse = np.mean((noisy - clean) ** 2)
    smooth_mse = np.mean((smoothed.to_numpy() - clean) ** 2)
    assert smooth_mse < orig_mse, "smoothing should reduce noise"


def test_multivariate_dataframe():
    """Transformer should handle a multi-column DataFrame."""
    df = pd.DataFrame(
        {
            "sin": np.sin(np.linspace(0, 10, 50)),
            "cos": np.cos(np.linspace(0, 10, 50)),
        }
    )
    t = SavitzkyGolayTransformer(window_length=5, polyorder=2, deriv=1)
    result = t.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert list(result.columns) == ["sin", "cos"]


def test_rejects_even_window_length():
    """Even window_length should raise a clear error."""
    with pytest.raises(ValueError, match="must be odd"):
        t = SavitzkyGolayTransformer(window_length=4, polyorder=2)
        t.fit_transform(pd.Series(np.arange(10)))


def test_rejects_polyorder_too_large():
    """polyorder >= window_length should raise."""
    with pytest.raises(ValueError, match="must be less than"):
        t = SavitzkyGolayTransformer(window_length=5, polyorder=5)
        t.fit_transform(pd.Series(np.arange(10)))


def test_direct_numerical_comparison_with_scipy():
    """Output should match scipy.signal.savgol_filter directly."""
    from scipy.signal import savgol_filter

    arr = np.sin(np.linspace(0, 10, 50))
    series = pd.Series(arr)
    t = SavitzkyGolayTransformer(window_length=7, polyorder=2, deriv=0)
    res = t.fit_transform(series)
    expected = savgol_filter(arr, window_length=7, polyorder=2, deriv=0)
    np.testing.assert_allclose(res.to_numpy(), expected)


def test_short_input_length_validation_raises():
    """Data length <= polyorder should raise ValueError."""
    series = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="must be greater than `polyorder`"):
        SavitzkyGolayTransformer(window_length=5, polyorder=2).fit_transform(series)
