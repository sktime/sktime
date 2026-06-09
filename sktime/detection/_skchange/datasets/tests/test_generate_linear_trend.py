"""Tests for continous piecewise linear data generation."""

import pandas as pd
import pytest

from sktime.detection._skchange.datasets import (
    generate_continuous_piecewise_linear_data,
    generate_continuous_piecewise_linear_signal,
)


def test_generate_continuous_piecewise_linear_data_default():
    df = generate_continuous_piecewise_linear_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


@pytest.mark.parametrize(
    "slopes",
    [
        None,
        1,
        [1],
        [2, 3],
        [0.5, -0.5, 1],
    ],
)
def test_generate_continuous_piecewise_linear_data_valid_slopes(
    slopes: float | list[float],
):
    """Test that invalid slopes raise ValueError."""
    df = generate_continuous_piecewise_linear_data(slopes=slopes)
    assert not df.empty


def test_generate_continuous_piecewise_linear_data_invalid_noise_std():
    """Test that invalid noise_std raises ValueError."""
    with pytest.raises(ValueError):
        generate_continuous_piecewise_linear_data(noise_std=-1)


def test_generate_continuous_piecewise_linear_data_single_segment():
    """Test single-segment generation does not raise and has expected shape."""
    df = generate_continuous_piecewise_linear_data(
        n_segments=1,
        n_samples=25,
        slopes=2.0,
        noise_std=0.0,
        seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (25, 1)


def test_generate_continuous_piecewise_linear_signal_empty_change_points():
    """Test legacy generator supports empty change_points."""
    df = generate_continuous_piecewise_linear_signal(
        change_points=[],
        slopes=[1.5],
        n_samples=20,
        noise_std=0.0,
        random_seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 1)
