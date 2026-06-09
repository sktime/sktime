"""Tests for continous piecewise linear data generation."""

import pandas as pd
import pytest

from sktime.detection._skchange.datasets import generate_continuous_piecewise_linear_data


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
