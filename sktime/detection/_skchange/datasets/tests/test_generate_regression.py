"""Tests for regression data generation"""

import pandas as pd
import pytest

from sktime.detection._skchange.datasets import generate_piecewise_regression_data


def test_generate_piecewise_regression_data_default():
    """Test default generation of piecewise regression data."""
    df, feature_cols, target_cols = generate_piecewise_regression_data()
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in feature_cols + target_cols)


@pytest.mark.parametrize("lengths", [100, [100], [50, 50], [30, 20, 50]])
def test_generate_piecewise_regression_data_valid_lengths(lengths):
    df, feature_cols, target_cols, params = generate_piecewise_regression_data(
        lengths=lengths, return_params=True
    )
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in feature_cols)
    assert all(col in df.columns for col in target_cols)
    assert isinstance(params, dict)


@pytest.mark.parametrize("lengths", [[], -10, [100, -50]])
def test_generate_piecewise_regression_data_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        generate_piecewise_regression_data(lengths=lengths)
