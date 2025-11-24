"""Tests for TSFeaturesTransformer."""

__author__ = ["Faakhir30"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_arrow_head
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.tsfeatures import (
    TSFeaturesTransformer,
    TSFeaturesWideTransformer,
)
from sktime.utils._testing.panel import make_classification_problem


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_extractor():
    """Test that TSFeaturesTransformer extracts features correctly."""
    X, _ = make_classification_problem()

    transformer = TSFeaturesTransformer(freq=1, scale=True, threads=1)

    Xt = transformer.fit_transform(X)

    # Check that output is a DataFrame
    assert isinstance(Xt, type(X))
    # Check that we have features (columns)
    assert Xt.shape[1] > 0
    # Check that we have the same number of instances (rows)
    assert Xt.shape[0] == X.shape[0]
    # Check that all values are numeric (no object dtype)
    assert Xt.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_docs_tsfeatures_transformer():
    """Test whether doc example runs through."""
    X, y = load_arrow_head(return_X_y=True)
    transformer = TSFeaturesTransformer(freq=1, scale=True)
    Xt = transformer.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("freq", [1, 12])
@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("threads", [1, None])
def test_tsfeatures_parameters(freq, scale, threads):
    """Test TSFeaturesTransformer with different parameter combinations."""
    X, _ = make_classification_problem()

    transformer = TSFeaturesTransformer(freq=freq, scale=scale, threads=threads)
    Xt = transformer.fit_transform(X)

    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] > 0
    assert Xt.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_empty_input():
    """Test TSFeaturesTransformer with empty input raises error."""
    X = pd.DataFrame({"dim_0": []})

    transformer = TSFeaturesTransformer(freq=1)
    with pytest.raises(ValueError, match="No objects to concatenate"):
        transformer.fit_transform(X)


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_custom_features():
    """Test TSFeaturesTransformer with custom feature functions."""
    from tsfeatures.tsfeatures import hurst, series_length

    X, _ = make_classification_problem()

    transformer = TSFeaturesTransformer(freq=1, features=[series_length, hurst])
    Xt = transformer.fit_transform(X)

    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] >= 2  # At least series_length and hurst features
    assert "series_length" in Xt.columns
    assert "hurst" in Xt.columns


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesWideTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_wide_extractor():
    """Test that TSFeaturesWideTransformer extracts features correctly."""
    # Create wide format data
    data = pd.DataFrame(
        {
            "unique_id": ["ts1", "ts2", "ts3"],
            "seasonality": [12, 4, 1],
            "y": [np.random.randn(100), np.random.randn(50), np.random.randn(75)],
        }
    )

    transformer = TSFeaturesWideTransformer(scale=True, threads=1)
    Xt = transformer.fit_transform(data)

    # Check that output is a DataFrame
    assert isinstance(Xt, pd.DataFrame)
    # Check that we have features (columns)
    assert Xt.shape[1] > 0
    # Check that we have the same number of instances (rows)
    assert Xt.shape[0] == data.shape[0]
    # Check that all feature columns are numeric (exclude 'unique_id' if present)
    feature_cols = [col for col in Xt.columns if col != "unique_id"]
    assert all(np.issubdtype(Xt[col].dtype, np.number) for col in feature_cols)


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesWideTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_docs_tsfeatures_wide_transformer():
    """Test whether doc example runs through."""
    data = pd.DataFrame(
        {
            "unique_id": ["ts1", "ts2"],
            "seasonality": [12, 4],
            "y": [np.random.randn(100), np.random.randn(50)],
        }
    )
    transformer = TSFeaturesWideTransformer(scale=True)
    Xt = transformer.fit_transform(data)
    assert Xt.shape[0] == data.shape[0]
    assert Xt.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesWideTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("threads", [1, None])
def test_tsfeatures_wide_parameters(scale, threads):
    """Test TSFeaturesWideTransformer with different parameter combinations."""
    data = pd.DataFrame(
        {
            "unique_id": ["ts1", "ts2"],
            "seasonality": [12, 4],
            "y": [np.random.randn(100), np.random.randn(50)],
        }
    )

    transformer = TSFeaturesWideTransformer(scale=scale, threads=threads)
    Xt = transformer.fit_transform(data)

    assert Xt.shape[0] == data.shape[0]
    assert Xt.shape[1] > 0
    # Check that all feature columns are numeric (exclude 'unique_id' if present)
    feature_cols = [col for col in Xt.columns if col != "unique_id"]
    assert all(np.issubdtype(Xt[col].dtype, np.number) for col in feature_cols)


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesWideTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_wide_missing_columns():
    """Test TSFeaturesWideTransformer raises error for missing columns."""
    # Missing 'seasonality' column
    data = pd.DataFrame(
        {"unique_id": ["ts1", "ts2"], "y": [np.random.randn(100), np.random.randn(50)]}
    )

    transformer = TSFeaturesWideTransformer()
    with pytest.raises(ValueError, match="X must have columns"):
        transformer.fit_transform(data)


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesWideTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_wide_custom_features():
    """Test TSFeaturesWideTransformer with custom feature functions."""
    from tsfeatures.tsfeatures import hurst, series_length

    data = pd.DataFrame(
        {
            "unique_id": ["ts1", "ts2"],
            "seasonality": [12, 4],
            "y": [np.random.randn(100), np.random.randn(50)],
        }
    )

    transformer = TSFeaturesWideTransformer(features=[series_length, hurst])
    Xt = transformer.fit_transform(data)

    assert Xt.shape[0] == data.shape[0]
    assert Xt.shape[1] >= 2  # At least series_length and hurst features
    assert "series_length" in Xt.columns
    assert "hurst" in Xt.columns
