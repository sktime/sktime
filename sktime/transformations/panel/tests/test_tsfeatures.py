"""Tests for TSFeaturesTransformer."""

__author__ = ["Faakhir30"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.tsfeatures import TSFeaturesTransformer
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_extractor():
    """Test that TSFeaturesTransformer extracts features correctly."""
    X = _make_series()

    transformer = TSFeaturesTransformer(scale=True)

    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    # Check that we have features (columns)
    assert Xt.shape[1] > 0
    # Check that we have the same number of instances (rows)
    assert Xt.shape[0] == 1
    # Check that all values are numeric (no object dtype)
    assert Xt.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_auto_frequency_calc():
    """Test TSFeaturesTransformer with auto frequency calculation."""
    X = _make_series(n_timepoints=2)

    transformer = TSFeaturesTransformer()
    with pytest.raises(ValueError, match="Need at least 3 dates to infer frequency"):
        transformer.fit_transform(X)


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_custom_features():
    """Test TSFeaturesTransformer with custom feature functions."""
    from tsfeatures.tsfeatures import hurst, series_length

    X = _make_series()

    transformer = TSFeaturesTransformer(features=[series_length, hurst])
    Xt = transformer.fit_transform(X)

    assert Xt.shape[0] == 1
    assert Xt.shape[1] >= 2  # At least series_length and hurst features
    assert "series_length" in Xt.columns
    assert "hurst" in Xt.columns


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_with_range_index():
    """Test TSFeaturesTransformer with Series that has RangeIndex (non-DatetimeIndex)"""
    X = pd.Series(np.random.randn(50), index=pd.RangeIndex(start=0, stop=50))

    transformer = TSFeaturesTransformer()
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 1
    assert Xt.shape[1] > 0  # Has features
    # Check that all values are numeric
    assert Xt.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_with_panel_data():
    """Test TSFeaturesTransformer with Panel data (nested_univ format / RangeIndex)."""
    X = _make_panel_X(n_instances=5, n_columns=1, n_timepoints=30, return_numpy=False)

    transformer = TSFeaturesTransformer()
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 5
    assert Xt.shape[1] > 0  # Has features
    assert Xt.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
