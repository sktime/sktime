"""DilationMapping transformer test code."""

import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.dilation_mapping import DilationMappingTransformer


@pytest.mark.skipif(
    not run_test_for_class(DilationMappingTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_dilation_mapping_reorders_values():
    """Test that dilation mapping reorders values by dilation groups."""
    X = pd.Series([1, 2, 3, 4, 5], name="y")

    result = DilationMappingTransformer(dilation=2).fit_transform(X)
    expected = pd.Series([1, 3, 5, 2, 4], name="y")

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(
    not run_test_for_class(DilationMappingTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_dilation_mapping_preserves_missing_values():
    """Test that missing values are preserved during dilation mapping."""
    X = pd.Series([1.0, None, 3.0, 4.0, 5.0], name="z")

    result = DilationMappingTransformer(dilation=2).fit_transform(X)
    expected = pd.Series([1.0, 3.0, 5.0, None, 4.0], name="z")

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(
    not run_test_for_class(DilationMappingTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("dilation", [1, 10])
def test_dilation_mapping_boundary_dilations(dilation):
    """Test boundary dilation values preserve values and reset the index."""
    X = pd.Series(
        [10, 20, 30],
        index=pd.date_range("2020-01-01", periods=3),
        name="short",
    )

    result = DilationMappingTransformer(dilation=dilation).fit_transform(X)
    expected = pd.Series([10, 20, 30], name="short")

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(
    not run_test_for_class(DilationMappingTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("dilation", [0, -1])
def test_dilation_mapping_invalid_dilation(dilation):
    """Test that non-positive dilation values raise an error."""
    with pytest.raises(ValueError, match="Dilation must be greater than 0"):
        DilationMappingTransformer(dilation=dilation)
