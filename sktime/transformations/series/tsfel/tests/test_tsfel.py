#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for TSFEL transformer functionality."""

__author__ = ["Faakhir30"]

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.tsfel import TSFELTransformer
from sktime.utils._testing.series import _make_series


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_basic_functionality():
    """Test basic functionality with default parameters."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features=None, verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] > 0
    assert X_transformed.shape[1] > 0  # should have some features


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_statistical_domain():
    """Test with statistical domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="statistical", verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_domain():
    """Test with temporal domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="temporal", verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_spectral_domain():
    """Test with spectral domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="spectral", verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fractal_domain():
    """Test with fractal domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="fractal", verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_specific_features_list():
    """Test with specific feature names list."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(
        features=["abs_energy", "auc", "autocorr"],
        fs=100,  # auc requires fs parameter
        verbose=0,
    )
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mixed_domains_and_features():
    """Test with mixed domain strings and individual feature names."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features=["statistical", "abs_energy"], verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_feature_with_custom_parameters():
    """Test individual feature with custom parameters."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(
        features=["ecdf_percentile_count"],
        percentile=[0.6, 0.9, 1.0],
        verbose=0,
    )
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_data():
    """Test with multivariate time series."""
    X = _make_series(n_timepoints=50, n_columns=3)

    transformer = TSFELTransformer(features="statistical", verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_with_window_size():
    """Test with window_size parameter."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(
        features="statistical",
        window_size=10,
        overlap=0.5,
        verbose=0,
    )
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] > 1  # Should have multiple windows


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_required_parameter_error():
    """Test that error is raised when required parameter is missing."""
    X = _make_series(n_timepoints=50, n_columns=1)

    # auc requires fs parameter
    transformer = TSFELTransformer(features=["auc"], verbose=0)
    with pytest.raises(ValueError, match="requires parameter 'fs'"):
        transformer.fit_transform(X)


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_required_parameter_provided():
    """Test that feature works when required parameter is provided."""
    X = _make_series(n_timepoints=50, n_columns=1)

    # auc requires fs parameter - provide it
    transformer = TSFELTransformer(features=["auc"], fs=100, verbose=0)
    X_transformed = transformer.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[1] > 0
