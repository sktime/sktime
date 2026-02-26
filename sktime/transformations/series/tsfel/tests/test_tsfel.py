#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for TSFEL transformer functionality."""

__author__ = ["Faakhir30"]

import numpy as np
import pandas as pd
import pytest

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
    all_features = X_transformed["all"].iloc[0]
    assert isinstance(all_features, pd.DataFrame)
    assert all_features.shape[0] > 0
    assert all_features.shape[1] > 0  # should have some features


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_statistical_domain():
    """Test with statistical domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="statistical", verbose=0)
    X_transformed = transformer.fit_transform(X)

    statistical_features = X_transformed["statistical"].iloc[0]
    assert isinstance(statistical_features, pd.DataFrame)
    assert statistical_features.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_domain():
    """Test with temporal domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="temporal", verbose=0)
    X_transformed = transformer.fit_transform(X)
    temporal_features = X_transformed["temporal"].iloc[0]
    assert isinstance(temporal_features, pd.DataFrame)
    assert temporal_features.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_spectral_domain():
    """Test with spectral domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="spectral", verbose=0)
    X_transformed = transformer.fit_transform(X)

    spectral_features = X_transformed["spectral"].iloc[0]
    assert isinstance(spectral_features, pd.DataFrame)
    assert spectral_features.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fractal_domain():
    """Test with fractal domain features."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features="fractal", verbose=0)
    X_transformed = transformer.fit_transform(X)

    fractal_features = X_transformed["fractal"].iloc[0]
    assert isinstance(fractal_features, pd.DataFrame)
    assert fractal_features.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_specific_features_list():
    """Test with specific feature names list."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(
        features=["abs_energy", "auc", "autocorr"],
        fs=100,
        verbose=0,
    )
    X_transformed = transformer.fit_transform(X)

    abs_energy = X_transformed["abs_energy"].iloc[0]
    assert isinstance(abs_energy, np.float64)


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mixed_domains_and_features():
    """Test with mixed domain strings and individual feature names."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(features=["statistical", "abs_energy"], verbose=0)
    X_transformed = transformer.fit_transform(X)

    statistical_features = X_transformed["statistical"].iloc[0]
    assert isinstance(statistical_features, pd.DataFrame)
    assert statistical_features.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_feature_with_custom_parameters():
    """Test individual feature with custom parameters."""
    X = _make_series(n_timepoints=50, n_columns=1)

    transformer = TSFELTransformer(
        features=["ecdf_percentile_count"],
        verbose=0,
        kwargs={"percentile": [0.6, 0.9, 1.0]},
    )
    X_transformed = transformer.fit_transform(X)

    ecdf_percentile_count = X_transformed["ecdf_percentile_count"].iloc[0]
    assert isinstance(ecdf_percentile_count, tuple)
    assert len(ecdf_percentile_count) == 3


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_data():
    """Test with multivariate time series."""
    X = _make_series(n_timepoints=50, n_columns=3)

    transformer = TSFELTransformer(features="statistical", verbose=0)
    X_transformed = transformer.fit_transform(X)

    statistical_features = X_transformed["statistical"].iloc[0]
    assert isinstance(statistical_features, pd.DataFrame)
    assert statistical_features.shape[1] > 0


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

    auc = X_transformed["auc"].iloc[0]
    assert isinstance(auc, np.float64)


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_nonexistent_feature_error():
    """Test that error is raised when non-existent feature name is provided."""
    with pytest.raises(
        ValueError, match="not found in tsfel.feature_extraction.features"
    ):
        TSFELTransformer(features=["nonexistent_feature_xyz"], verbose=0)
