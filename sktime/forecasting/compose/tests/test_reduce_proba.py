# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for MCRecursiveProbaReductionForecaster."""

__author__ = ["marrov"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import MCRecursiveProbaReductionForecaster
from sktime.split import temporal_train_test_split
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.dependencies import _check_soft_dependencies

SKPRO_INSTALLED = _check_soft_dependencies("skpro", severity="none")


def _make_probabilistic_regressor():
    """Create a simple probabilistic regressor for testing."""
    from sklearn.linear_model import LinearRegression
    from skpro.regression.residual import ResidualDouble

    return ResidualDouble(LinearRegression())


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_basic_fit_predict():
    """Test basic fit and predict returns correct output type and shape."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=12,
        n_samples=20,
        random_state=42,
    )

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(y_test)
    assert y_pred.index.equals(y_test.index)
    assert not y_pred.isna().any()


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_predict_proba_returns_distribution():
    """Test predict_proba returns a valid distribution object."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=6)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=6,
        n_samples=50,
        random_state=42,
    )

    forecaster.fit(y_train)
    pred_dist = forecaster.predict_proba(fh)

    # Check distribution has expected interface
    assert hasattr(pred_dist, "sample")
    assert hasattr(pred_dist, "mean")
    assert hasattr(pred_dist, "quantile")

    # Check that sampling works
    samples = pred_dist.sample(n_samples=5)
    assert len(samples) == 5 * len(fh)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_predict_equals_distribution_mean():
    """Test that predict() returns the mean of the distribution."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=6)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=6,
        n_samples=100,
        random_state=42,
    )

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    pred_dist = forecaster.predict_proba(fh)
    dist_mean = pred_dist.mean()

    # Convert to comparable format
    if isinstance(dist_mean, pd.DataFrame) and dist_mean.shape[1] == 1:
        dist_mean = dist_mean.iloc[:, 0]

    pd.testing.assert_series_equal(y_pred, dist_mean, check_names=False)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_with_exogenous():
    """Test with exogenous variables."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=3)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=2,
        n_samples=20,
        random_state=42,
    )

    forecaster.fit(y_train, X=X_train)
    y_pred = forecaster.predict(fh, X=X_test)

    assert len(y_pred) == len(y_test)
    assert not y_pred.isna().any()


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_reproducibility_with_random_state():
    """Test that same random_state produces identical results."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=6)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    params = dict(
        estimator=_make_probabilistic_regressor(),
        window_length=6,
        n_samples=20,
        random_state=42,
    )

    f1 = MCRecursiveProbaReductionForecaster(**params)
    f1.fit(y_train)
    pred1 = f1.predict(fh)

    f2 = MCRecursiveProbaReductionForecaster(**params)
    f2.fit(y_train)
    pred2 = f2.predict(fh)

    pd.testing.assert_series_equal(pred1, pred2)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_different_random_state_produces_different_results():
    """Test that different random_state produces different results."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=6)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    f1 = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=6,
        n_samples=20,
        random_state=42,
    )
    f1.fit(y_train)
    pred1 = f1.predict(fh)

    f2 = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=6,
        n_samples=20,
        random_state=43,
    )
    f2.fit(y_train)
    pred2 = f2.predict(fh)

    assert not pred1.equals(pred2)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_hierarchical_global_pooling():
    """Test with hierarchical data and global pooling."""
    y = _make_hierarchical(
        hierarchy_levels=(2, 2),
        max_timepoints=30,
        min_timepoints=30,
        n_columns=1,
        random_state=42,
    )

    y_train, y_test = temporal_train_test_split(y, test_size=5)
    fh = ForecastingHorizon(np.arange(1, 6), is_relative=True)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=5,
        n_samples=10,
        pooling="global",
        random_state=42,
    )

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == len(y_test)
    assert y_pred.index.nlevels == 3  # h0, h1, time

    # Check predict_proba works
    pred_dist = forecaster.predict_proba(fh)
    assert hasattr(pred_dist, "sample")
    assert hasattr(pred_dist, "mean")

    # Check distribution samples have correct shape
    samples = pred_dist.sample(n_samples=3)
    assert len(samples) == 3 * len(y_test)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_trajectories_stored():
    """Test that trajectories are stored after prediction."""
    y = load_airline()[:50]
    y_train, y_test = temporal_train_test_split(y, test_size=5)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=3,
        n_samples=10,
        random_state=42,
    )

    forecaster.fit(y_train)
    forecaster.predict(fh)

    assert forecaster.trajectories_ is not None
    assert None in forecaster.trajectories_
    traj = forecaster.trajectories_[None]
    assert traj.shape == (10, 5)  # (n_samples, n_horizons)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_non_contiguous_horizon():
    """Test with non-contiguous forecasting horizon (e.g., fh=[1, 3, 5])."""
    y = load_airline()[:50]
    y_train, _ = temporal_train_test_split(y, test_size=10)
    fh = ForecastingHorizon([1, 3, 5], is_relative=True)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=3,
        n_samples=10,
        random_state=42,
    )

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    # Should only return predictions for requested horizons
    assert len(y_pred) == 3
    assert not y_pred.isna().any()


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_caching_works():
    """Test that caching avoids recomputation for same fh/X."""
    y = load_airline()[:50]
    y_train, y_test = temporal_train_test_split(y, test_size=5)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=3,
        n_samples=10,
        random_state=42,
    )

    forecaster.fit(y_train)

    # First call - should compute
    pred1 = forecaster.predict(fh)
    assert forecaster._cached_pred_dist_ is not None

    # Second call with same fh - should use cache
    pred2 = forecaster.predict(fh)
    pd.testing.assert_series_equal(pred1, pred2)


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_window_length_1():
    """Test with window_length=1 (single lag)."""
    y = load_airline()[:50]
    y_train, y_test = temporal_train_test_split(y, test_size=5)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=_make_probabilistic_regressor(),
        window_length=1,
        n_samples=10,
        random_state=42,
    )

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    assert len(y_pred) == len(y_test)
    assert not y_pred.isna().any()


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_rejects_non_probabilistic_regressor():
    """Test that ValueError is raised for non-probabilistic regressors."""
    from sklearn.linear_model import LinearRegression

    with pytest.raises(ValueError, match="requires an skpro probabilistic regressor"):
        MCRecursiveProbaReductionForecaster(
            estimator=LinearRegression(),
            window_length=3,
        )


@pytest.mark.skipif(not SKPRO_INSTALLED, reason="skpro required")
def test_invalid_pooling_raises():
    """Test that invalid pooling value raises ValueError."""
    with pytest.raises(ValueError, match="pooling"):
        MCRecursiveProbaReductionForecaster(
            estimator=_make_probabilistic_regressor(),
            window_length=3,
            pooling="invalid",
        )
