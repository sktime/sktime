# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test MCRecursiveProbaReductionForecaster."""

__author__ = ["marrov"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import MCRecursiveProbaReductionForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.dependencies import _check_soft_dependencies


def _make_probabilistic_regressor():
    """Create a simple probabilistic regressor for testing."""
    from sklearn.linear_model import LinearRegression
    from skpro.regression.residual import ResidualDouble

    # Use ResidualDouble to wrap a standard sklearn regressor
    # This creates a probabilistic regressor that estimates mean with the estimator
    # and variance from residuals
    return ResidualDouble(LinearRegression())


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro is required for MCRecursiveProbaReductionForecaster",
)
def test_mc_reduction_basic():
    """Test basic fit and predict for MCRecursiveProbaReductionForecaster."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = _make_probabilistic_regressor()
    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=regressor, window_length=12, n_samples=20, random_state=42
    )

    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(y_test)
    assert y_pred.index.equals(y_test.index)


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro is required for MCRecursiveProbaReductionForecaster",
)
def test_mc_reduction_predict_proba():
    """Test predict_proba returns proper distribution."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = _make_probabilistic_regressor()
    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=regressor, window_length=12, n_samples=50, random_state=42
    )

    forecaster.fit(y_train)
    pred_dist = forecaster.predict_proba(fh)

    assert hasattr(pred_dist, "sample")
    assert hasattr(pred_dist, "mean")
    assert hasattr(pred_dist, "quantile")

    # Check dimensions of sampled distribution
    samples = pred_dist.sample(n_samples=10)
    assert len(samples) == 10 * len(fh)

    # Check mean matches roughly the predict() output (which is mean of samples)
    y_pred = forecaster.predict(fh)
    dist_mean = pred_dist.mean()

    # Should be close but not exact due to MC sampling vs direct mean calculation
    # Note: MCRecursiveProbaReductionForecaster.predict returns mean of trajectories
    # predict() output is converted to Series by BaseForecaster if input was Series
    # but dist.mean() returns DataFrame (because distribution was created with columns)
    if (
        isinstance(dist_mean, pd.DataFrame)
        and dist_mean.shape[1] == 1
        and isinstance(y_pred, pd.Series)
    ):
        dist_mean = dist_mean.iloc[:, 0]

    pd.testing.assert_series_equal(y_pred, dist_mean, check_names=False)


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro is required for MCRecursiveProbaReductionForecaster",
)
def test_mc_reduction_with_exogenous():
    """Test MCRecursiveProbaReductionForecaster with exogenous variables."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=3)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = _make_probabilistic_regressor()
    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=regressor, window_length=2, n_samples=20, random_state=42
    )

    forecaster.fit(y_train, X=X_train)
    y_pred = forecaster.predict(fh, X=X_test)

    assert len(y_pred) == len(y_test)
    assert not y_pred.isna().any()


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro is required for MCRecursiveProbaReductionForecaster",
)
def test_mc_reduction_reproducibility():
    """Test that random_state ensures reproducibility."""
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=6)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    regressor = _make_probabilistic_regressor()

    # Run 1
    f1 = MCRecursiveProbaReductionForecaster(
        estimator=regressor, window_length=12, n_samples=20, random_state=42
    )
    f1.fit(y_train)
    pred1 = f1.predict(fh)

    # Run 2
    f2 = MCRecursiveProbaReductionForecaster(
        estimator=regressor, window_length=12, n_samples=20, random_state=42
    )
    f2.fit(y_train)
    pred2 = f2.predict(fh)

    # Run 3 (different seed)
    f3 = MCRecursiveProbaReductionForecaster(
        estimator=regressor, window_length=12, n_samples=20, random_state=43
    )
    f3.fit(y_train)
    pred3 = f3.predict(fh)

    pd.testing.assert_series_equal(pred1, pred2)
    assert not pred1.equals(pred3)


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"),
    reason="skpro is required for MCRecursiveProbaReductionForecaster",
)
def test_mc_reduction_hierarchical():
    """Test MCRecursiveProbaReductionForecaster with hierarchical data."""
    # Create simple hierarchical data
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range("2020-01-01", periods=20, freq="D")],
        names=["id", "time"],
    )
    y = pd.DataFrame(np.random.normal(size=40), index=index, columns=["y"])

    # Split manually
    train = y.groupby("id").head(15)
    test = y.groupby("id").tail(5)

    # Create FH (absolute) for test period
    # Need to be careful with MultiIndex for FH
    # For global pooling, we can pass relative FH
    fh = ForecastingHorizon(np.arange(1, 6), is_relative=True)

    regressor = _make_probabilistic_regressor()
    forecaster = MCRecursiveProbaReductionForecaster(
        estimator=regressor,
        window_length=5,
        n_samples=10,
        pooling="global",
        random_state=42,
    )

    forecaster.fit(train)
    y_pred = forecaster.predict(fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == len(test)
    # Check index structure matches
    assert y_pred.index.nlevels == 2

    # Check predict_proba
    pred_dist = forecaster.predict_proba(fh)
    assert hasattr(pred_dist, "sample")

    # Sample and check structure
    # samples = pred_dist.sample(5) <- Unfinished?
