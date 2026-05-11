"""Tests for Chronos2Forecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.chronos2 import Chronos2Forecaster
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(Chronos2Forecaster, severity="none"),
    reason="autots not available",
)
def test_chronos2_fit_truncates_context_on_time_axis():
    """`context_length` truncation should apply to time axis, not feature axis."""
    pytest.importorskip("torch")

    y = pd.DataFrame(np.arange(300).reshape(100, 3), columns=["a", "b", "c"])
    forecaster = Chronos2Forecaster(config={"context_length": 10}, ignore_deps=True)
    forecaster._load_pipeline = lambda: object()

    forecaster._fit(y)

    assert forecaster._context.shape == (3, 10)

@pytest.mark.skipif(
    not _check_estimator_deps(Chronos2Forecaster, severity="none"),
    reason="chronos-forecasting not available",
)
def test_chronos2_predict_quantiles():
    """Test predict_quantiles method."""
    pytest.importorskip("torch")
    
    y = pd.DataFrame(np.arange(30).reshape(10, 3), columns=["a", "b", "c"])
    forecaster = Chronos2Forecaster(config={"context_length": 5}, ignore_deps=True)
    forecaster.fit(y)
    
    # Test with default quantiles
    quantiles = forecaster.predict_quantiles(fh=[1, 2, 3])
    
    assert isinstance(quantiles, pd.DataFrame)
    assert quantiles.shape[0] == 3  # 3 forecast horizons
    assert quantiles.columns.nlevels == 2  # (variable, quantile)
    assert set(quantiles.columns.get_level_values(0)) == {"a", "b", "c"}
    
    # Test with specific alpha values
    alpha = [0.1, 0.5, 0.9]
    quantiles_alpha = forecaster.predict_quantiles(fh=[1, 2], alpha=alpha)
    
    assert quantiles_alpha.shape[0] == 2
    assert set(quantiles_alpha.columns.get_level_values(1)) == set(alpha)


@pytest.mark.skipif(
    not _check_estimator_deps(Chronos2Forecaster, severity="none"),
    reason="chronos-forecasting not available",
)
def test_chronos2_predict_interval():
    """Test predict_interval method."""
    pytest.importorskip("torch")
    
    y = pd.DataFrame(np.arange(30).reshape(10, 3), columns=["a", "b", "c"])
    forecaster = Chronos2Forecaster(config={"context_length": 5}, ignore_deps=True)
    forecaster.fit(y)
    
    # Test with default coverage
    intervals = forecaster.predict_interval(fh=[1, 2, 3])
    
    assert isinstance(intervals, pd.DataFrame)
    assert intervals.shape[0] == 3
    assert intervals.columns.nlevels == 3  # (variable, coverage, bound)
    assert set(intervals.columns.get_level_values(2)) == {"lower", "upper"}
    
    # Use coverage values whose symmetric quantiles exist in the model grid
    # e.g. 0.8 -> (0.1, 0.9), 0.9 -> (0.05, 0.95) — both available
    coverage = [0.8, 0.9]
    intervals_cov = forecaster.predict_interval(fh=[1, 2], coverage=coverage)
    
    assert intervals_cov.shape[0] == 2
    assert set(intervals_cov.columns.get_level_values(1)) == set(coverage)


@pytest.mark.skipif(
    not _check_estimator_deps(Chronos2Forecaster, severity="none"),
    reason="chronos-forecasting not available",
)
def test_chronos2_predict_proba():
    """Test predict_proba method."""
    pytest.importorskip("torch")
    pytest.importorskip("skpro")
    
    y = pd.DataFrame(np.arange(30).reshape(10, 3), columns=["a", "b", "c"])
    forecaster = Chronos2Forecaster(config={"context_length": 5}, ignore_deps=True)
    forecaster.fit(y)
    
    # Test predict_proba
    pred_dist = forecaster.predict_proba(fh=[1, 2, 3])
    
    from skpro.distributions.base import BaseDistribution
    assert isinstance(pred_dist, BaseDistribution)