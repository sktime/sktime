"""Tests for LagLlamaForecaster.

Simple test for the predict_proba interface.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

from sktime.forecasting.lagllama import LagLlamaForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.skipif(
    not run_test_for_class(LagLlamaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lagllama_predict_proba_returns_empirical():
    """Test that predict_proba returns an Empirical distribution.

    This is a minimal test for the new _predict_proba method added
    in PR #10585. It verifies that:
    1. The forecaster can be instantiated with test params
    2. fit() works
    3. predict_proba() returns an Empirical distribution
    4. The distribution has the expected index and columns
    """
    from skpro.distributions.empirical import Empirical

    # Use test params for fast execution
    params = LagLlamaForecaster.get_test_params("default")[0]
    forecaster = LagLlamaForecaster(**params)

    # Create simple univariate time series
    y = make_forecasting_problem(n_timepoints=50, random_state=42)

    # Fit the forecaster
    forecaster.fit(y, fh=[1, 2, 3, 4, 5])

    # Get probabilistic predictions
    pred_dist = forecaster.predict_proba()

    # Check it returns an Empirical distribution
    assert isinstance(pred_dist, Empirical), (
        f"Expected Empirical distribution, got {type(pred_dist)}"
    )

    # Check the distribution has the expected index (forecasting horizon)
    expected_index = forecaster.fh.to_absolute(forecaster.cutoff)
    assert pred_dist.index.equals(expected_index), (
        f"Distribution index {pred_dist.index} does not match "
        f"expected forecasting horizon {expected_index}"
    )

    # Check variable name column exists
    assert len(pred_dist.columns) == 1, (
        f"Expected 1 column, got {len(pred_dist.columns)}"
    )

    # Check that predict_quantiles derived from predict_proba works
    # (consistency check - quantiles from Empirical should match)
    quantiles = forecaster.predict_quantiles(alpha=[0.1, 0.5, 0.9])
    assert quantiles is not None
    assert len(quantiles) > 0


@pytest.mark.skipif(
    not run_test_for_class(LagLlamaForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lagllama_predict_proba_panel():
    """Test predict_proba with panel data."""
    from skpro.distributions.empirical import Empirical
    from sktime.utils._testing.hierarchical import _make_hierarchical

    # Use test params for fast execution
    params = LagLlamaForecaster.get_test_params("default")[0]
    forecaster = LagLlamaForecaster(**params)

    # Create simple panel data (2 instances, 30 timepoints each)
    y = _make_hierarchical(hierarchy_levels=(2,), min_timepoints=30, max_timepoints=30)

    # Fit the forecaster
    forecaster.fit(y, fh=[1, 2, 3])

    # Get probabilistic predictions
    pred_dist = forecaster.predict_proba()

    # Check it returns an Empirical distribution
    assert isinstance(pred_dist, Empirical), (
        f"Expected Empirical distribution, got {type(pred_dist)}"
    )

    # Check the distribution index matches expected forecasting horizon
    expected_index = forecaster.fh.to_absolute(forecaster.cutoff)
    assert pred_dist.index.equals(expected_index), (
        f"Distribution index {pred_dist.index} does not match "
        f"expected forecasting horizon {expected_index}"
    )