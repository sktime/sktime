"""Tests for AutoTS custom functionality."""

import pandas as pd
import pytest

from sktime.forecasting.autots import AutoTS
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_prediction_intervals():
    """Test that AutoTS can predict intervals."""
    from sktime.datasets import load_airline

    y = load_airline()

    # Configure with specific interval and FAST settings for TESTING ONLY
    # This ensures our local test doesn't take forever, without modifying class defaults
    coverage = 0.9
    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        prediction_interval=coverage,
        random_seed=42,  # Ensure reproducibility
    )

    forecaster.fit(y, fh=[1, 2, 3])

    # Test successful prediction with single coverage
    intervals = forecaster.predict_interval(coverage=coverage)

    assert isinstance(intervals, pd.DataFrame)
    assert intervals.shape == (3, 2)
    assert intervals.columns.nlevels == 3

    # Check values
    lower = intervals.iloc[:, 0]
    upper = intervals.iloc[:, 1]
    assert (upper >= lower).all()

    # Test with multiple coverages (our new feature)
    coverages = [0.9, 0.5]
    intervals_multi = forecaster.predict_interval(coverage=coverages)
    assert intervals_multi.shape == (3, 4)  # 2 coverages * 2 bounds

    # Verify column structure
    expected_cols = pd.MultiIndex.from_product(
        [["Number of airline passengers"], [0.5, 0.9], ["lower", "upper"]],
        names=["variable", "coverage", "lower/upper"],
    )
    # Sort to ensure order matches
    intervals_multi = intervals_multi.sort_index(axis=1)
    pd.testing.assert_index_equal(intervals_multi.columns, expected_cols)


@pytest.mark.skipif(
    not _check_estimator_deps(AutoTS, severity="none"),
    reason="autots not available",
)
def test_autots_tags():
    """Test that AutoTS has correct tags."""
    forecaster = AutoTS()
    assert forecaster.get_tag("capability:pred_int") is True
