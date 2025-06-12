import pytest
import pandas as pd
from sktime.utils.dependencies import _check_soft_dependencies

class TestEvaluate:
    @pytest.mark.skipif(
        not _check_soft_dependencies("pandas", severity="none"),
        reason="pandas not available",
    )
    def test_evaluate_score_across_windows(self):
        """Test that score_across_windows calculates metrics correctly."""
        pytest.importorskip("pandas")
        from sktime.datasets import load_airline
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.model_selection import ExpandingWindowSplitter
        from sktime.performance_metrics.forecasting import (
            MeanAbsolutePercentageError,
            MeanSquaredError,
        )
        from sktime.forecasting.model_evaluation import evaluate

        # Test setup
        y = load_airline()[:24]
        forecaster = NaiveForecaster(strategy="mean", sp=3)
        cv = ExpandingWindowSplitter(initial_window=12, step_length=6, fh=[1, 2, 3])

        # Test case 1: Single scorer
        scorer = MeanAbsolutePercentageError()
        results = evaluate(
            forecaster=forecaster,
            y=y,
            cv=cv,
            scoring=scorer,
            return_data=True,
            score_across_windows=True,
        )
        assert isinstance(results, pd.DataFrame)
        assert "all" in results.index
        assert not pd.isna(results.loc["all", "test_MeanAbsolutePercentageError"])

        # Test case 2: Multiple scorers
        scorers = [MeanAbsolutePercentageError(), MeanSquaredError(square_root=True)]
        results = evaluate(
            forecaster=forecaster,
            y=y,
            cv=cv,
            scoring=scorers,
            return_data=True,
            score_across_windows=True,
        )
        assert isinstance(results, pd.DataFrame)
        assert "all" in results.index
        assert "test_MeanAbsolutePercentageError" in results.columns
        assert "test_MeanSquaredError" in results.columns

# Add a simple test to verify test discovery
def test_simple():
    """Verify test discovery is working."""
    assert True