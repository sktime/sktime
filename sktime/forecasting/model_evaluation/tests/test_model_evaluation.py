import pytest
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import (
    MeanSquaredError,
    MeanAbsolutePercentageError,
)

def test_metric_name_clash_resolution():
    """Test that metrics with different parameters get unique column names."""
    y = load_airline()[:24]
    forecaster = NaiveForecaster(strategy="mean", sp=3)
    cv = ExpandingWindowSplitter(initial_window=12, step_length=6, fh=[1])

    scorers = [
        MeanSquaredError(square_root=True),  
        MeanSquaredError(),  
        MeanAbsolutePercentageError(symmetric=False), 
    ]

    results = evaluate(
        forecaster=forecaster,
        y=y,
        cv=cv,
        scoring=scorers,
    )

    expected_columns = {
        'test_MeanSquaredError_square_root=True',
        'test_MeanSquaredError',
        'test_MeanAbsolutePercentageError_symmetric=False',
        'fit_time',
        'pred_time',
        'len_train_window',
        'cutoff'
    }
    
    assert set(results.columns) == expected_columns
    
    metric_cols = [c for c in results.columns if c.startswith('test_')]
    assert len(metric_cols) == len(scorers), "Duplicate metric columns detected"

def test_parameter_order_insensitivity():
    """Test that parameter order doesn't affect metric naming."""
    scorer1 = MeanAbsolutePercentageError(symmetric=True, multioutput="uniform_average")
    scorer2 = MeanAbsolutePercentageError(multioutput="uniform_average", symmetric=True)
    
    assert scorer1.name == scorer2.name

def test_default_parameters_no_suffix():
    """Test metrics with default parameters get no parameter suffix."""
    scorer = MeanSquaredError()
    assert "_" not in scorer.name, "Default parameters should not add suffix"
    scorer_modified = MeanSquaredError(square_root=True)
    assert "_square_root=True" in scorer_modified.name

def test_custom_metric_naming():
    """Test with custom parameter combinations."""
    scorer = MeanAbsolutePercentageError(
        symmetric=True,
        multioutput="raw_values",
        horizon_weighting={"weights": [0.5, 0.5]}
    )
    
    expected_suffix = (
        "horizon_weighting={'weights': [0.5, 0.5]}_"
        "multioutput=raw_values_"
        "symmetric=True"
    )
    assert expected_suffix in scorer.name

def test_duplicate_metric_configurations():
    """Test identical metric configurations produce same column name."""
    y = load_airline()[:24]
    forecaster = NaiveForecaster(strategy="mean", sp=3)
    cv = ExpandingWindowSplitter(initial_window=12, step_length=6, fh=[1])
    
    scorers = [
        MeanSquaredError(square_root=True),
        MeanSquaredError(square_root=True),
    ]

    with pytest.raises(ValueError, match="Duplicate column names"):
        evaluate(
            forecaster=forecaster,
            y=y,
            cv=cv,
            scoring=scorers,
        )
