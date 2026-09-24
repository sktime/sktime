# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for interactive plotting functionality."""

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.plotting_interactive import (
    InteractiveForecaster,
    plot_interactive_cv,
    plot_interactive_series,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_plot_interactive_cv():
    """Test interactive CV plotting."""
    y = load_airline().iloc[:50]
    cv = ExpandingWindowSplitter(fh=np.arange(1, 13), initial_window=24, step_length=12)
    
    fig, controls = plot_interactive_cv(cv, y, title="Test CV")
    
    # Check that figure is created
    assert fig is not None
    assert hasattr(fig, 'show')
    
    # Check that controls are created
    assert controls is not None
    assert len(controls) == 2  # controls and output


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_plot_interactive_series():
    """Test interactive series plotting."""
    y = load_airline().iloc[:50]
    outliers = [10, 20, 30]
    corrections = {10: 400, 20: 450, 30: 500}
    
    fig, controls = plot_interactive_series(y, outliers, corrections)
    
    # Check that figure is created
    assert fig is not None
    assert hasattr(fig, 'show')
    
    # Check that controls are created
    assert controls is not None
    assert 'threshold' in controls
    assert 'method' in controls
    assert 'custom_value' in controls


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_interactive_forecaster():
    """Test InteractiveForecaster wrapper."""
    y = load_airline().iloc[:50]
    base_forecaster = ThetaForecaster(sp=12)
    interactive_fc = InteractiveForecaster(base_forecaster)
    
    # Test fitting
    interactive_fc.fit(y, fh=np.arange(1, 13))
    assert interactive_fc.is_fitted
    
    # Test prediction
    pred = interactive_fc.predict()
    assert len(pred) == 12
    
    # Test interactive CV plotting
    cv = ExpandingWindowSplitter(fh=np.arange(1, 13), initial_window=24, step_length=12)
    fig, controls = interactive_fc.plot_interactive_cv(cv, title="Test")
    assert fig is not None
    
    # Test interactive series plotting
    fig, controls = interactive_fc.plot_interactive_series()
    assert fig is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_interactive_forecaster_model_comparison():
    """Test model comparison functionality."""
    y = load_airline().iloc[:50]
    
    # Create multiple forecasters
    forecasters = [
        NaiveForecaster(strategy="last"),
        ThetaForecaster(sp=12)
    ]
    
    # Fit all forecasters
    for fc in forecasters:
        fc.fit(y, fh=np.arange(1, 13))
    
    # Create interactive forecaster
    interactive_fc = InteractiveForecaster(forecasters[0])
    interactive_fc.fit(y, fh=np.arange(1, 13))
    
    # Test model comparison
    fig = interactive_fc.compare_models(forecasters[1:])
    assert fig is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_seasonal_parameter_adjustment():
    """Test seasonal parameter adjustment."""
    y = load_airline().iloc[:50]
    base_forecaster = ThetaForecaster(sp=12)
    interactive_fc = InteractiveForecaster(base_forecaster)
    interactive_fc.fit(y, fh=np.arange(1, 13))
    
    # Test seasonal parameter adjustment
    fig, controls, output = interactive_fc.adjust_seasonal_parameters(sp=12)
    assert fig is not None
    assert controls is not None
    assert output is not None


def test_missing_dependencies():
    """Test that appropriate errors are raised when dependencies are missing."""
    y = load_airline().iloc[:50]
    cv = ExpandingWindowSplitter(fh=np.arange(1, 13), initial_window=24, step_length=12)
    
    # This should raise an error if plotly/ipywidgets are not available
    try:
        plot_interactive_cv(cv, y)
    except ImportError:
        # Expected behavior when dependencies are missing
        pass
    except Exception as e:
        # Other exceptions should not occur
        pytest.fail(f"Unexpected exception: {e}")


def test_data_validation():
    """Test data validation in interactive functions."""
    # Test with invalid data types
    with pytest.raises(TypeError):
        plot_interactive_cv("not_a_splitter", "not_a_series")
    
    with pytest.raises(TypeError):
        plot_interactive_series("not_a_series")


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_cv_parameter_extraction():
    """Test that CV parameters are correctly extracted for controls."""
    y = load_airline().iloc[:50]
    
    # Test with different CV types
    cv_expanding = ExpandingWindowSplitter(
        fh=np.arange(1, 13),
        initial_window=24,
        step_length=12
    )
    
    fig, controls = plot_interactive_cv(cv_expanding, y)
    assert controls is not None
    
    # Check that expected parameters are extracted
    control_dict, output = controls
    expected_params = ['initial_window', 'step_length']
    
    for param in expected_params:
        if hasattr(cv_expanding, param):
            assert param in control_dict or param.replace('_', ' ') in [c.description for c in control_dict.values()]


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_outlier_controls():
    """Test that outlier controls are properly created."""
    y = load_airline().iloc[:50]
    outliers = [10, 20]
    corrections = {10: 400, 20: 450}
    
    fig, controls = plot_interactive_series(y, outliers, corrections)
    
    # Check that all expected controls are present
    expected_controls = ['threshold', 'method', 'custom_value']
    for control_name in expected_controls:
        assert control_name in controls
    
    # Check that method control has expected options
    method_control = controls['method']
    expected_options = ['Linear Interpolation', 'Median', 'Mean', 'Custom']
    assert all(option in method_control.options for option in expected_options)


@pytest.mark.skipif(
    not _check_soft_dependencies(["plotly", "ipywidgets"], severity="none"),
    reason="plotly and ipywidgets are required for interactive plotting",
)
def test_interactive_forecaster_inheritance():
    """Test that InteractiveForecaster properly inherits from BaseForecaster."""
    y = load_airline().iloc[:50]
    base_forecaster = ThetaForecaster(sp=12)
    interactive_fc = InteractiveForecaster(base_forecaster)
    
    # Test that it has BaseForecaster methods
    assert hasattr(interactive_fc, 'fit')
    assert hasattr(interactive_fc, 'predict')
    assert hasattr(interactive_fc, 'is_fitted')
    
    # Test that it can be used like a regular forecaster
    interactive_fc.fit(y, fh=np.arange(1, 13))
    pred = interactive_fc.predict()
    assert len(pred) == 12
    assert interactive_fc.is_fitted


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__]) 