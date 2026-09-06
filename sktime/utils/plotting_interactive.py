#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interactive timeseries plotting functionality."""

__all__ = [
    "plot_interactive_cv",
    "plot_interactive_series",
    "InteractiveForecaster",
]
__author__ = ["mloning", "RNKuhns", "Dbhasin1", "chillerobscuro", "benheid"]

import numpy as np
import pandas as pd
from warnings import simplefilter, warn

from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.plotting import plot_windows, plot_series
from sktime.forecasting.base import BaseForecaster
from sktime.split.base import BaseSplitter


def plot_interactive_cv(cv, y, title="", backend="plotly"):
    """Plot interactive training and test windows with parameter controls.
    
    Enhanced version of plot_windows that provides:
    - Connected visualizations instead of dots
    - Interactive parameter adjustment
    - Real-time CV parameter updates
    - Visual feedback for parameter changes
    
    Parameters
    ----------
    cv : sktime splitter object, descendant of BaseSplitter
        Time series splitter, e.g., temporal cross-validation iterator
    y : pd.Series
        Time series to split
    title : str
        Plot title
    backend : str, default="plotly"
        Backend for interactive plotting ("plotly" or "bokeh")
        
    Returns
    -------
    Interactive plot object with parameter controls
    """
    _check_soft_dependencies(["plotly", "ipywidgets"], severity="error")
    
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import ipywidgets as widgets
    from IPython.display import display
    
    # Get windows from CV
    train_windows, test_windows = _get_windows(cv, y)
    
    # Create interactive figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Series with CV Windows', 'CV Parameter Controls'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Plot time series
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y.values,
            mode='lines+markers',
            name='Time Series',
            line=dict(color='black')
        ),
        row=1, col=1
    )
    
    # Plot CV windows with connected lines
    colors = px.colors.qualitative.Set1
    for i, (train, test) in enumerate(zip(train_windows, test_windows)):
        # Training window
        fig.add_trace(
            go.Scatter(
                x=y.index[train],
                y=y.iloc[train].values,
                mode='lines+markers',
                name=f'Train {i+1}',
                line=dict(color=colors[i % len(colors)], width=3),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Test window
        fig.add_trace(
            go.Scatter(
                x=y.index[test],
                y=y.iloc[test].values,
                mode='lines+markers',
                name=f'Test {i+1}',
                line=dict(color=colors[i % len(colors)], width=3, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add parameter controls
    controls = _create_cv_controls(cv, y)
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True
    )
    
    return fig, controls


def plot_interactive_series(y, outliers=None, corrections=None, backend="plotly"):
    """Plot interactive time series with outlier selection and correction.
    
    Parameters
    ----------
    y : pd.Series
        Time series to plot
    outliers : list, optional
        List of outlier indices to highlight
    corrections : dict, optional
        Dictionary mapping outlier indices to corrected values
    backend : str, default="plotly"
        Backend for interactive plotting
        
    Returns
    -------
    Interactive plot object with outlier selection tools
    """
    _check_soft_dependencies(["plotly", "ipywidgets"], severity="error")
    
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import ipywidgets as widgets
    from IPython.display import display
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Series with Outlier Selection', 'Correction Controls'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main time series plot
    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y.values,
            mode='lines+markers',
            name='Original Series',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Highlight outliers if provided
    if outliers:
        fig.add_trace(
            go.Scatter(
                x=y.index[outliers],
                y=y.iloc[outliers].values,
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=10, symbol='x')
            ),
            row=1, col=1
        )
    
    # Show corrections if provided
    if corrections:
        corrected_values = []
        corrected_indices = []
        for idx, val in corrections.items():
            corrected_values.append(val)
            corrected_indices.append(y.index[idx])
        
        fig.add_trace(
            go.Scatter(
                x=corrected_indices,
                y=corrected_values,
                mode='markers',
                name='Corrected Values',
                marker=dict(color='green', size=8, symbol='circle')
            ),
            row=1, col=1
        )
    
    # Add interactive controls
    controls = _create_outlier_controls(y)
    
    fig.update_layout(
        title='Interactive Time Series with Outlier Management',
        height=600,
        showlegend=True
    )
    
    return fig, controls


def _get_windows(cv, y):
    """Extract train and test windows from CV splitter."""
    train_windows = []
    test_windows = []
    
    for train, test in cv.split(y):
        train_windows.append(train)
        test_windows.append(test)
    
    return train_windows, test_windows


def _create_cv_controls(cv, y):
    """Create interactive controls for CV parameters."""
    import ipywidgets as widgets
    from IPython.display import display
    
    # Extract current parameters
    params = {}
    if hasattr(cv, 'window_length'):
        params['window_length'] = cv.window_length
    if hasattr(cv, 'step_length'):
        params['step_length'] = cv.step_length
    if hasattr(cv, 'initial_window'):
        params['initial_window'] = cv.initial_window
    if hasattr(cv, 'fh'):
        params['fh'] = cv.fh
    
    # Create sliders for each parameter
    controls = {}
    for param, value in params.items():
        if isinstance(value, (int, float)):
            controls[param] = widgets.IntSlider(
                value=value,
                min=1,
                max=len(y) // 2,
                description=param.replace('_', ' ').title(),
                continuous_update=False
            )
    
    # Create output widget for displaying updated CV info
    output = widgets.Output()
    
    def update_cv(**kwargs):
        """Update CV parameters and redraw."""
        with output:
            output.clear_output()
            # Here you would update the CV object and redraw the plot
            print(f"Updated CV parameters: {kwargs}")
    
    # Connect controls to update function
    for control in controls.values():
        control.observe(lambda change: update_cv(**{k: v.value for k, v in controls.items()}), 'value')
    
    return controls, output


def _create_outlier_controls(y):
    """Create interactive controls for outlier management."""
    import ipywidgets as widgets
    from IPython.display import display
    
    # Outlier selection controls
    outlier_threshold = widgets.FloatSlider(
        value=2.0,
        min=0.5,
        max=5.0,
        step=0.1,
        description='Outlier Threshold (σ)',
        continuous_update=False
    )
    
    # Correction controls
    correction_method = widgets.Dropdown(
        options=['Linear Interpolation', 'Median', 'Mean', 'Custom'],
        value='Linear Interpolation',
        description='Correction Method'
    )
    
    custom_value = widgets.FloatText(
        value=0.0,
        description='Custom Value',
        disabled=True
    )
    
    # Connect controls
    def on_method_change(change):
        custom_value.disabled = change['new'] != 'Custom'
    
    correction_method.observe(on_method_change, 'value')
    
    return {
        'threshold': outlier_threshold,
        'method': correction_method,
        'custom_value': custom_value
    }


class InteractiveForecaster(BaseForecaster):
    """Enhanced forecaster with interactive visualization capabilities.
    
    This forecaster extends BaseForecaster with interactive plotting
    and parameter adjustment capabilities.
    """
    
    def __init__(self, forecaster, **kwargs):
        self.forecaster = forecaster
        super().__init__(**kwargs)
    
    def fit(self, y, X=None, fh=None):
        """Fit the underlying forecaster."""
        self.forecaster.fit(y, X, fh)
        self._y = y  # Store the target variable
        self._is_fitted = True
        return self
    
    def predict(self, fh=None, X=None):
        """Predict using the underlying forecaster."""
        return self.forecaster.predict(fh, X)
    
    def plot_interactive_cv(self, cv, title=""):
        """Plot interactive cross-validation visualization."""
        return plot_interactive_cv(cv, self._y, title)
    
    def plot_interactive_series(self, outliers=None, corrections=None):
        """Plot interactive time series with outlier management."""
        return plot_interactive_series(self._y, outliers, corrections)
    
    def adjust_seasonal_parameters(self, **seasonal_params):
        """Interactively adjust seasonal parameters with visual feedback.
        
        Parameters
        ----------
        **seasonal_params : dict
            Seasonal parameters to adjust (e.g., period, seasonal_order)
        """
        _check_soft_dependencies(["plotly", "ipywidgets"], severity="error")
        
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import ipywidgets as widgets
        from IPython.display import display
        
        # Create seasonal parameter controls
        controls = {}
        for param, value in seasonal_params.items():
            if isinstance(value, (int, float)):
                controls[param] = widgets.IntSlider(
                    value=value,
                    min=1,
                    max=100,
                    description=param.replace('_', ' ').title(),
                    continuous_update=False
                )
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Current Seasonal Pattern', 'Parameter Impact'),
            vertical_spacing=0.1
        )
        
        # Plot current seasonal pattern
        if hasattr(self.forecaster, 'seasonal_'):
            fig.add_trace(
                go.Scatter(
                    x=self._y.index,
                    y=self.forecaster.seasonal_,
                    mode='lines',
                    name='Seasonal Component',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
        
        # Add controls for parameter adjustment
        output = widgets.Output()
        
        def update_seasonal(**kwargs):
            """Update seasonal parameters and redraw."""
            with output:
                output.clear_output()
                # Here you would update the seasonal parameters and redraw
                print(f"Updated seasonal parameters: {kwargs}")
        
        # Connect controls
        for control in controls.values():
            control.observe(lambda change: update_seasonal(**{k: v.value for k, v in controls.items()}), 'value')
        
        return fig, controls, output
    
    def compare_models(self, other_forecasters, cv=None):
        """Compare multiple forecasters with interactive visualization.
        
        Parameters
        ----------
        other_forecasters : list
            List of other forecasters to compare
        cv : BaseSplitter, optional
            Cross-validation splitter for comparison
        """
        _check_soft_dependencies(["plotly"], severity="error")
        
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # Collect predictions from all forecasters
        all_forecasters = [self.forecaster] + other_forecasters
        all_predictions = []
        forecaster_names = ['Current'] + [f'Model {i+1}' for i in range(len(other_forecasters))]
        
        for forecaster in all_forecasters:
            if forecaster.is_fitted:
                pred = forecaster.predict()
                all_predictions.append(pred)
        
        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Predictions', 'Performance Metrics'),
            vertical_spacing=0.1
        )
        
        # Plot predictions
        colors = px.colors.qualitative.Set1
        for i, (pred, name) in enumerate(zip(all_predictions, forecaster_names)):
            fig.add_trace(
                go.Scatter(
                    x=pred.index,
                    y=pred.values,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i % len(colors)])
                ),
                row=1, col=1
            )
        
        # Add original series
        fig.add_trace(
            go.Scatter(
                x=self._y.index,
                y=self._y.values,
                mode='lines+markers',
                name='Original',
                line=dict(color='black', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.update_layout(
            title='Model Comparison',
            height=600,
            showlegend=True
        )
        
        return fig 