Interactive Forecasting Visualization
====================================

Overview
--------

sktime now provides interactive visualization capabilities that enhance the user experience for time series forecasting. These features address common challenges in understanding cross-validation splits, managing outliers, and tuning model parameters.

Key Features
------------

1. **Enhanced Cross-Validation Visualization**: Connected visualizations instead of dots
2. **Interactive Data Selection**: Select which historical data to include
3. **Outlier Correction**: Interactive outlier detection and correction
4. **Seasonal Parameter Adjustment**: Visual adjustment of seasonal parameters
5. **Model Comparison**: Layered visualization of multiple models

Installation
------------

The interactive features require additional dependencies:

.. code-block:: bash

    pip install plotly ipywidgets

For Jupyter notebook support:

.. code-block:: bash

    pip install jupyter

Enhanced Cross-Validation Visualization
--------------------------------------

Traditional CV visualization shows dots for each split, which can be hard to interpret. The new interactive version provides connected visualizations that make it easier to understand how parameters affect the splits.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from sktime.utils.plotting_interactive import plot_interactive_cv
    from sktime.split import ExpandingWindowSplitter
    from sktime.datasets import load_airline
    import numpy as np

    # Load data
    y = load_airline()

    # Create CV splitter
    cv = ExpandingWindowSplitter(
        fh=np.arange(1, 13),
        initial_window=24,
        step_length=12
    )

    # Create interactive visualization
    fig, controls = plot_interactive_cv(cv, y, title="Interactive CV Visualization")
    fig.show()

Interactive Data Selection and Outlier Correction
------------------------------------------------

The interactive series plotting allows you to:

- Visualize outliers with different markers
- Apply corrections with immediate visual feedback
- Adjust outlier detection thresholds
- Choose correction methods (interpolation, median, mean, custom)

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from sktime.utils.plotting_interactive import plot_interactive_series
    import numpy as np

    # Create data with outliers
    y_with_outliers = y.copy()
    outlier_indices = [50, 80, 120, 150]
    outlier_values = [800, 300, 900, 200]

    for idx, val in zip(outlier_indices, outlier_values):
        if idx < len(y_with_outliers):
            y_with_outliers.iloc[idx] = val

    # Create corrections
    corrections = {
        50: 450,   # Corrected value
        80: 380,   # Corrected value
        120: 520,  # Corrected value
        150: 480   # Corrected value
    }

    # Plot interactive series
    fig, controls = plot_interactive_series(
        y_with_outliers,
        outliers=outlier_indices,
        corrections=corrections
    )
    fig.show()

Interactive Forecaster
---------------------

The ``InteractiveForecaster`` wrapper adds interactive capabilities to any sktime forecaster.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from sktime.utils.plotting_interactive import InteractiveForecaster
    from sktime.forecasting.theta import ThetaForecaster

    # Create interactive forecaster
    base_forecaster = ThetaForecaster(sp=12)
    interactive_fc = InteractiveForecaster(base_forecaster)

    # Fit the forecaster
    interactive_fc.fit(y, fh=np.arange(1, 13))

    # Use interactive features
    fig, controls = interactive_fc.plot_interactive_cv(cv, title="CV from Forecaster")
    fig.show()

Seasonal Parameter Adjustment
----------------------------

For forecasters that support seasonal parameters, you can adjust them interactively with visual feedback.

.. code-block:: python

    # Adjust seasonal parameters
    fig, controls, output = interactive_fc.adjust_seasonal_parameters(
        sp=12,
        seasonal_order=1
    )
    
    fig.show()

Model Comparison
---------------

Compare multiple forecasters with interactive overlays.

.. code-block:: python

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import STLForecaster

    # Create multiple forecasters
    forecasters = [
        NaiveForecaster(strategy="last"),
        ThetaForecaster(sp=12),
        STLForecaster(sp=12)
    ]

    # Fit all forecasters
    for fc in forecasters:
        fc.fit(y, fh=np.arange(1, 13))

    # Compare models
    fig = interactive_fc.compare_models(forecasters)
    fig.show()

Advanced Usage
-------------

Custom Outlier Detection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def detect_outliers_zscore(y, threshold=2.0):
        """Detect outliers using Z-score method."""
        z_scores = np.abs((y - y.mean()) / y.std())
        return np.where(z_scores > threshold)[0]

    # Apply custom detection
    outlier_indices = detect_outliers_zscore(y_with_outliers, threshold=2.0)
    print(f"Detected {len(outlier_indices)} outliers")

CV Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def optimize_cv_parameters(y, cv_class, param_ranges):
        """Simple grid search for CV parameters."""
        best_score = float('inf')
        best_params = None
        
        for window_length in param_ranges['window_length']:
            for step_length in param_ranges['step_length']:
                cv = cv_class(
                    fh=np.arange(1, 13),
                    window_length=window_length,
                    step_length=step_length
                )
                
                score = cv.get_n_splits(y)
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'window_length': window_length,
                        'step_length': step_length,
                        'n_splits': score
                    }
        
        return best_params

    # Test optimization
    param_ranges = {
        'window_length': [12, 24, 36],
        'step_length': [6, 12, 18]
    }
    
    best_params = optimize_cv_parameters(y.iloc[:100], SlidingWindowSplitter, param_ranges)
    print(f"Best CV parameters: {best_params}")

API Reference
-------------

plot_interactive_cv
~~~~~~~~~~~~~~~~~~

.. autofunction:: sktime.utils.plotting_interactive.plot_interactive_cv

plot_interactive_series
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sktime.utils.plotting_interactive.plot_interactive_series

InteractiveForecaster
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sktime.utils.plotting_interactive.InteractiveForecaster
   :members:
   :inherited-members:

Benefits and Use Cases
---------------------

1. **Educational**: Interactive visualizations help users understand CV concepts
2. **Exploratory**: Quick experimentation with different parameters
3. **Debugging**: Visual identification of data quality issues
4. **Model Development**: Iterative improvement of forecasting models
5. **Communication**: Interactive plots for presentations and reports

Future Enhancements
-------------------

1. **Integration with Model Selection**: Automatic CV parameter optimization
2. **Advanced Outlier Detection**: Multiple detection algorithms
3. **Seasonal Decomposition**: Interactive seasonal pattern analysis
4. **Performance Metrics**: Real-time calculation and display
5. **Export Capabilities**: Save interactive plots and configurations

Examples
--------

See the `interactive_forecasting_demo.ipynb <../../examples/interactive_forecasting_demo.ipynb>`_ notebook for complete examples of all features. 