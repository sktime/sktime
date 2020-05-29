.. _forecasting_ref:

:mod:`sktime.forecasting`: Time series forecasting
==================================================

The ``sktime.forecasting`` module contains algorithms and composition
tools for forecasting.

.. automodule:: sktime.forecasting
    :no-members:
    :no-inherited-members:

Naive
-----

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveForecaster

Trend
-----

.. currentmodule:: sktime.forecasting.trend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PolynomialTrendForecaster

Exponential Smoothing
---------------------

.. currentmodule:: sktime.forecasting.exp_smoothing

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentialSmoothing

ARIMA
-----

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoARIMA

Theta
-----

.. currentmodule:: sktime.forecasting.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaForecaster

Composition
-----------

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EnsembleForecaster
    TransformedTargetForecaster
    DirectRegressionForecaster
    DirectTimeSeriesRegressionForecaster
    RecursiveRegressionForecaster
    RecursiveTimeSeriesRegressionForecaster
    ReducedRegressionForecaster
    ReducedTimeSeriesRegressionForecaster
    StackingForecaster

Model selection
---------------

.. currentmodule:: sktime.forecasting.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CutoffSplitter
    SingleWindowSplitter
    SlidingWindowSplitter
    ForecastingGridSearchCV

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    temporal_train_test_split
