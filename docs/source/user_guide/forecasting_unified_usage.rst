.. _forecasting_unified_usage:

Unified Forecasting Usage
=========================

This page provides a unified interface overview for using forecasting models in sktime.

The goal is to show how all sktime forecasters can be used in a consistent, modular way.

Notebook
--------

You can also view the unified forecasting notebook example:

- :doc:`../examples/01d_forecasting_unified_example.ipynb`

Overview
--------

All forecasters in sktime follow a common interface:

- ``fit(y)``
- ``predict(fh)``
- ``update(y_new)``, optionally
- ``predict_interval``, ``predict_var`` for probabilistic forecasts

Typical Usage Example
---------------------

Here’s the basic usage pattern:

.. code-block:: python

    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.datasets import load_airline

    y = load_airline()
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    forecaster = AutoARIMA()

    forecaster.fit(y)
    y_pred = forecaster.predict(fh)

This interface is consistent across all forecasters, whether statistical, machine learning-based, or hierarchical.

Forecaster Types
----------------

Here are some types of forecasters in sktime:

- **Classical models**: e.g., ARIMA, ExponentialSmoothing
- **Machine-learning based**: e.g., RegressionForecaster
- **Probabilistic**: support `predict_interval`, `predict_proba`
- **Hierarchical/Global**: support panel data inputs

See also: :doc:`../estimator_overview` and the full API reference.
