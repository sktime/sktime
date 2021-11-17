
.. _forecasting_ref:

Forecasting
===========

The :mod:`sktime.forecasting` module contains algorithms and composition tools for forecasting.

.. automodule:: sktime.forecasting
    :no-members:
    :no-inherited-members:


Base
----

.. currentmodule:: sktime.forecasting.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ForecastingHorizon

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

    TrendForecaster
    PolynomialTrendForecaster

Exponential Smoothing
---------------------

.. currentmodule:: sktime.forecasting.exp_smoothing

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentialSmoothing

.. currentmodule:: sktime.forecasting.ets

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoETS

ARIMA
-----

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoARIMA
    ARIMA

Theta
-----

.. currentmodule:: sktime.forecasting.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaForecaster

BATS/TBATS
----------

.. currentmodule:: sktime.forecasting.bats

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BATS

.. currentmodule:: sktime.forecasting.tbats

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TBATS

Croston
-------

.. currentmodule:: sktime.forecasting.croston

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Croston

Prophet
-------

.. currentmodule:: sktime.forecasting.fbprophet

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Prophet

Unobserved Components
--------------------

.. currentmodule:: sktime.forecasting.structural

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UnobservedComponents

Composition
-----------

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnEnsembleForecaster
    EnsembleForecaster
    AutoEnsembleForecaster
    StackingForecaster
    TransformedTargetForecaster
    ForecastingPipeline
    DirectTabularRegressionForecaster
    DirectTimeSeriesRegressionForecaster
    MultioutputTabularRegressionForecaster
    MultioutputTimeSeriesRegressionForecaster
    RecursiveTabularRegressionForecaster
    RecursiveTimeSeriesRegressionForecaster
    DirRecTabularRegressionForecaster
    DirRecTimeSeriesRegressionForecaster
    MultiplexForecaster

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_reduction

Online Forecasting
------------------

.. currentmodule:: sktime.forecasting.online_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    OnlineEnsembleForecaster
    NormalHedgeEnsemble
    NNLSEnsemble

Model Selection
---------------

.. currentmodule:: sktime.forecasting.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CutoffSplitter
    SingleWindowSplitter
    SlidingWindowSplitter
    ExpandingWindowSplitter
    ForecastingGridSearchCV
    ForecastingRandomizedSearchCV

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    temporal_train_test_split

Model Evaluation (Backtesting)
------------------------------

.. currentmodule:: sktime.forecasting.model_evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    evaluate
