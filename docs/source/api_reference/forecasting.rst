
.. _forecasting_ref:

Forecasting
===========

The :mod:`sktime.forecasting` module contains algorithms and composition tools for forecasting.

All clusterers in ``sktime``can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="forecaster"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

Base
----

.. currentmodule:: sktime.forecasting.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseForecaster
    ForecastingHorizon

Pipeline composition
--------------------

Compositors for building forecasting pipelines.
Pipelines can also be constructed using ``*``, ``+``, and ``|`` dunders.

.. currentmodule:: sktime.pipeline

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_pipeline

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TransformedTargetForecaster
    ForecastingPipeline
    ColumnEnsembleForecaster
    MultiplexForecaster
    ForecastX
    ForecastByLevel
    Permute
    HierarchyEnsembleForecaster

Reduction
---------

Reduction forecasters that use ``sklearn`` regressors or ``sktime`` time series regressors to make forecasts.
Use ``make_reduction`` for easy specification.

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_reduction

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DirectTabularRegressionForecaster
    DirectTimeSeriesRegressionForecaster
    MultioutputTabularRegressionForecaster
    MultioutputTimeSeriesRegressionForecaster
    RecursiveTabularRegressionForecaster
    RecursiveTimeSeriesRegressionForecaster
    DirRecTabularRegressionForecaster
    DirRecTimeSeriesRegressionForecaster

Naive forecaster
----------------

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveForecaster

Prediction intervals
--------------------

Wrappers that add prediction intervals to any forecaster.

.. currentmodule:: sktime.forecasting.squaring_residuals

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SquaringResiduals

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveVariance

.. currentmodule:: sktime.forecasting.conformal

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ConformalIntervals

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaggingForecaster


Trend forecasters
-----------------

.. currentmodule:: sktime.forecasting.trend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TrendForecaster
    PolynomialTrendForecaster
    STLForecaster

Exponential smoothing based forecasters
---------------------------------------

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

.. currentmodule:: sktime.forecasting.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaForecaster

.. currentmodule:: sktime.forecasting.croston

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Croston

AR/MA type forecasters
----------------------

Forecasters with AR or MA component.
All "ARIMA" models below include SARIMAX capability.

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoARIMA
    ARIMA

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoARIMA

.. currentmodule:: sktime.forecasting.sarimax

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SARIMAX

.. currentmodule:: sktime.forecasting.var

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VAR

.. currentmodule:: sktime.forecasting.varmax

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VARMAX

Structural time series models
-----------------------------

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

.. currentmodule:: sktime.forecasting.fbprophet

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Prophet

.. currentmodule:: sktime.forecasting.structural

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UnobservedComponents

.. currentmodule:: sktime.forecasting.dynamic_factor

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DynamicFactor

Ensembles and stacking
----------------------

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EnsembleForecaster
    AutoEnsembleForecaster
    StackingForecaster

Hierarchical reconciliation
---------------------------

.. currentmodule:: sktime.forecasting.reconcile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ReconcilerForecaster

Online and stream forecasting
-----------------------------

.. currentmodule:: sktime.forecasting.online_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    OnlineEnsembleForecaster
    NormalHedgeEnsemble
    NNLSEnsemble

.. currentmodule:: sktime.forecasting.stream

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UpdateEvery
    UpdateRefitsEvery
    DontUpdate

Adapters to other forecasting framework packages
------------------------------------------------

Generic framework adapters that expose other frameworks in the ``sktime`` interface.

.. currentmodule:: sktime.forecasting.adapters

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HCrystalBallAdapter

Model selection and tuning
--------------------------

.. currentmodule:: sktime.forecasting.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ForecastingGridSearchCV
    ForecastingRandomizedSearchCV

Model Evaluation (Backtesting)
------------------------------

.. currentmodule:: sktime.forecasting.model_evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    evaluate

Time series splitters
---------------------

Time series splitters can be used in both evaluation and tuning.

.. currentmodule:: sktime.forecasting.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CutoffSplitter
    SingleWindowSplitter
    SlidingWindowSplitter
    ExpandingWindowSplitter

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    temporal_train_test_split
