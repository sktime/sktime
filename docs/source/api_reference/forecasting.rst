
.. _forecasting_ref:

Forecasting
===========

The :mod:`sktime.forecasting` module contains algorithms and composition tools for forecasting.

All forecasters in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
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
Pipelines can also be constructed using ``*``, ``**``, ``+``, and ``|`` dunders.

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
    TransformSelectForecaster
    FhPlexForecaster
    IgnoreX
    FallbackForecaster

Reduction
---------

Reduction forecasters that use ``sklearn`` regressors or ``sktime``
time series regressors to make forecasts.

Concurrent tabular strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses exogeneous data at the same time stamp - simple reduction strategy.

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst


    YfromX


Direct and recursive - ``sktime`` native 1st generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1st generation direct and recursive reduction forecasters, ``numpy`` based.

Different strategies can be constructed via  ``make_reduction`` for easy specification.

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


Direct and recursive - ``sktime`` native 2nd generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2nd generation rearchitecture of direct and recursive reduction forecasters,
``pandas`` based.

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DirectReductionForecaster


Direct and recursive - 3rd party
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SkforecastAutoreg


Naive forecasters
-----------------

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveForecaster

.. currentmodule:: sktime.forecasting.dummy

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ForecastKnownValues

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

.. currentmodule:: sktime.forecasting.enbpi

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EnbPIForecaster


Trend forecasters
-----------------

.. currentmodule:: sktime.forecasting.trend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TrendForecaster
    PolynomialTrendForecaster
    STLForecaster
    CurveFitForecaster
    ProphetPiecewiseLinearTrendForecaster

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastMSTL

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

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoETS
    StatsForecastAutoCES

.. currentmodule:: sktime.forecasting.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaForecaster

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoTheta

AR/MA type forecasters
----------------------

Forecasters with AR or MA component.

All "ARIMA" and "Auto-ARIMA" models below include SARIMAX capability.

(V)AR(I)MAX models
~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.forecasting.auto_reg

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoREG

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ARIMA
    StatsModelsARIMA

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

Auto-ARIMA models
~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoARIMA

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoARIMA


ARCH models
-----------

.. currentmodule:: sktime.forecasting.arch

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastARCH
    StatsForecastGARCH
    ARCH

Structural time series models
-----------------------------

.. currentmodule:: sktime.forecasting.ardl

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ARDL

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

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoTBATS

.. currentmodule:: sktime.forecasting.fbprophet

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Prophet

.. currentmodule:: sktime.forecasting.prophetverse

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Prophetverse

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

Deep learning based forecasters
---------------------------------------------

.. currentmodule:: sktime.forecasting.ltsf

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    LTSFLinearForecaster
    LTSFDLinearForecaster
    LTSFNLinearForecaster

.. currentmodule:: sktime.forecasting.hf_transformers_forecaster

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HFTransformersForecaster

.. currentmodule:: sktime.forecasting.conditional_invertible_neural_network

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CINNForecaster

.. currentmodule:: sktime.forecasting.neuralforecast

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NeuralForecastRNN
    NeuralForecastLSTM

.. currentmodule:: sktime.forecasting.pytorchforecasting

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PytorchForecastingTFT
    PytorchForecastingNBeats

.. currentmodule:: sktime.forecasting.pykan_forecaster

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PyKANForecaster


Intermittent time series forecasters
------------------------------------

.. currentmodule:: sktime.forecasting.croston

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Croston

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
    ForecastingSkoptSearchCV
    ForecastingOptunaSearchCV

Model Evaluation (Backtesting)
------------------------------

.. currentmodule:: sktime.forecasting.model_evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    evaluate

Time index splitters
--------------------

Evaluation and tuning can be customized using time index based splitters,
for a list of these consult the :ref:`splitter API <split_ref>`
