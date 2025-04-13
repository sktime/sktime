
.. _forecasting_ref:

Forecasting
===========

The :mod:`sktime.forecasting` module contains algorithms and composition tools for forecasting.

All forecasters in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="forecaster"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
:doc:`Estimator Search Page </estimator_overview>`
(select "forecaster" in the "Estimator type" dropdown).


Base
----

.. currentmodule:: sktime.forecasting.base

.. autosummary::
    :signatures: none
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
    :signatures: none
    :toctree: auto_generated/
    :template: function.rst

    make_pipeline

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    TransformedTargetForecaster
    ForecastingPipeline
    ColumnEnsembleForecaster
    MultiplexForecaster
    ForecastX
    ForecastByLevel
    TransformSelectForecaster
    GroupbyCategoryForecaster
    HierarchyEnsembleForecaster
    Permute
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
    :signatures: none
    :toctree: auto_generated/
    :template: function.rst


    YfromX


Direct and recursive - ``sktime`` native 1st generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1st generation direct and recursive reduction forecasters, ``numpy`` based.

Different strategies can be constructed via  ``make_reduction`` for easy specification.

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: function.rst

    make_reduction

.. autosummary::
    :signatures: none
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
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    DirectReductionForecaster
    RecursiveReductionForecaster

Direct and recursive - 3rd party
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    SkforecastAutoreg
    SkforecastRecursive

.. currentmodule:: sktime.forecasting.darts

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    DartsRegressionModel
    DartsLinearRegressionModel
    DartsXGBModel


Naive forecasters
-----------------

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    NaiveForecaster

.. currentmodule:: sktime.forecasting.dummy

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ForecastKnownValues

Prediction intervals
--------------------

Wrappers that add prediction intervals to any forecaster.

.. currentmodule:: sktime.forecasting.squaring_residuals

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    SquaringResiduals

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    NaiveVariance

.. currentmodule:: sktime.forecasting.conformal

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ConformalIntervals

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BaggingForecaster

.. currentmodule:: sktime.forecasting.enbpi

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    EnbPIForecaster


Calibration and bias adjustment
-------------------------------

.. currentmodule:: sktime.forecasting.boxcox_bias_adjusted_forecaster

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxBiasAdjustedForecaster


Trend forecasters
-----------------

.. currentmodule:: sktime.forecasting.trend

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    TrendForecaster
    PolynomialTrendForecaster
    STLForecaster
    CurveFitForecaster
    ProphetPiecewiseLinearTrendForecaster
    SplineTrendForecaster

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastMSTL

Exponential smoothing based forecasters
---------------------------------------

.. currentmodule:: sktime.forecasting.exp_smoothing

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ExponentialSmoothing

.. currentmodule:: sktime.forecasting.ets

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    AutoETS

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoETS
    StatsForecastAutoCES

.. currentmodule:: sktime.forecasting.theta

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ThetaForecaster

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :signatures: none
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
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    AutoREG

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ARIMA
    StatsModelsARIMA

.. currentmodule:: sktime.forecasting.sarimax

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    SARIMAX

.. currentmodule:: sktime.forecasting.var

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    VAR

.. currentmodule:: sktime.forecasting.var_reduce

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    VARReduce

.. currentmodule:: sktime.forecasting.varmax

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    VARMAX

.. currentmodule:: sktime.forecasting.vecm

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    VECM

Auto-ARIMA models
~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    AutoARIMA

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoARIMA


ARCH models
-----------

.. currentmodule:: sktime.forecasting.arch

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastARCH
    StatsForecastGARCH
    ARCH

Structural time series models
-----------------------------

.. currentmodule:: sktime.forecasting.ardl

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ARDL

.. currentmodule:: sktime.forecasting.bats

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BATS

.. currentmodule:: sktime.forecasting.tbats

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    TBATS

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastAutoTBATS

.. currentmodule:: sktime.forecasting.fbprophet

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    Prophet

.. currentmodule:: sktime.forecasting.prophetverse

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    Prophetverse
    HierarchicalProphet

.. currentmodule:: sktime.forecasting.structural

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    UnobservedComponents

.. currentmodule:: sktime.forecasting.dynamic_factor

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    DynamicFactor

Deep learning based forecasters
-------------------------------

.. currentmodule:: sktime.forecasting.ltsf

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    LTSFLinearForecaster
    LTSFDLinearForecaster
    LTSFNLinearForecaster
    LTSFTransformerForecaster

.. currentmodule:: sktime.forecasting.scinet

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    SCINetForecaster

.. currentmodule:: sktime.forecasting.conditional_invertible_neural_network

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    CINNForecaster

.. currentmodule:: sktime.forecasting.neuralforecast

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    NeuralForecastRNN
    NeuralForecastLSTM

.. currentmodule:: sktime.forecasting.pytorchforecasting

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    PytorchForecastingTFT
    PytorchForecastingDeepAR
    PytorchForecastingNHiTS
    PytorchForecastingNBeats

.. currentmodule:: sktime.forecasting.pykan_forecaster

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    PyKANForecaster

.. currentmodule:: sktime.forecasting.rbf_forecaster

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    RBFForecaster

.. currentmodule:: sktime.forecasting.es_rnn

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ESRNNForecaster

Pre-trained and foundation models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.forecasting.hf_transformers_forecaster

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    HFTransformersForecaster

.. currentmodule:: sktime.forecasting.chronos

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ChronosForecaster

.. currentmodule:: sktime.forecasting.moirai_forecaster

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    MOIRAIForecaster

.. currentmodule:: sktime.forecasting.timesfm_forecaster

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    TimesFMForecaster

.. currentmodule:: sktime.forecasting.timemoe
    :toctree: auto_generated/
    :template: class.rst

    TimeMoEForecaster

.. currentmodule:: sktime.forecasting.ttm

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    TinyTimeMixerForecaster

.. currentmodule:: sktime.forecasting.time_llm

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    TimeLLMForecaster

Intermittent time series forecasters
------------------------------------

.. currentmodule:: sktime.forecasting.croston

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    Croston

.. currentmodule:: sktime.forecasting.statsforecast

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    StatsForecastADIDA

Ensembles and stacking
----------------------

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    EnsembleForecaster
    AutoEnsembleForecaster
    StackingForecaster


.. currentmodule:: sktime.forecasting.autots

    AutoTS

Hierarchical reconciliation
---------------------------

.. currentmodule:: sktime.forecasting.reconcile

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ReconcilerForecaster

Online and stream forecasting
-----------------------------

.. currentmodule:: sktime.forecasting.online_learning

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    OnlineEnsembleForecaster
    NormalHedgeEnsemble
    NNLSEnsemble

.. currentmodule:: sktime.forecasting.stream

.. autosummary::
    :signatures: none
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
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    HCrystalBallAdapter

Model selection and tuning
--------------------------

.. currentmodule:: sktime.forecasting.model_selection

.. autosummary::
    :signatures: none
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
    :signatures: none
    :toctree: auto_generated/
    :template: function.rst

    evaluate

Time index splitters
--------------------

Evaluation and tuning can be customized using time index based splitters,
for a list of these consult the :ref:`splitter API <split_ref>`
