.. _api_reference:

=============
API Reference
=============

This is the class and function reference for ``sktime``.

.. autosummary::
    :toctree: modules/auto_generated/

.. include:: includes/api_css.rst

.. _classification_ref:

sktime.classification: Time series classification
=================================================

The :mod:`sktime.classification` module contains algorithms and composition tools for time series classification.

.. automodule:: sktime.classification
    :no-members:
    :no-inherited-members:

Composition
-----------

.. currentmodule:: sktime.classification.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    TimeSeriesForestClassifier
    ColumnEnsembleClassifier

Dictionary-based
----------------

.. currentmodule:: sktime.classification.dictionary_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    IndividualBOSS
    BOSSEnsemble
    ContractableBOSS
    WEASEL
    MUSE
    TemporalDictionaryEnsemble
    IndividualTDE

Distance-based
--------------

.. currentmodule:: sktime.classification.distance_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    KNeighborsTimeSeriesClassifier
    ElasticEnsemble
    ProximityForest
    ProximityTree
    ProximityStump

Frequency-based
---------------

.. currentmodule:: sktime.classification.frequency_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    RandomIntervalSpectralForest

Interval-based
--------------

.. currentmodule:: sktime.classification.interval_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    TimeSeriesForest
    CanonicalIntervalForest

Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransformClassifier
    MrSEQLClassifier

.. _regression_ref:

sktime.regression: Time series regression
=========================================

The :mod:`sktime.regression` module contains algorithms and composition
tools for time series regression.

.. automodule:: sktime.regression
    :no-members:
    :no-inherited-members:

Composition
-----------

.. currentmodule:: sktime.regression.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    TimeSeriesForestRegressor


.. _series_as_features_ref:

sktime.series_as_features: Series-as-features tools
===================================================

The :mod:`sktime.series_as_features` module contains algorithms and composition tools that are shared by the classification and regression modules.

.. automodule:: sktime.series_as_features
    :no-members:
    :no-inherited-members:

Composition
-----------

.. currentmodule:: sktime.series_as_features.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    FeatureUnion

Model selection
---------------

.. currentmodule:: sktime.series_as_features.model_selection

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PresplitFilesCV
    SingleSplit

.. _forecasting_ref:

sktime.forecasting: Time series forecasting
===========================================

The :mod:`sktime.forecasting` module contains algorithms and composition tools for forecasting.

.. automodule:: sktime.forecasting
    :no-members:
    :no-inherited-members:

Base
----

.. currentmodule:: sktime.forecasting.base

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ForecastingHorizon

Naive
-----

.. currentmodule:: sktime.forecasting.naive

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    NaiveForecaster

Trend
-----

.. currentmodule:: sktime.forecasting.trend

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PolynomialTrendForecaster

Exponential Smoothing
---------------------

.. currentmodule:: sktime.forecasting.exp_smoothing

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ExponentialSmoothing

.. currentmodule:: sktime.forecasting.ets

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    AutoETS

ARIMA
-----

.. currentmodule:: sktime.forecasting.arima

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    AutoARIMA

Theta
-----

.. currentmodule:: sktime.forecasting.theta

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ThetaForecaster

Composition
-----------

.. currentmodule:: sktime.forecasting.compose

.. autosummary::
    :toctree: modules/auto_generated/
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

Online Forecasting
------------------

.. currentmodule:: sktime.forecasting.online_forecasting

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    OnlineEnsembleForecaster
    NormalHedgeEnsemble
    NNLSEnsemble

Model selection
---------------

.. currentmodule:: sktime.forecasting.model_selection

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    CutoffSplitter
    SingleWindowSplitter
    SlidingWindowSplitter
    ForecastingGridSearchCV

.. autosummary::
    :toctree: modules/auto_generated/
    :template: function.rst

    temporal_train_test_split

.. _transformers_ref:

sktime.transformers: Time series transformers
=============================================

The :mod:`sktime.transformers` module contains classes for data transformations.

.. automodule:: sktime.transformers
    :no-members:
    :no-inherited-members:

Series-as-features transformers
-------------------------------

Dictionary-based
~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformers.panel.dictionary_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PAA
    SFA
    SAX

Summarize
~~~~~~~~~

.. currentmodule:: sktime.transformers.panel.summarize

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    DerivativeSlopeTransformer
    PlateauFinder
    RandomIntervalFeatureExtractor
    FittedParamExtractor
    TSFreshRelevantFeatureExtractor
    TSFreshFeatureExtractor


Compose
~~~~~~~

.. currentmodule:: sktime.transformers.panel.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ColumnTransformer
    RowTransformer
    ColumnConcatenator

Matrix profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformers.panel.matrix_profile

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    MatrixProfile

PCA
~~~

.. currentmodule:: sktime.transformers.panel.pca

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PCATransformer

Reduce
~~~~~~

.. currentmodule:: sktime.transformers.panel.reduce

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Tabularizer

Rocket
~~~~~~

.. currentmodule:: sktime.transformers.panel.rocket

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Rocket

Segment
~~~~~~~

.. currentmodule:: sktime.transformers.panel.segment

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

Shapelet
~~~~~~~~

.. currentmodule:: sktime.transformers.panel.shapelets

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransform
    ContractedShapeletTransform

Series transformers
-------------------

Detrend
~~~~~~~

.. currentmodule:: sktime.transformers.series.detrend

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer

Adapt
~~~~~

.. currentmodule:: sktime.transformers.series.adapt

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    SingleSeriesTransformAdaptor

Box-cox
~~~~~~~

.. currentmodule:: sktime.transformers.series.boxcox

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    BoxCoxTransformer

.. _datasets_ref:

sktime.datasets: Datasets
=========================

.. currentmodule:: sktime.datasets.base

.. autosummary::
    :toctree: modules/auto_generated/
    :template: function.rst

    load_airline
    load_arrow_head
    load_gunpoint
    load_osuleaf
    load_italy_power_demand
    load_basic_motions
    load_japanese_vowels
    load_shampoo_sales
    load_longley
    load_lynx
    load_acsf1
    load_uschange
    load_UCR_UEA_dataset

.. _utils_ref:

sktime.utils: Utility function
==============================

The :mod:`sktime.utils` module contains utility functions.

.. autosummary::
    :template: function.rst

    sktime.utils.plotting
    sktime.utils.validation
    sktime.utils.data_container

.. _exceptions_ref:

sktime.exceptions: Exceptions
=============================

The :mod:`sktime.exceptions` module contains classes for exceptions and warnings.

.. autosummary::
    :template: class.rst

    sktime.exceptions
