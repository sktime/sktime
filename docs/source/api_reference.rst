.. _api_reference:

=============
API Reference
=============

This is the class and function reference for ``sktime``.

.. autosummary::
    :toctree: modules/auto_generated/

.. include:: includes/api_css.rst

.. _classification_ref:

:mod:`sktime.classification`: Time series classification
========================================================

The ``sktime.classification`` module contains algorithms and composition
tools for time series classification.

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

    BOSSIndividual
    BOSSEnsemble

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

Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransformClassifier
    MrSEQLClassifier

.. _regression_ref:

:mod:`sktime.regression`: Time series regression
================================================

The ``sktime.regression`` module contains algorithms and composition
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

:mod:`sktime.series_as_features`: Series-as-features tools
==========================================================

The ``sktime.series_as_features`` module contains algorithms and composition
tools that are shared by the classification and regression modules.

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

:mod:`sktime.transformers`: Time series transformers
========================================================

The ``sktime.transformers`` module contains classes for data transformations.

.. automodule:: sktime.transformers
    :no-members:
    :no-inherited-members:

Series-as-features transformers
-------------------------------

Dictionary-based
~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.dictionary_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PAA
    SFA
    SAX

Summarize
~~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.summarize

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

.. currentmodule:: sktime.transformers.series_as_features.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ColumnTransformer
    RowTransformer
    ColumnConcatenator

Matrix profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.matrix_profile

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    MatrixProfile

PCA
~~~

.. currentmodule:: sktime.transformers.series_as_features.pca

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PCATransformer

Reduce
~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.reduce

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Tabularizer

Rocket
~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.rocket

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Rocket

Segment
~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.segment

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

Shapelet
~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.shapelets

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransform
    ContractedShapeletTransform

Single-series transformers
--------------------------

Detrend
~~~~~~~

.. currentmodule:: sktime.transformers.single_series.detrend

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer

Adapt
~~~~~

.. currentmodule:: sktime.transformers.single_series.adapt

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    SingleSeriesTransformAdaptor

Box-cox
~~~~~~~

.. currentmodule:: sktime.transformers.single_series.boxcox

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    BoxCoxTransformer

.. _utils_ref:

:mod:`sktime.utils`: Utility function
=====================================

The ``sktime.utils`` module contains utility functions.

.. autosummary::
    :template: function.rst

    sktime.utils.plotting
    sktime.utils.validation
    sktime.utils.data_container

.. _exceptions_ref:

:mod:`sktime.exceptions`: Exceptions
====================================

The ``sktime.exceptions`` module contains classes for exceptions and warnings.

.. autosummary::
    :template: class.rst

    sktime.exceptions
