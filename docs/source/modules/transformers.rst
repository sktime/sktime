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
    :toctree: auto_generated/
    :template: class.rst

    PAA
    SFA
    SAX

Summarize
~~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.summarize

.. autosummary::
    :toctree: auto_generated/
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
    :toctree: auto_generated/
    :template: class.rst

    ColumnTransformer
    RowTransformer
    ColumnConcatenator

Matrix profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfile

PCA
~~~

.. currentmodule:: sktime.transformers.series_as_features.pca

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PCATransformer

Reduce
~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer

Rocket
~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.rocket

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket

Segment
~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.segment

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

Shapelet
~~~~~~~~

.. currentmodule:: sktime.transformers.series_as_features.shapelets

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransform
    ContractedShapeletTransform

Single-series transformers
--------------------------

Detrend
~~~~~~~

.. currentmodule:: sktime.transformers.single_series.detrend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer

Adapt
~~~~~

.. currentmodule:: sktime.transformers.single_series.adapt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SingleSeriesTransformAdaptor

Box-cox
~~~~~~~

.. currentmodule:: sktime.transformers.single_series.boxcox

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxTransformer
