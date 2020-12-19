.. _transformations_ref:

sktime.transformations: Time series transformers
=============================================

The :mod:`sktime.transformations` module contains classes for data
transformations.

.. automodule:: sktime.transformations
    :no-members:
    :no-inherited-members:

Panel transformers
------------------

Dictionary-based
~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.dictionary_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PAA
    SFA
    SAX

Summarize
~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.summarize

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    DerivativeSlopeTransformer
    PlateauFinder
    RandomIntervalFeatureExtractor
    FittedParamExtractor
    TSFreshRelevantFeatureExtractor
    TSFreshFeatureExtractor

tsfresh
~~~~~~~

.. currentmodule:: sktime.transformations.panel.tsfresh

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

Compose
~~~~~~~

.. currentmodule:: sktime.transformations.panel.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ColumnTransformer
    ColumnConcatenator
    SeriesToSeriesRowTransformer
    SeriesToPrimitivesRowTransformer

.. autosummary::
    :toctree: modules/auto_generated/
    :template: function.rst

    make_row_transformer

Matrix profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.matrix_profile

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    MatrixProfile

PCA
~~~

.. currentmodule:: sktime.transformations.panel.pca

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    PCATransformer

Reduce
~~~~~~

.. currentmodule:: sktime.transformations.panel.reduce

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Tabularizer

Rocket
~~~~~~

.. currentmodule:: sktime.transformations.panel.rocket

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Rocket

Segment
~~~~~~~

.. currentmodule:: sktime.transformations.panel.segment

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

Shapelet
~~~~~~~~

.. currentmodule:: sktime.transformations.panel.shapelets

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransform
    ContractedShapeletTransform

Series transformers
-------------------

Detrend
~~~~~~~

.. currentmodule:: sktime.transformations.series.detrend

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer

Adapt
~~~~~

.. currentmodule:: sktime.transformations.series.adapt

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    TabularToSeriesAdaptor

Box-cox
~~~~~~~

.. currentmodule:: sktime.transformations.series.boxcox

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    BoxCoxTransformer
    LogTransformer

Matrix Profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.matrix_profile

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    MatrixProfileTransformer
