.. _transformations_ref:

Time series transformations
===========================

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
    :toctree: auto_generated/
    :template: class.rst

    PAA
    SFA
    SAX

Summarize
~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DerivativeSlopeTransformer
    PlateauFinder
    RandomIntervalFeatureExtractor
    FittedParamExtractor

tsfresh
~~~~~~~

.. currentmodule:: sktime.transformations.panel.tsfresh

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSFreshRelevantFeatureExtractor
    TSFreshFeatureExtractor

Catch22
~~~~~~~

.. currentmodule:: sktime.transformations.panel.catch22

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22

Compose
~~~~~~~

.. currentmodule:: sktime.transformations.panel.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnTransformer
    ColumnConcatenator
    SeriesToSeriesRowTransformer
    SeriesToPrimitivesRowTransformer

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_row_transformer

Matrix profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfile

PCA
~~~

.. currentmodule:: sktime.transformations.panel.pca

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PCATransformer

Reduce
~~~~~~

.. currentmodule:: sktime.transformations.panel.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer

Rocket
~~~~~~

.. currentmodule:: sktime.transformations.panel.rocket

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket
    MiniRocket
    MiniRocketMultivariate

Segment
~~~~~~~

.. currentmodule:: sktime.transformations.panel.segment

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

Shapelet
~~~~~~~~

.. currentmodule:: sktime.transformations.panel.shapelets

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransform
    ContractedShapeletTransform

Signature
~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.signature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureTransformer

Series transformers
-------------------

Detrend
~~~~~~~

.. currentmodule:: sktime.transformations.series.detrend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer
    STLTransformer

Adapt
~~~~~

.. currentmodule:: sktime.transformations.series.adapt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TabularToSeriesAdaptor

Box-Cox
~~~~~~~

.. currentmodule:: sktime.transformations.series.boxcox

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxTransformer
    LogTransformer

ClaSP
~~~~~

.. currentmodule:: sktime.transformations.series.clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPTransformer

Difference
~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.difference

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Differencer

Auto-correlation
~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.acf

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationTransformer
    PartialAutoCorrelationTransformer

Cosine
~~~~~~

.. currentmodule:: sktime.transformations.series.cos

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CosineTransformer

Exponent
~~~~~~~~

.. currentmodule:: sktime.transformations.series.exponent

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentTransformer
    SqrtTransformer

Matrix Profile
~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfileTransformer

Missing value imputation
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.impute

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Imputer

Datetime feature generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.date

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DateTimeFeatures

Outlier detection
~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.outlier_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HampelFilter

Composition
~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    OptionalPassthrough
    ColumnwiseTransformer

Theta
~~~~~

.. currentmodule:: sktime.transformations.series.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaLinesTransformer

Summary
~~~~~~~

.. currentmodule:: sktime.transformations.series.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SummaryTransformer
    MeanTransformer

FeatureSelection
~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.feature_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FeatureSelection
