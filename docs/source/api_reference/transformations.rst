.. _transformations_ref:

Time series transformations
===========================

The :mod:`sktime.transformations` module contains classes for data
transformations.

.. automodule:: sktime.transformations
   :no-members:
   :no-inherited-members:


Composition
-----------

Pipeline building
~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TransformerPipeline
    FeatureUnion
    FitInTransform
    MultiplexTransformer

.. currentmodule:: sktime.transformations.series.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    OptionalPassthrough
    ColumnwiseTransformer

.. currentmodule:: sktime.transformations.panel.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnTransformer

Sklearn and pandas adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer

.. currentmodule:: sktime.transformations.series.adapt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TabularToSeriesAdaptor
    PandasTransformAdaptor

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

Box-Cox, log, logit
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.boxcox

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxTransformer
    LogTransformer

.. currentmodule:: sktime.transformations.series.scaledlogit

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScaledLogitTransformer

Summarization
~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SummaryTransformer
    MeanTransformer
    WindowSummarizer

Differencing
~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.difference

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Differencer

Auto-correlation features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.acf

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationTransformer
    PartialAutoCorrelationTransformer

Element-wise transforms
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.cos

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CosineTransformer

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.date

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DateTimeFeatures

Outlier detection, changepoint detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.outlier_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HampelFilter

.. currentmodule:: sktime.transformations.series.clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPTransformer


Augmentation
~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.augmenter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    InvertAugmenter
    RandomSamplesAugmenter
    ReverseAugmenter
    WhiteNoiseAugmenter

FeatureSelection
~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.feature_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FeatureSelection

Filtering and denoising
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.kalman_filter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KalmanFilterTransformerPK
    KalmanFilterTransformerFP

.. currentmodule:: sktime.transformations.series.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaLinesTransformer


Bootstrap transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.bootstrap

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    STLBootstrapTransformer
    MovingBlockBootstrapTransformer
