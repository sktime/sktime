.. _transformations_ref:

Time series transformations
===========================

The :mod:`sktime.transformations` module contains classes for data
transformations.

All (simple) transformers in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="transformer"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
:doc:`Estimator Search Page </estimator_overview>`
(select "transformere" in the "Estimator type" dropdown).

For pairwise transformers (time series distances, kernels), instead see :ref:`_transformations_pairwise_ref`.

Transformations are categorized as follows:

.. list-table::
   :header-rows: 1

   * - Category
     - Explanation
     - Example
   * - Composition
     - Building blocks for pipelines, wrappers, adapters
     - Transformer pipeline
   * - Series-to-features
     - Transforms series to float/category vector
     - Length and mean
   * - Series-to-series
     - Transforms individual series to series
     - Differencing, detrending
   * - Series-to-Panel
     - transforms a series into a panel
     - Bootstrap, sliding window
   * - Panel transform
     - Transforms panel to panel, not by-series
     - Padding to equal length
   * - Hierarchical
     - uses hierarchy information non-trivially
     - Reconciliation

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
    ColumnEnsembleTransformer
    ColumnwiseTransformer
    FitInTransform
    MultiplexTransformer
    OptionalPassthrough
    InvertTransform
    Id
    YtoX
    IxToX
    TransformByLevel
    TransformIf

.. currentmodule:: sktime.transformations.panel.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnTransformer

.. currentmodule:: sktime.transformations.series.func_transform

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FunctionTransformer


Sklearn and pandas adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer
    TimeBinner

.. currentmodule:: sktime.transformations.series.adapt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TabularToSeriesAdaptor
    PandasTransformAdaptor

Series-to-features transformers
-------------------------------

Series-to-features transformers transform individual time series to a collection of primitive features.
Primitive features are usually a vector of floats, but can also be categorical.

When applied to panels or hierarchical data, the transformation result is a table with as many rows as time series in the collection.

Summarization
~~~~~~~~~~~~~

These transformers extract simple summary features.

.. currentmodule:: sktime.transformations.series.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SummaryTransformer
    WindowSummarizer
    SplitterSummarizer

.. currentmodule:: sktime.transformations.panel.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DerivativeSlopeTransformer
    PlateauFinder
    RandomIntervalFeatureExtractor
    FittedParamExtractor

.. currentmodule:: sktime.transformations.series.adi_cv

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ADICVTransformer

Shapelets, wavelets, and convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.shapelet_transform

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransform
    RandomShapeletTransform
    ShapeletTransformPyts

.. currentmodule:: sktime.transformations.panel.rocket

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket
    MiniRocket
    MiniRocketMultivariate
    MiniRocketMultivariateVariable

.. currentmodule:: sktime.transformations.panel.dwt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DWTTransformer

Distance-based features
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfile

Dictionary-based features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SFA

Moment-based features
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.signature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureTransformer

Feature collections
~~~~~~~~~~~~~~~~~~~

These transformers extract larger collections of features.

.. currentmodule:: sktime.transformations.panel.tsfresh

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSFreshRelevantFeatureExtractor
    TSFreshFeatureExtractor

.. currentmodule:: sktime.transformations.panel.catch22

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22

.. currentmodule:: sktime.transformations.panel.catch22wrapper

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Wrapper

Series-to-series transformers
-----------------------------

Series-to-series transformers transform individual time series into another time series.

When applied to panels or hierarchical data, individual series are transformed.

Lagging
~~~~~~~

.. currentmodule:: sktime.transformations.series.lag

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Lag
    ReducerTransform

Element-wise transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations apply a function element-wise.

Depending on the transformer, the transformation parameters can be fitted.

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

.. currentmodule:: sktime.transformations.series.scaledasinh

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScaledAsinhTransformer

Detrending and Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.detrend

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer
    STLTransformer
    MSTL

.. currentmodule:: sktime.transformations.series.vmd

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VmdTransformer

.. currentmodule:: sktime.transformations.series.clear_sky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClearSky


Filtering and denoising
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.filter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Filter

.. currentmodule:: sktime.transformations.series.bkfilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BKFilter

.. currentmodule:: sktime.transformations.series.cffilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CFFilter

.. currentmodule:: sktime.transformations.series.hpfilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HPFilter

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

.. currentmodule:: sktime.transformations.series.bollinger

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Bollinger

Differencing, slope, kinematics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.difference

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Differencer

.. currentmodule:: sktime.transformations.panel.slope

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SlopeTransformer

.. currentmodule:: sktime.transformations.series.kinematic

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KinematicFeatures

Binning, sampling and segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.binning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeBinAggregate

.. currentmodule:: sktime.transformations.panel.interpolate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSInterpolator

.. currentmodule:: sktime.transformations.panel.segment

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter

.. currentmodule:: sktime.transformations.series.dilation_mapping

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DilationMappingTransformer

.. currentmodule:: sktime.transformations.series.paa

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PAA

.. currentmodule:: sktime.transformations.series.sax

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SAX

.. currentmodule:: sktime.transformations.panel.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PAAlegacy
    SAXlegacy

Missing value treatment
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.impute

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Imputer

.. currentmodule:: sktime.transformations.series.dropna

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DropNA

Seasonality and Date-Time Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.date

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DateTimeFeatures

.. currentmodule:: sktime.transformations.series.holiday

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HolidayFeatures
    CountryHolidaysTransformer
    FinancialHolidaysTransformer

.. currentmodule:: sktime.transformations.series.time_since

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSince

.. currentmodule:: sktime.transformations.series.fourier

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FourierFeatures

.. currentmodule:: sktime.transformations.series.fourier

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FourierTransform

.. currentmodule:: sktime.transformations.series.peak

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PeakTimeFeature

Auto-correlation series
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.series.acf

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationTransformer
    PartialAutoCorrelationTransformer

Window-based series transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers create a series based on a sequence of sliding windows.

.. currentmodule:: sktime.transformations.series.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfileTransformer

.. currentmodule:: sktime.transformations.panel.hog1d

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HOG1DTransformer

.. currentmodule:: sktime.transformations.series.subsequence_extraction

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SubsequenceExtractionTransformer


Multivariate-to-univariate
~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers convert multivariate series to univariate.

.. currentmodule:: sktime.transformations.panel.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnConcatenator

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

These transformers select features in `X` based on `y`.

.. currentmodule:: sktime.transformations.series.feature_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FeatureSelection

.. currentmodule:: sktime.transformations.panel.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ElbowClassSum
    ElbowClassPairwise

Subsetting time points and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers subset `X` by time points (`pandas` index or index level) or variables (`pandas` columns).

.. currentmodule:: sktime.transformations.series.subset

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnSelect
    IndexSubset

Adapters to other frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generic framework adapters that expose other frameworks in the ``sktime`` interface.

.. currentmodule:: sktime.transformations.series.temporian

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TemporianTransformer


Panel transformers
------------------

Panel transformers transform a panel of time series into a panel of time series.

A panel transformer is fitted on an entire panel, and not per series.

Equal length transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations ensure all series in a panel have equal length

.. currentmodule:: sktime.transformations.panel.padder

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PaddingTransformer

.. currentmodule:: sktime.transformations.panel.truncation

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TruncationTransformer

Dimension reduction
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.panel.pca

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PCATransformer

Series-to-Panel transformers
----------------------------

These transformers create a panel from a single series.

Bootstrap transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.bootstrap

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MovingBlockBootstrapTransformer
    SplitterBootstrapTransformer
    STLBootstrapTransformer

Panel-to-Series transformers
----------------------------

These transformers create a single series from a panel.

.. currentmodule:: sktime.transformations.merger

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Merger

Outlier detection, changepoint detection
----------------------------------------

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

Hierarchical transformers
-------------------------

These transformers are specifically for hierarchical data and panel data.

The transformation depends on the specified hierarchy in a non-trivial way.

.. currentmodule:: sktime.transformations.hierarchical.aggregate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Aggregator

.. currentmodule:: sktime.transformations.hierarchical.reconcile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Reconciler
