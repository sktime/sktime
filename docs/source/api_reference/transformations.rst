.. _transformations_ref:

Time series transformations
===========================

The :mod:`sktime.transformations` module contains classes for data
transformations.

All (simple) transformers in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="transformer"``, optionally filtered by tags.

Valid tags are listed in :ref:`the transformations tags API reference <transformer_tags>`,
and can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
:doc:`Estimator Search Page </estimator_overview>`
(select "transformer" in the "Estimator type" dropdown).

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

Pipeline building - Structural
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TransformerPipeline
    FeatureUnion
    ColumnEnsembleTransformer
    FitInTransform
    InvertTransform
    YtoX
    IxToX

.. currentmodule:: sktime.transformations.subset

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnSelect
    IndexSubset

Pipeline building - Broadcasting and apply-map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnwiseTransformer
    TransformByLevel

.. currentmodule:: sktime.transformations.func_transform

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FunctionTransformer

Pipeline building - AutoML, switches and multiplexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MultiplexTransformer
    OptionalPassthrough
    TransformIf
    Id

Pipeline building - Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Logger

Pipeline building - Output combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CombineTransformers

Sklearn, pandas, numpy adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.reduce

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Tabularizer
    TimeBinner

.. currentmodule:: sktime.transformations.adapt

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

.. currentmodule:: sktime.transformations.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SummaryTransformer
    WindowSummarizer
    SplitterSummarizer

.. currentmodule:: sktime.transformations.summarize

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DerivativeSlopeTransformer
    PlateauFinder
    RandomIntervalFeatureExtractor
    FittedParamExtractor

.. currentmodule:: sktime.transformations.adi_cv

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ADICVTransformer

Shapelets, wavelets, and convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.shapelet_transform

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransform
    RandomShapeletTransform
    ShapeletTransformPyts

.. currentmodule:: sktime.transformations.rocket

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Rocket
    RocketPyts
    MiniRocket
    MiniRocketMultivariate
    MiniRocketMultivariateCython
    MiniRocketMultivariateVariable
    MultiRocket
    MultiRocketMultivariate

.. currentmodule:: sktime.transformations.dwt

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DWTTransformer

Distance-based features
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfileFeatures

.. currentmodule:: sktime.transformations.compose_distance

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DistanceFeatures

Dictionary-based features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SFA
    SFAFast
    PAAlegacy
    SAXlegacy

.. currentmodule:: sktime.transformations.fabba

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FABBA

Auto-correlation-based features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.hurst

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HurstExponentTransformer

Moment-based features
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.signature

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureMoments

.. currentmodule:: sktime.transformations.signature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureTransformer

Feature collections
~~~~~~~~~~~~~~~~~~~

These transformers extract larger collections of features.

.. currentmodule:: sktime.transformations.tsfresh

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSFreshRelevantFeatureExtractor
    TSFreshFeatureExtractor

.. currentmodule:: sktime.transformations.tsfeatures

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSFeaturesTransformer

.. currentmodule:: sktime.transformations.catch22

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22

.. currentmodule:: sktime.transformations.catch22wrapper

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Wrapper

.. currentmodule:: sktime.transformations.evoforest_tswm

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EvoForestTSWM

.. currentmodule:: sktime.transformations.tsfel

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSFELTransformer

Series-to-series transformers
-----------------------------

Series-to-series transformers transform individual time series into another time series.

When applied to panels or hierarchical data, individual series are transformed.

Lagging
~~~~~~~

.. currentmodule:: sktime.transformations.lag

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Lag
    ReducerTransform

Element-wise transforms
~~~~~~~~~~~~~~~~~~~~~~~

These transformations apply a function element-wise.

Depending on the transformer, the transformation parameters can be fitted.

.. currentmodule:: sktime.transformations.boxcox

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BoxCoxTransformer
    LogTransformer

.. currentmodule:: sktime.transformations.scaledlogit

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScaledLogitTransformer

.. currentmodule:: sktime.transformations.cos

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CosineTransformer

.. currentmodule:: sktime.transformations.exponent

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentTransformer
    SqrtTransformer

.. currentmodule:: sktime.transformations.scaledasinh

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScaledAsinhTransformer

.. currentmodule:: sktime.transformations.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CombineTransformers

Detrending and Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.detrend

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    Detrender
    Deseasonalizer
    ConditionalDeseasonalizer
    STLTransformer
    mstl.MSTL

.. currentmodule:: sktime.transformations.vmd

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    VmdTransformer

.. currentmodule:: sktime.transformations.clear_sky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClearSky


Filtering and denoising
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.filter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Filter

.. currentmodule:: sktime.transformations.bkfilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BKFilter

.. currentmodule:: sktime.transformations.cffilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CFFilter

.. currentmodule:: sktime.transformations.hpfilter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HPFilter

.. currentmodule:: sktime.transformations.kalman_filter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KalmanFilterTransformerPK
    KalmanFilterTransformerFP
    KalmanFilterTransformerSIMD

.. currentmodule:: sktime.transformations.theta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThetaLinesTransformer

.. currentmodule:: sktime.transformations.bollinger

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Bollinger

Differencing, slope, kinematics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.difference

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Differencer

.. currentmodule:: sktime.transformations.slope

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SlopeTransformer

.. currentmodule:: sktime.transformations.kinematic

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KinematicFeatures

Binning, sampling and segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.binning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeBinAggregate

.. currentmodule:: sktime.transformations.interpolate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TSInterpolator

.. currentmodule:: sktime.transformations.segment

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IntervalSegmenter
    RandomIntervalSegmenter
    SlidingWindowSegmenter

.. currentmodule:: sktime.transformations.random_intervals

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomIntervals

.. currentmodule:: sktime.transformations.supervised_intervals

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SupervisedIntervals

.. currentmodule:: sktime.transformations.dilation_mapping

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DilationMappingTransformer

.. currentmodule:: sktime.transformations.paa

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PAA

.. currentmodule:: sktime.transformations.sax

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SAX

.. currentmodule:: sktime.transformations.fabba

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    fABBA

Missing value treatment
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.impute

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Imputer

.. currentmodule:: sktime.transformations.dropna

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DropNA

Seasonality and Date-Time Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.date

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DateTimeFeatures

.. currentmodule:: sktime.transformations.holiday

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    HolidayFeatures
    country_holidays.CountryHolidaysTransformer
    financial_holidays.FinancialHolidaysTransformer

.. currentmodule:: sktime.transformations.time_since

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSince

.. currentmodule:: sktime.transformations.fourier

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FourierFeatures

.. currentmodule:: sktime.transformations.fourier

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FourierTransform

.. currentmodule:: sktime.transformations.dummies

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SeasonalDummiesOneHot

.. currentmodule:: sktime.transformations.basisfunction

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RBFTransformer

.. currentmodule:: sktime.transformations.peak

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PeakTimeFeature

Auto-correlation series
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.acf

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoCorrelationTransformer
    PartialAutoCorrelationTransformer

Window-based series transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers create a series based on a sequence of sliding windows.

.. currentmodule:: sktime.transformations.matrix_profile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MatrixProfileTransformer

.. currentmodule:: sktime.transformations.hog1d

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HOG1DTransformer

.. currentmodule:: sktime.transformations.subsequence_extraction

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SubsequenceExtractionTransformer


Multivariate-to-univariate
~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers convert multivariate series to univariate.

.. currentmodule:: sktime.transformations.colconcat

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnConcatenator

Augmentation
~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.augmenter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    InvertAugmenter
    RandomSamplesAugmenter
    ReverseAugmenter
    WhiteNoiseAugmenter

Trajectory
~~~~~~~~~~

Transformers for spatial trajectory data, wrapping ``movingpandas``.

.. currentmodule:: sktime.transformations.series.trajectory

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DouglasPeuckerTrajectoryGeneralizer

FeatureSelection
~~~~~~~~~~~~~~~~

These transformers select features in `X` based on `y`.

.. currentmodule:: sktime.transformations.feature_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FeatureSelection

.. currentmodule:: sktime.transformations.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ElbowClassSum
    ElbowClassPairwise

Subsetting time points and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transformers subset `X` by time points (`pandas` index or index level) or variables (`pandas` columns).

.. currentmodule:: sktime.transformations.subset

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnSelect
    IndexSubset

Adapters to other frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generic framework adapters that expose other frameworks in the ``sktime`` interface.

.. currentmodule:: sktime.transformations.temporian

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

.. currentmodule:: sktime.transformations.padder

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PaddingTransformer

.. currentmodule:: sktime.transformations.truncation

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TruncationTransformer

Dimension reduction
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.pca

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
    TSBootstrapAdapter
    RepeatBootstrapTransformer

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

.. currentmodule:: sktime.transformations.outlier_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HampelFilter

.. currentmodule:: sktime.transformations.clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPTransformer

.. currentmodule:: sktime.transformations.dobin

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DOBIN

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

For usage of Reconciliation with pipelines, these transformations below are more
efficient and should be preferred:

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    BottomUpReconciler
    MiddleOutReconciler
    NonNegativeOptimalReconciler
    OptimalReconciler
    TopdownReconciler


Domain Specific Transformations
-------------------------------

These transformers are designed for specific domains and inputs.
They compute features that are related to a domain of application.

Energy, weather and climate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.transformations.clear_sky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClearSky

.. currentmodule:: sktime.transformations.degree_day

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DegreeDayFeatures
