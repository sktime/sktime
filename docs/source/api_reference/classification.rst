.. _classification_ref:

Time series classification
==========================

The :mod:`sktime.classification` module contains algorithms and composition tools for time series classification.

All classifiers in ``sktime``can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="classifier"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

Composition
-----------

.. currentmodule:: sktime.classification.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClassifierPipeline
    ColumnEnsembleClassifier
    ComposableTimeSeriesForestClassifier
    SklearnClassifierPipeline
    WeightedEnsembleClassifier

Deep learning
-------------

.. currentmodule:: sktime.classification.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CNNClassifier
    FCNClassifier
    MLPClassifier
    TapNetClassifier

Dictionary-based
----------------

.. currentmodule:: sktime.classification.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BOSSEnsemble
    ContractableBOSS
    IndividualBOSS
    IndividualTDE
    MUSE
    TemporalDictionaryEnsemble
    WEASEL

Distance-based
--------------

.. currentmodule:: sktime.classification.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ElasticEnsemble
    KNeighborsTimeSeriesClassifier
    ProximityForest
    ProximityStump
    ProximityTree
    ShapeDTW

Dummy
-----

.. currentmodule:: sktime.classification.dummy

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyClassifier

Early classification
--------------------

.. currentmodule:: sktime.classification.early_classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ProbabilityThresholdEarlyClassifier
    TEASER

Feature-based
-------------

.. currentmodule:: sktime.classification.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Classifier
    FreshPRINCE
    MatrixProfileClassifier
    RandomIntervalClassifier
    SignatureClassifier
    SummaryClassifier
    TSFreshClassifier

Hybrid
------

.. currentmodule:: sktime.classification.hybrid

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HIVECOTEV1
    HIVECOTEV2

Interval-based
--------------

.. currentmodule:: sktime.classification.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CanonicalIntervalForest
    DrCIF
    RandomIntervalSpectralEnsemble
    SupervisedTimeSeriesForest
    TimeSeriesForestClassifier

Kernel-based
------------

.. currentmodule:: sktime.classification.kernel_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesSVC
    Arsenal
    RocketClassifier

Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransformClassifier

sklearn
-------

.. currentmodule:: sktime.classification.sklearn

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ContinuousIntervalTree
    RotationForest

Base
----

.. currentmodule:: sktime.classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClassifier

.. currentmodule:: sktime.classification.deep_learning.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepClassifier

.. currentmodule:: sktime.classification.early_classification.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseEarlyClassifier
