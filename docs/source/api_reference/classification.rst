.. _classification_ref:

Time series classification
==========================

The :mod:`sktime.classification` module contains algorithms and composition tools for time series classification.

Composition
-----------

.. currentmodule:: sktime.classification.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ColumnEnsembleClassifier

Dictionary-based
----------------

.. currentmodule:: sktime.classification.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IndividualBOSS
    BOSSEnsemble
    ContractableBOSS
    WEASEL
    MUSE
    IndividualTDE
    TemporalDictionaryEnsemble

Distance-based
--------------

.. currentmodule:: sktime.classification.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KNeighborsTimeSeriesClassifier
    ElasticEnsemble
    ProximityForest
    ProximityTree
    ProximityStump

Dummy
-----

.. currentmodule:: sktime.classification.dummy

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyClassifier

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

    TimeSeriesForestClassifier
    SupervisedTimeSeriesForest
    CanonicalIntervalForest
    DrCIF
    RandomIntervalSpectralEnsemble


Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransformClassifier

Kernel-based
------------

.. currentmodule:: sktime.classification.kernel_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RocketClassifier
    Arsenal

Feature-based
-------------

.. currentmodule:: sktime.classification.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Classifier
    MatrixProfileClassifier
    TSFreshClassifier
    SignatureClassifier
    FreshPRINCE
    SummaryClassifier
    RandomIntervalClassifier
