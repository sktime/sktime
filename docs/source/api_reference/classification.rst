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
    IndividualTDE
    TemporalDictionaryEnsemble

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

Hybrid
--------------

.. currentmodule:: sktime.classification.hybrid

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    Catch22ForestClassifier
    HIVECOTEV1

Interval-based
--------------

.. currentmodule:: sktime.classification.interval_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    TimeSeriesForest
    RandomIntervalSpectralForest
    SupervisedTimeSeriesForest
    CanonicalIntervalForest
    DrCIF

Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransformClassifier
    MrSEQLClassifier
    ROCKETClassifier
