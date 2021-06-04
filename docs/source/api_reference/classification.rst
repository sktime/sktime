.. _classification_ref:

sktime.classification: Time series classification
=================================================

The :mod:`sktime.classification` module contains algorithms and composition tools for time series classification.

.. automodule:: sktime.classification
    :no-members:

Composition
-----------

.. currentmodule:: sktime.classification.compose

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

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

    HIVECOTEV1

Interval-based
--------------

.. currentmodule:: sktime.classification.interval_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    TimeSeriesForestClassifier
    RandomIntervalSpectralForest
    SupervisedTimeSeriesForest

Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ShapeletTransformClassifier
    MrSEQLClassifier

Kernel-based
--------------

.. currentmodule:: sktime.classification.kernel_based

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    ROCKETClassifier
    Arsenal
