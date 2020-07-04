.. _classification_ref:

:mod:`sktime.classification`: Time series classification
========================================================

The ``sktime.classification`` module contains algorithms and composition
tools for time series classification.

.. automodule:: sktime.classification
    :no-members:
    :no-inherited-members:

Composition
-----------

.. currentmodule:: sktime.classification.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesForestClassifier
    ColumnEnsembleClassifier

Dictionary-based
----------------

.. currentmodule:: sktime.classification.dictionary_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BOSSIndividual
    BOSSEnsemble

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

Frequency-based
---------------

.. currentmodule:: sktime.classification.frequency_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomIntervalSpectralForest

Interval-based
--------------

.. currentmodule:: sktime.classification.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesForest

Shapelet-based
--------------

.. currentmodule:: sktime.classification.shapelet_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ShapeletTransformClassifier
    MrSEQLClassifier
