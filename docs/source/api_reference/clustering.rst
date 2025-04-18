.. _clustering_ref:

Time series clustering
======================

The :mod:`sktime.clustering` module contains algorithms for time series clustering.

All clusterers in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="clusterer"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
:doc:`Estimator Search Page </estimator_overview>`
(select "clustering" in the "Estimator type" dropdown).


Partitioning based
------------------

.. currentmodule:: sktime.clustering.k_means

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKMeans
    TimeSeriesKMeansTslearn

.. currentmodule:: sktime.clustering.k_medoids

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKMedoids

.. currentmodule:: sktime.clustering.k_shapes

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKShapes

Spectral and kernel clustering
------------------------------

.. currentmodule:: sktime.clustering.kernel_k_means

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKernelKMeans

Density-based
-------------

.. currentmodule:: sktime.clustering.dbscan

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesDBSCAN

Graph- or network-based
-----------------------

.. currentmodule:: sktime.clustering.kvisibility

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKvisibility

Spatio-temporal clustering
--------------------------

Spatio-temporal clusterers assume that the time series are, or include,
observations of locations in space.

.. currentmodule:: sktime.clustering.spatio_temporal

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    STDBSCAN

Compose
-------

.. currentmodule:: sktime.clustering.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClustererAsTransformer
    ClustererPipeline
    SklearnClustererPipeline

Base
----

.. currentmodule:: sktime.clustering.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClusterer

.. currentmodule:: sktime.clustering.partitioning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseTimeSeriesLloyds
