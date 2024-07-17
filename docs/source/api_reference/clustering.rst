
.. _clustering_ref:

Time series clustering
======================

The :mod:`sktime.clustering` module contains algorithms for time series clustering.

All clusterers in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="clusterer"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
`Estimator Search Page <https://www.sktime.net/en/latest/estimator_overview.html>`_
(select "clustering" in the "Estimator type" dropdown).


Clustering models
-----------------

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

.. currentmodule:: sktime.clustering.kernel_k_means

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKernelKMeans

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

    TimeSeriesLloyds
