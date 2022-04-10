
.. _clustering_ref:

Clustering
==========

The :mod:`sktime.clustering` module contains algorithms for time series clustering.

.. automodule:: sktime.clustering
    :no-members:
    :no-inherited-members:


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

Algorithms
----------

.. currentmodule:: sktime.clustering.k_means

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKMeans

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

Metrics
-------

.. currentmodule:: sktime.clustering.metrics.averaging

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    mean_average

.. currentmodule:: sktime.clustering.metrics.medoids

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    medoids
