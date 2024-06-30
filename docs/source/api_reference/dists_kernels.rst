.. _transformations_pairwise_ref:

Time series distances/kernels
=============================

The :mod:`sktime.dists_kernels` module contains pairwise transformers, such as
distances and kernel functions on time series data. It also contains some distances/kernel functions for tabular data.

Distances and kernel functions are treated the same, as they have the same formal signature - that of a "pairwise transformer".

Below, we list separately pairwise transformers for time series, and pairwise transformers for tabular data.

All time series distances and kernels in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="transformer-pairwise-panel"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

Distances and kernels for vector-valued features can be listed using ``estimator_types="transformer-pairwise"``.

Standalone, performant ``numba`` distance functions are available in the :mod:`sktime.distance` module.
These are not wrapped in the ``sktime`` ``BaseObject`` interface and can therefore
be used within other ``numba`` compiled functions for end-to-end compilation.

Time series distances/kernels
-----------------------------

Distances or kernels between time series, following the
pairwise panel transformer interface of ``BasePairwiseTransformerPanel``.

Composition
~~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PwTrafoPanelPipeline

.. currentmodule:: sktime.dists_kernels.algebra

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CombinedDistance

.. currentmodule:: sktime.dists_kernels.indep

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IndepDist

.. currentmodule:: sktime.dists_kernels.compose_tab_to_panel

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AggrDist
    FlatDist

.. currentmodule:: sktime.dists_kernels.compose_from_align

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DistFromAligner

.. currentmodule:: sktime.dists_kernels.dist_to_kern

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KernelFromDist
    DistFromKernel

Simple Time Series Distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple time series distance, including flat/vector distance, bag-of-value distance, or
mean pairwise distance can be obtained by applying ``AggrDist`` or ``FlatDist``
to pairwise distances in ``ScipyDist``. See docstring of ``AggrDist`` and ``FlatDist``.

.. currentmodule:: sktime.dists_kernels.compose_tab_to_panel

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AggrDist
    FlatDist

Dynamic Time Warping Distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels.dtw

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DtwDist
    DtwPythonDist
    DtwDistTslearn
    SoftDtwDistTslearn
    DtwDtaidistUniv
    DtwDtaidistMultiv

.. currentmodule:: sktime.dists_kernels.ctw

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CtwDistTslearn

.. currentmodule:: sktime.dists_kernels.lucky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    LuckyDtwDist

Time warping distances can also be obtained by composing ``DistFromAligner`` with
a time warping aligner, see docstring of ``DistFromAligner``:

.. currentmodule:: sktime.dists_kernels.compose_from_align

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DistFromAligner


Edit Distances
~~~~~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels.edit_dist

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EditDist

.. currentmodule:: sktime.dists_kernels.lcss

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    LcssTslearn

Time Series Kernels
~~~~~~~~~~~~~~~~~~~

Simple time series kernels, including flat/vector kernels, bag-of-value kernels, or
mean pairwise kernels can be obtained by applying ``AggrDist`` or ``FlatDist``
to kernels from ``sklearn.gaussian_process.kernels``.
See docstring of ``AggrDist`` and ``FlatDist``.

.. currentmodule:: sktime.dists_kernels.compose_tab_to_panel

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AggrDist
    FlatDist

Advanced time series kernels that cannot be expressed as aggregates or flat applicates:

.. currentmodule:: sktime.dists_kernels.gak

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GAKernel

.. currentmodule:: sktime.dists_kernels.signature_kernel

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SignatureKernel

Base class
~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BasePairwiseTransformerPanel

Tabular distances/kernels
-------------------------

Distances or kernels between tabular vectors or data frame rows, following the
pairwise transformer interface of ``BasePairwiseTransformer``.

Distance metrics from ``scipy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels.scipy_dist

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScipyDist

Base class
~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BasePairwiseTransformer

Standalone ``numba`` distances
------------------------------

Standalong functions not wrapped in the ``sktime`` ``BaseObject`` interface.
Can be used within other ``numba`` compiled functions for end-to-end compilation.

.. currentmodule:: sktime.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    ddtw_distance
    dtw_distance
    edr_distance
    erp_distance
    euclidean_distance
    lcss_distance
    msm_distance
    pairwise_distance
    squared_distance
    twe_distance
    wddtw_distance
    wdtw_distance
