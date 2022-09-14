.. _transformations_ref:

Time series distances/kernels
=============================

The :mod:`sktime.dists_kernels` module contains pairwise transformers, such as
distances and kernel functions on time series data. It also contains some distances/kernel functions for tabular data.

Distances and kernel functions are treated the same, as they have the same formal signature - that of a "pairwise transformer".

Below, we list separately pairwise transformers for time series, and pairwise transformers for tabular data.

.. automodule:: sktime.dists_kernels
   :no-members:
   :no-inherited-members:

Time series distances/kernels
-----------------------------

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

Dynamic Time Warping Distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels.dtw

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DtwDist

Edit Distances
~~~~~~~~~~~~~~

.. currentmodule:: sktime.dists_kernels.edit_dist

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EditDist

Tabular distances/kernels
-------------------------

.. currentmodule:: sktime.dists_kernels.scipy_dist

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScipyDist
