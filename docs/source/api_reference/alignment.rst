.. _alignment_ref:

Time series alignment
=====================

The :mod:`sktime.alignment` module contains time series aligners, such as
dynamic time warping aligners.

All time series aligners in ``sktime`` can be listed using the
``sktime.registry.all_estimators`` utility,
using ``estimator_types="aligner"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
`Estimator Search Page <https://www.sktime.net/en/latest/estimator_overview.html>`_
(select "aligner" in the "Estimator type" dropdown).


Naive aligners
--------------

.. currentmodule:: sktime.alignment.naive

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerNaive


Dynamic time warping
--------------------

.. currentmodule:: sktime.alignment.dtw_python

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerDTW
    AlignerDTWfromDist

.. currentmodule:: sktime.alignment.dtaidist

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerDtwDtai

.. currentmodule:: sktime.alignment.dtw_numba

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerDtwNumba

.. currentmodule:: sktime.alignment.lucky

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerLuckyDtw


Edit distance based aligners
----------------------------

.. currentmodule:: sktime.alignment.edit_numba

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerEditNumba
