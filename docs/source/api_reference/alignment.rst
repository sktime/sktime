.. _alignment_ref:

Time series alignment
=====================

The :mod:`sktime.alignment` module contains time series aligners, such as
dynamic time warping aligners.

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

.. currentmodule:: sktime.alignment.dtw_numba

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerDtwNumba


Edit distance based aligners
----------------------------

.. currentmodule:: sktime.alignment.edit_numba

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AlignerEditNumba
