.. _utils_ref:

Utility functions
=================

``sktime`` has a number of modules dedicated to utilities:

* :mod:`sktime.datatypes`, which contains utilities for data format checks and conversion.
* :mod:`sktime.registry`, which contains utilities for estimator and tag search.
* :mod:`sktime.utils`, which contains genneric utility functions.

.. automodule:: sktime.datatypes
    :no-members:
    :no-inherited-members:

.. automodule:: sktime.registry
    :no-members:
    :no-inherited-members:

.. automodule:: sktime.utils
    :no-members:
    :no-inherited-members:

Data Format Checking and Conversion
-----------------------------------

.. currentmodule:: sktime.datatypes

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    convert_to
    convert
    check_raise
    check_is_mtype
    check_is_scitype
    mtype
    scitype
    mtype_to_scitype
    scitype_to_mtype

Estimator Search and Retrieval, Estimator Tags
----------------------------------------------

.. currentmodule:: sktime.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    all_estimators
    all_tags
    check_tag_is_valid

Plotting
--------

.. currentmodule:: sktime.utils.plotting

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    plot_series
    plot_lags
    plot_correlations

Estimator Validity Checking
---------------------------

.. currentmodule:: sktime.utils.estimator_checks

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    check_estimator
