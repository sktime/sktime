.. _utils_ref:

Utility functions
=================

``sktime`` has a number of modules dedicated to utilities:

* :mod:`sktime.datatypes`, which contains utilities for data format checks and conversion.
* :mod:`sktime.pipeline`, which contains generics for pipeline construction.
* :mod:`sktime.registry`, which contains utilities for estimator and tag search,
  as well as serialization of estimator blueprints as strings.
* :mod:`sktime.utils`, which contains generic utility functions.


Data Format Checking and Conversion
-----------------------------------

:mod:`sktime.datatypes`

.. automodule:: sktime.datatypes
    :no-members:
    :no-inherited-members:

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


Pipeline construction generics
------------------------------

:mod:`sktime.pipeline`

.. automodule:: sktime.pipeline
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.pipeline

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_pipeline
    sklearn_to_sktime


Estimator Search and Retrieval, Estimator Tags
----------------------------------------------

:mod:`sktime.registry`

.. automodule:: sktime.registry
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    all_estimators
    all_tags
    check_tag_is_valid


Estimator Blueprint Serialization
---------------------------------

:mod:`sktime.registry`

.. automodule:: sktime.registry
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    craft
    deps
    imports


Plotting
--------

:mod:`sktime.utils.plotting`

.. automodule:: sktime.utils.plotting
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.utils.plotting

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    plot_series
    plot_lags
    plot_correlations

Estimator Validity Checking
---------------------------

:mod:`sktime.utils.estimator_checks`

.. automodule:: sktime.utils.estimator_checks
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.utils.estimator_checks

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    check_estimator
