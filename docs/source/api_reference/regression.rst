.. _regression_ref:

Time series regression
======================

The :mod:`sktime.regression` module contains algorithms and composition tools for time series regression.

All current sktime Regressors can be listed using the ``sktime.registry import
all_estimators`` function.


Composition
-----------

.. currentmodule:: sktime.regression.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ComposableTimeSeriesForestRegressor

Deep learning
-------------

.. currentmodule:: sktime.regression.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CNNRegressor
    TapNetRegressor

Interval-based
--------------

.. currentmodule:: sktime.regression.interval_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesForestRegressor

Kernel-based
------------

.. currentmodule:: sktime.regression.kernel_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RocketRegressor

Base
----

.. currentmodule:: sktime.regression
.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseRegressor

.. currentmodule:: sktime.regression.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepRegressor
