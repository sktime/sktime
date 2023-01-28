.. _regression_ref:

Time series regression
======================

The :mod:`sktime.regression` module contains algorithms and composition tools for time series regression.

All regressors in ``sktime``can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="regressor"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

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

Distance-based
--------------

.. currentmodule:: sktime.regression.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KNeighborsTimeSeriesRegressor

Dummy
-----

.. currentmodule:: sktime.regression.dummy

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyRegressor

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

.. currentmodule:: sktime.regression.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseRegressor

.. currentmodule:: sktime.regression.deep_learning.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepRegressor
