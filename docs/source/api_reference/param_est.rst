
.. _param_est_ref:

Parameter estimation
====================

The :mod:`sktime.param_est` module contains parameter estimators, e.g., for
seasonality, and utilities for plugging the estimated parameters into other estimators.
For example, seasonality estimators can be combined with any seasonal forecaster
to an auto-seasonality version.

.. automodule:: sktime.param_est
    :no-members:
    :no-inherited-members:

Parameter estimators
--------------------

Composition
~~~~~~~~~~~

.. currentmodule:: sktime.param_est.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ParamFitterPipeline

.. currentmodule:: sktime.param_est.plugin

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PluginParamsForecaster

Naive
~~~~~

.. currentmodule:: sktime.param_est.fixed

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FixedParams

Seasonality estimators
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param_est.seasonality

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SeasonalityACF
    SeasonalityACFqstat
    SeasonalityPeriodogram

Stationarity estimators
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param_est.stationarity

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StationarityADF
    StationarityKPSS
